import argparse
import os
from typing import Tuple, cast
import os
os.environ.setdefault("WANDB_SILENT", "true")
try:
	import wandb
except Exception:
	wandb = None

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.audio_dataset import AudioCsvDataset
from torchvision import models, transforms


class SmallAudioCNN(nn.Module):
	def __init__(self, num_classes: int):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Linear(64, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = x.flatten(1)
		# small dropout before classifier
		x = nn.functional.dropout(x, p=0.25, training=self.training)
		return self.classifier(x)


def build_dataloaders(train_csv: str, val_csv: str, batch_size: int, num_workers: int, to_3ch: bool = False) -> Tuple[DataLoader, DataLoader, int, dict]:
	train_ds = AudioCsvDataset(train_csv, to_3ch=to_3ch)
	val_ds = AudioCsvDataset(val_csv, class_to_index=train_ds.class_to_index, to_3ch=to_3ch)
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader, len(train_ds.class_to_index), train_ds.class_to_index


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
	model.eval()
	correct = 0
	total = 0
	loss_total = 0.0
	crit = nn.CrossEntropyLoss()
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			logits = model(x)
			loss = crit(logits, y)
			loss_total += loss.item() * x.size(0)
			pred = logits.argmax(dim=1)
			correct += (pred == y).sum().item()
			total += y.size(0)
	avg_loss = loss_total / max(1, total)
	acc = correct / max(1, total)
	return avg_loss, acc


def train(args: argparse.Namespace) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_3ch = bool(args.model and args.model.startswith('resnet'))
	train_loader, val_loader, num_classes, class_to_index = build_dataloaders(
		args.train_csv, args.val_csv, args.batch_size, args.num_workers, to_3ch=use_3ch
	)
	# cast datasets so static type checkers know the concrete dataset type
	train_dataset = cast(AudioCsvDataset, train_loader.dataset)
	val_dataset = cast(AudioCsvDataset, val_loader.dataset)
	if args.model and args.model.startswith('resnet'):
		# Transfer learning from torchvision ResNet on 3-channel mel-spec images
		backbone = getattr(models, args.model)(weights=models.ResNet18_Weights.IMAGENET1K_V1)
		in_features = backbone.fc.in_features
		backbone.fc = torch.nn.Identity()
		model = torch.nn.Sequential(
			backbone,
			torch.nn.Flatten(1),
			torch.nn.Dropout(0.2),
			torch.nn.Linear(in_features, num_classes)
		).to(device)
		# If requested, freeze backbone
		if getattr(args, 'freeze_backbone', False):
			for p in backbone.parameters():
				p.requires_grad = False
	else:
		model = SmallAudioCNN(num_classes).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
	# optional scheduler
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
	# Automatic class weights (by default enabled). Disable with --no_class_weights
	if not getattr(args, 'no_class_weights', False):
		try:
			import numpy as np
			# map labels to indices using dataset mapping
			labels = np.asarray(train_dataset.df["label"].astype(str).map(train_dataset.class_to_index).values, dtype=np.int64)
			counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
			total = counts.sum() if counts.sum() > 0 else 1.0
			# inverse frequency weights, normalized to mean=1
			weights = total / (counts + 1e-6)
			weights = weights / (weights.mean() + 1e-9)
			weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
			crit = nn.CrossEntropyLoss(weight=weights_tensor)
			print("Using class weights:", weights)
		except Exception as e:
			print("Failed to compute class weights, falling back to uniform loss:", e)
			crit = nn.CrossEntropyLoss()
	else:
		crit = nn.CrossEntropyLoss()
	os.makedirs(args.out_dir, exist_ok=True)
	best_acc = 0.0
	if args.wandb and wandb is not None:
		mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
		wandb.init(project=args.wandb_project or "medical-trainer", mode=mode, config={
			"epochs": args.epochs,
			"batch_size": args.batch_size,
			"lr": args.lr,
			"dataset_csvs": {"train": args.train_csv, "val": args.val_csv},
			"model": args.model or "small_audio_cnn",
		})
	for epoch in range(1, args.epochs + 1):
		model.train()
		# enable augmentation if requested (use typed dataset)
		if getattr(args, 'augment', False):
			train_dataset.enable_augment(True)
		for x, y in train_loader:
			x = x.to(device)
			y = y.to(device)
			optimizer.zero_grad(set_to_none=True)
			logits = model(x)
			loss = crit(logits, y)
			loss.backward()
			optimizer.step()
		val_loss, val_acc = evaluate(model, val_loader, device)
		print(f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
		if args.wandb and wandb is not None:
			wandb.log({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})
		if val_acc > best_acc:
			best_acc = val_acc
			save_path = os.path.join(args.out_dir, "best.pt")
			torch.save({
				"state_dict": model.state_dict(),
				"classes": {i: c for c, i in class_to_index.items()},
			}, save_path)
			if args.wandb and wandb is not None:
				wandb.save(save_path)
		# step scheduler
		scheduler.step()
	print(f"Best val_acc={best_acc:.4f}")
	if args.wandb and wandb is not None:
		wandb.finish()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_csv", type=str, required=True)
	parser.add_argument("--val_csv", type=str, required=True)
	parser.add_argument("--epochs", type=int, default=15)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--out_dir", type=str, default="runs/audio")
	parser.add_argument("--wandb", action="store_true")
	parser.add_argument("--wandb_project", type=str, default=None)
	parser.add_argument("--model", type=str, default=None, help="Model to use: 'resnet18' to use pretrained ResNet on mel-spec images")
	parser.add_argument("--freeze_backbone", action="store_true", help="When using an image backbone, freeze the pretrained layers")
	parser.add_argument("--augment", action="store_true", help="Enable audio augmentations during training")
	parser.add_argument("--no_class_weights", action="store_true", help="Disable automatic class weights computation")
	args = parser.parse_args()
	train(args)


if __name__ == "__main__":
	main()

