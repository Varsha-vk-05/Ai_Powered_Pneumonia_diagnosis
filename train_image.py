import argparse
import os
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from datasets.image_dataset import ImageCsvDataset
import os
os.environ.setdefault("WANDB_SILENT", "true")
try:
    import wandb
except Exception:
    wandb = None  # optional


def build_dataloaders(train_csv: str, val_csv: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, int, dict]:
	train_tfms = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	val_tfms = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	train_ds = ImageCsvDataset(train_csv, transform=train_tfms)
	val_ds = ImageCsvDataset(val_csv, transform=val_tfms, class_to_index=train_ds.class_to_index)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader, len(train_ds.class_to_index), train_ds.class_to_index


def build_model(num_classes: int) -> nn.Module:
	model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
	in_features = model.fc.in_features
	model.fc = nn.Linear(in_features, num_classes)
	return model


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
	train_loader, val_loader, num_classes, class_to_index = build_dataloaders(
		args.train_csv, args.val_csv, args.batch_size, args.num_workers
	)
	model = build_model(num_classes).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
			"model": "resnet18",
		})
	for epoch in range(1, args.epochs + 1):
		model.train()
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
	print(f"Best val_acc={best_acc:.4f}")
	if args.wandb and wandb is not None:
		wandb.finish()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_csv", type=str, required=True)
	parser.add_argument("--val_csv", type=str, required=True)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--out_dir", type=str, default="runs/images")
	parser.add_argument("--wandb", action="store_true")
	parser.add_argument("--wandb_project", type=str, default=None)
	args = parser.parse_args()
	train(args)


if __name__ == "__main__":
	main()

