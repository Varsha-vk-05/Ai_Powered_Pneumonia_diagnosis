import os
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageCsvDataset(Dataset):
	def __init__(
		self,
		csv_path: str,
		transform: Optional[Callable] = None,
		class_to_index: Optional[Dict[str, int]] = None,
	):
		self.df = pd.read_csv(csv_path)
		if "filepath" not in self.df.columns or "label" not in self.df.columns:
			raise ValueError("CSV must have columns: filepath,label")
		self.transform = transform
		self.classes = sorted(self.df["label"].astype(str).unique().tolist())
		if class_to_index is None:
			self.class_to_index = {c: i for i, c in enumerate(self.classes)}
		else:
			self.class_to_index = class_to_index
		self.index_to_class = {i: c for c, i in self.class_to_index.items()}

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		row = self.df.iloc[idx]
		path = str(row["filepath"])
		label_str = str(row["label"])
		label = self.class_to_index[label_str]
		img = Image.open(path).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		return img, label

