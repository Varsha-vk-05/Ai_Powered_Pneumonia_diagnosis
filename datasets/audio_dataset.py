import os
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import cv2
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


class AudioCsvDataset(Dataset):
	# class-level attribute annotations to help static checkers (Pylance/pyright)
	df: 'pd.DataFrame'
	sample_rate: int
	target_len: int
	n_mels: int
	fmin: int
	fmax: Optional[int]
	to_3ch: bool
	img_size: int
	transform: Optional[Callable]
	augment: bool
	classes: List[str]
	class_to_index: Dict[str, int]
	index_to_class: Dict[int, str]

	def __init__(
		self,
		csv_path: str,
		sample_rate: int = 16000,
		duration_s: float = 5.0,
		n_mels: int = 64,
		fmin: int = 20,
		fmax: Optional[int] = None,
		to_3ch: bool = False,
		transform: Optional[Callable] = None,
		class_to_index: Optional[Dict[str, int]] = None,
	):
		self.df = pd.read_csv(csv_path)
		if "filepath" not in self.df.columns or "label" not in self.df.columns:
			raise ValueError("CSV must have columns: filepath,label")
		self.sample_rate = sample_rate
		self.target_len = int(duration_s * sample_rate)
		self.n_mels = n_mels
		self.fmin = fmin
		self.fmax = fmax
		self.to_3ch = bool(to_3ch)
		# if converting to 3-channel images, target image size for backbone
		self.img_size = 224
		self.transform = transform
		self.augment = False
		self.classes = sorted(self.df["label"].astype(str).unique().tolist())
		self.class_to_index = class_to_index or {c: i for i, c in enumerate(self.classes)}
		self.index_to_class = {i: c for c, i in self.class_to_index.items()}

	def __len__(self) -> int:
		return len(self.df)

	def _load_audio_fixed(self, path: str) -> np.ndarray:
		x, sr = librosa.load(path, sr=self.sample_rate, mono=True)
		if x.shape[0] < self.target_len:
			pad = self.target_len - x.shape[0]
			x = np.pad(x, (0, pad))
		else:
			x = x[: self.target_len]
		return x

	def _wav_to_logmelspec(self, x: np.ndarray) -> np.ndarray:
		mel = librosa.feature.melspectrogram(
			y=x,
			sr=self.sample_rate,
			n_mels=self.n_mels,
			fmin=self.fmin,
			fmax=self.fmax,
			power=2.0,
		)
		logmel = librosa.power_to_db(mel, ref=np.max)
		# Per-sample normalize
		m = logmel.mean()
		s = logmel.std() + 1e-6
		logmel = (logmel - m) / s
		return logmel.astype(np.float32)

	def enable_augment(self, on: bool = True):
		"""Enable simple audio augmentations applied online in __getitem__."""
		self.augment = bool(on)

	def _apply_augment(self, x: np.ndarray) -> np.ndarray:
		# time shift
		if np.random.rand() < 0.5:
			shift = int(self.sample_rate * 0.1 * (np.random.rand() - 0.5))
			if shift > 0:
				x = np.pad(x, (shift, 0))[: self.target_len]
			elif shift < 0:
				x = np.pad(x, (0, -shift))[-shift: self.target_len - shift]
		# additive noise
		if np.random.rand() < 0.5:
			noise_amp = 0.001 * np.random.rand()
			x = x + noise_amp * np.random.randn(*x.shape)
		# random gain
		if np.random.rand() < 0.3:
			x = x * (0.8 + 0.4 * np.random.rand())
		return x

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		row = self.df.iloc[idx]
		path = str(row["filepath"])
		label_str = str(row["label"])
		label = self.class_to_index[label_str]
		x = self._load_audio_fixed(path)
		if self.augment:
			x = self._apply_augment(x)
		feat = self._wav_to_logmelspec(x)  # [n_mels, time]
		if self.transform is not None:
			feat = self.transform(feat)
		# to tensor [1, n_mels, time]
		# If requested, resize mel-spec to square image and replicate channels
		if self.to_3ch:
			# feat shape: [n_mels, time]
			feat_img = cv2.resize(feat.astype(np.float32), (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
			feat_t = torch.from_numpy(feat_img).unsqueeze(0)
			feat_t = feat_t.repeat(3, 1, 1)
		else:
			feat_t = torch.from_numpy(feat).unsqueeze(0)
		return feat_t, label

