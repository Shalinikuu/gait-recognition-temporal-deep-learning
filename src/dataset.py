"""
dataset.py
----------
Preprocessed .npy sequences ko PyTorch model ke liye ready karta hai.

Har item:
    x    : (seq_len, 1, 64, 64) float tensor, values 0..1
    y    : label (int)
    cond : 'nm' / 'bg' / 'cl'
    seq  : sequence number
    view : camera angle (0..180)
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class GaitDataset(Dataset):
    def __init__(self, df, processed_dir, seq_len=30, train=True, label_map=None):
        """
        df          : index.csv ka (filtered) DataFrame
        label_map   : {subject_id: class_index}. Training me chahiye.
                      Test me None do -> label = subject id (retrieval ke liye).
        train=True  : random clip lete hain (augmentation jaisa)
        train=False : frames evenly spread (deterministic)
        """
        self.df = df.reset_index(drop=True)
        self.dir = processed_dir
        self.seq_len = seq_len
        self.train = train
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def _pick(self, n):
        L = self.seq_len
        if self.train:
            if n >= L:
                s = random.randint(0, n - L)
                return np.arange(s, s + L)
            return np.arange(L) % n  # chhoti sequence ko repeat kar do
        return np.linspace(0, n - 1, L).round().astype(int)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = np.load(os.path.join(self.dir, row["file"]), mmap_mode="r")
        idx = self._pick(arr.shape[0])
        x = torch.from_numpy(np.ascontiguousarray(arr[idx])).float() / 255.0
        x = x.unsqueeze(1)  # (T, 1, H, W)
        subject = int(row["subject"])
        y = self.label_map[subject] if self.label_map is not None else subject
        return x, y, row["cond"], int(row["seq"]), int(row["view"])