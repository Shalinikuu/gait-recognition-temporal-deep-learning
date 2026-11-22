"""
demo.py
-------
Do walking sequences (silhouette frames ke folders) ko compare karta hai.
Cosine similarity jitni zyada (1 ke kareeb), utna zyada chance ki same person hai.

Run:
  python src/demo.py --ckpt checkpoints/lstm/best.pt --seq1 "path/folder1" --seq2 "path/folder2"
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import GaitNet
from preprocess import IMG_EXT, load_sequence, natural_key


def folder_to_tensor(folder, seq_len):
    files = sorted([f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXT],
                   key=natural_key)
    arr = load_sequence([os.path.join(folder, f) for f in files])
    if arr is None:
        raise RuntimeError(f"Koi valid frame nahi mila: {folder}")
    idx = np.linspace(0, len(arr) - 1, seq_len).round().astype(int)
    x = torch.from_numpy(arr[idx]).float() / 255.0
    return x.unsqueeze(1).unsqueeze(0)  # (1, T, 1, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq1", required=True)
    ap.add_argument("--seq2", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = GaitNet(a["model"], num_classes=ck["num_classes"], emb_dim=a["emb_dim"])
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        e1 = F.normalize(model(folder_to_tensor(args.seq1, a["seq_len"]))[1], dim=1)
        e2 = F.normalize(model(folder_to_tensor(args.seq2, a["seq_len"]))[1], dim=1)
    print("Cosine similarity:", round((e1 * e2).sum().item(), 4))


if __name__ == "__main__":
    main()