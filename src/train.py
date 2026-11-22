"""
train.py
--------
Split:
  Train identities : subject 1..n_train (default 74)
  Train data       : in subjects ki sequences, sirf (nm-06, bg-02, cl-02) ko chhodke
  Val data         : wahi 3 held-out sequences (same subjects)  -> sirf training ko monitor karne ke liye
  Test identities  : 75..124 -> evaluate.py me use hote hain (model ne kabhi nahi dekhe)

Examples:
  # chhota test (Mac/CPU pe bhi chalega)
  python src/train.py --model lstm --debug --epochs 2 --batch_size 8

  # GPU pe full run
  python src/train.py --model lstm --epochs 60 --batch_size 64 --num_workers 4
  python src/train.py --model tcn ...
  python src/train.py --model transformer ...

  # beech me ruk gaya? dobara:
  python src/train.py --model lstm --epochs 60 --resume checkpoints/lstm/last.pt
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GaitDataset
from models import GaitNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y, *_ in tqdm(loader, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "tcn", "transformer"])
    ap.add_argument("--index", default="data/processed/index.csv")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=74, help="subjects 1..n_train train ke liye")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--debug", action="store_true", help="sirf 10 subjects, jaldi test ke liye")
    args = ap.parse_args()
    if args.debug:
        args.n_train = 10

    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    print("Device:", device, "| AMP:", use_amp)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---------- data ----------
    df = pd.read_csv(args.index)
    df = df[df.subject <= args.n_train]
    held = ((df.cond == "nm") & (df.seq == 6)) | ((df.cond == "bg") & (df.seq == 2)) | ((df.cond == "cl") & (df.seq == 2))
    tr_df, va_df = df[~held], df[held]
    label_map = {s: i for i, s in enumerate(sorted(df.subject.unique()))}
    num_classes = len(label_map)
    print(f"Train seqs: {len(tr_df)} | Val seqs: {len(va_df)} | classes: {num_classes}")

    pin = device.type == "cuda"
    tr_loader = DataLoader(GaitDataset(tr_df, args.processed, args.seq_len, True, label_map),
                           batch_size=args.batch_size, shuffle=True, drop_last=True,
                           num_workers=args.num_workers, pin_memory=pin)
    va_loader = DataLoader(GaitDataset(va_df, args.processed, args.seq_len, False, label_map),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin)

    # ---------- model ----------
    model = GaitNet(args.model, num_classes=num_classes, emb_dim=args.emb_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    start_epoch, best_val = 0, 0.0

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best_val"]
        print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(out_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "sec"])

    # ---------- loop ----------
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tl, ta = run_epoch(model, tr_loader, criterion, device, optimizer, scaler, use_amp)
        vl, va = run_epoch(model, va_loader, criterion, device, None, None, use_amp)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        sec = time.time() - t0
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | train loss {tl:.4f} acc {ta:.2f} | val loss {vl:.4f} acc {va:.2f} | {sec:.0f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{tl:.4f}", f"{ta:.2f}", f"{vl:.4f}", f"{va:.2f}", f"{lr:.6f}", f"{sec:.0f}"])

        if va > best_val:
            best_val = va
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val": best_val,
            "args": vars(args), "num_classes": num_classes,
        }
        torch.save(state, os.path.join(out_dir, "last.pt"))  # har epoch
        if va >= best_val:
            torch.save(state, os.path.join(out_dir, "best.pt"))

    print("Training complete. Checkpoints:", out_dir)


if __name__ == "__main__":
    main()