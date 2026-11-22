"""
evaluate.py
-----------
Test subjects (default 75..124) pe model ko evaluate karta hai. Ye subjects training me nahi the.

Protocol (CASIA-B ka standard):
  Gallery : nm-01..nm-04 (har view)
  Probe   : NM = nm-05,06 | BG = bg-01,02 | CL = cl-01,02
  Har probe view ke liye: gallery ke baaki sab views se match karo (same view chhodke),
  Rank-1 accuracy nikalo aur average lo.
  Matching = embeddings ki cosine similarity, nearest neighbour.

Run:
  python src/evaluate.py --ckpt checkpoints/lstm/best.pt
  python src/evaluate.py --ckpt checkpoints/lstm/best.pt --test_from 11 --test_to 20   # debug run ke baad
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GaitDataset
from models import GaitNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, subj, conds, seqs, views = [], [], [], [], []
    for x, y, cond, seq, view in tqdm(loader, desc="Embeddings"):
        _, e = model(x.to(device))
        embs.append(F.normalize(e, dim=1).cpu().numpy())
        subj.append(y.numpy())
        conds += list(cond)
        seqs.append(seq.numpy())
        views.append(view.numpy())
    return (np.concatenate(embs), np.concatenate(subj), np.array(conds),
            np.concatenate(seqs), np.concatenate(views))


def rank1_per_view(E, subj, conds, seqs, views, probe_mask, gallery_mask):
    """Har probe view ke liye rank-1 accuracy (%), gallery ke baaki views par average."""
    all_views = sorted(np.unique(views))
    result = []
    for pv in all_views:
        accs = []
        p_idx = np.where(probe_mask & (views == pv))[0]
        if len(p_idx) == 0:
            result.append(np.nan)
            continue
        for gv in all_views:
            if gv == pv:
                continue
            g_idx = np.where(gallery_mask & (views == gv))[0]
            if len(g_idx) == 0:
                continue
            sim = E[p_idx] @ E[g_idx].T                  # cosine similarity
            pred = subj[g_idx][sim.argmax(axis=1)]       # sabse milta-julta gallery sample
            accs.append((pred == subj[p_idx]).mean())
        result.append(100.0 * np.mean(accs) if accs else np.nan)
    return all_views, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", default="data/processed/index.csv")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--test_from", type=int, default=75)
    ap.add_argument("--test_to", type=int, default=124)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="results csv path (default: checkpoint ke saath)")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location=device)
    a = ck["args"]
    model = GaitNet(a["model"], num_classes=ck["num_classes"], emb_dim=a["emb_dim"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"Model: {a['model']} | trained epochs: {ck['epoch'] + 1} | device: {device}")

    df = pd.read_csv(args.index)
    df = df[(df.subject >= args.test_from) & (df.subject <= args.test_to)]
    print(f"Test subjects: {df.subject.nunique()} | sequences: {len(df)}")

    loader = DataLoader(GaitDataset(df, args.processed, a["seq_len"], train=False, label_map=None),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    E, subj, conds, seqs, views = extract_embeddings(model, loader, device)

    gallery = (conds == "nm") & (seqs <= 4)
    probes = {
        "NM": (conds == "nm") & (seqs >= 5),
        "BG": (conds == "bg") & (seqs <= 2),
        "CL": (conds == "cl") & (seqs <= 2),
    }

    rows = {}
    for name, mask in probes.items():
        v, r = rank1_per_view(E, subj, conds, seqs, views, mask, gallery)
        rows[name] = r
    table = pd.DataFrame(rows, index=[f"{x}deg" for x in v]).T
    table["Mean"] = table.mean(axis=1)

    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    print("\nRank-1 accuracy (%) per probe view:")
    print(table.to_string())

    out = args.out or os.path.join(os.path.dirname(args.ckpt), "results.csv")
    table.to_csv(out)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()