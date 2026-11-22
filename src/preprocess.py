"""
preprocess.py
-------------
CASIA-B silhouette frames ko padhta hai, har frame ko saaf karta hai
(person ke around crop -> 64x64 resize) aur har sequence ko ek .npy file
me save karta hai. Saath me index.csv banata hai jisme har sequence ki detail hoti hai.

Run:
    python src/preprocess.py --root "/path/to/casia-b" --out data/processed

Filename/folder pattern jo expect karta hai (standard CASIA-B):
    001/nm-01/000/001-nm-01-000-001.png
    (subject / condition-sequence / view / frame)
Agar tumhare dataset ka naming alag hai to PATTERN badalna padega.
"""
import argparse
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# subject(3 digit) - cond(nm/bg/cl) - seq(2 digit) - view(3 digit)
PATTERN = re.compile(r"(\d{3})[-/_\\](nm|bg|cl)[-/_\\](\d{2})[-/_\\](\d{3})", re.IGNORECASE)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(s):
    """'frame2' ko 'frame10' se pehle rakhne ke liye natural sorting."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def process_frame(img, size=64):
    """
    1) binary silhouette banao
    2) person ke bounding box tak crop karo
    3) height = size karo (aspect ratio same rakhke)
    4) width ko center me rakhke size x size canvas me daalo
    """
    img = ((img > 127).astype(np.uint8)) * 255
    ys, xs = np.where(img > 0)
    if len(ys) == 0:  # khali frame
        return None
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape
    new_w = max(1, int(round(w * size / h)))
    img = cv2.resize(img, (new_w, size), interpolation=cv2.INTER_AREA)

    out = np.zeros((size, size), dtype=np.uint8)
    if new_w >= size:
        s = (new_w - size) // 2
        out = img[:, s:s + size]
    else:
        s = (size - new_w) // 2
        out[:, s:s + new_w] = img
    return out


def load_sequence(paths, size=64):
    """Image paths ki list -> numpy array (T, size, size) uint8."""
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        f = process_frame(img, size)
        if f is not None:
            frames.append(f)
    if len(frames) == 0:
        return None
    return np.stack(frames)


def scan_dataset(root):
    """Dataset folder scan karke frames ko (subject, cond, seq, view) ke hisaab se group karta hai."""
    groups = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root)
            m = PATTERN.search(rel)
            if m is None:
                continue
            key = (int(m.group(1)), m.group(2).lower(), int(m.group(3)), int(m.group(4)))
            groups.setdefault(key, []).append(full)
    for k in groups:
        groups[k].sort(key=lambda p: natural_key(os.path.basename(p)))
    return groups


def worker(job):
    key, paths, out_dir, size, min_frames = job
    subject, cond, seq, view = key
    arr = load_sequence(paths, size)
    if arr is None or len(arr) < min_frames:
        return None
    fname = f"{subject:03d}_{cond}_{seq:02d}_{view:03d}.npy"
    np.save(os.path.join(out_dir, fname), arr)
    return dict(subject=subject, cond=cond, seq=seq, view=view, n_frames=len(arr), file=fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CASIA-B dataset folder")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--min_frames", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Scanning dataset...")
    groups = scan_dataset(args.root)
    print(f"Sequences mile: {len(groups)}")
    if len(groups) == 0:
        print("Koi sequence nahi mili. PATTERN ya --root check karo (inspect_dataset.py ka output bhejo).")
        return

    jobs = [(k, v, args.out, args.size, args.min_frames) for k, v in groups.items()]
    rows = []
    with Pool(args.workers) as pool:
        for r in tqdm(pool.imap_unordered(worker, jobs), total=len(jobs)):
            if r is not None:
                rows.append(r)

    df = pd.DataFrame(rows).sort_values(["subject", "cond", "seq", "view"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.out, "index.csv"), index=False)

    print("\nDone. Saved:", os.path.join(args.out, "index.csv"))
    print("Subjects :", df.subject.nunique())
    print("Conds    :", sorted(df.cond.unique()))
    print("Views    :", sorted(df.view.unique()))
    print("Frames/seq (min/mean/max):", df.n_frames.min(), round(df.n_frames.mean(), 1), df.n_frames.max())


if __name__ == "__main__":
    main()