"""Leakage audit: find val/test images that are near-duplicates of a TRAIN image (any class, any source).

    python ml/audit_leakage.py --root data/processed            # report only
    python ml/audit_leakage.py --root data/processed --delete   # also remove the offending val/test images

Why: prepare_data.py splits by group inside one source folder, but two different sources can contain the
same photo (re-uploaded datasets) and it would then land on both sides. This catches that.
A match whose class differs from the train image's class is flagged as LABEL CONFLICT: one of the two is mislabeled.
Run the audit BEFORE trusting any test accuracy. Re-run train.py after --delete.
"""
import argparse, os
import numpy as np
from prepare_data import dhash, hamming_to_all

def collect(root, split):
    items = []
    base = os.path.join(root, split)
    for cid in sorted(os.listdir(base)):
        d = os.path.join(base, cid)
        if os.path.isdir(d):
            items += [(cid, os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.lower().endswith(".jpg")]
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--max-ham", type=int, default=6)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()

    train = collect(a.root, "train")
    print(f"hashing {len(train)} train images ...")
    train_h = np.array([dhash(p) for _, p in train], dtype=np.uint64)

    for split in ("val", "test"):
        items = collect(a.root, split)
        bad = []
        for cid, p in items:
            d = hamming_to_all(dhash(p), train_h)
            j = int(d.argmin())
            if d[j] <= a.max_ham:
                bad.append((cid, p, train[j][0], train[j][1], int(d[j])))
        pct = 100 * len(bad) / max(len(items), 1)
        print(f"\n[{split}] {len(bad)} of {len(items)} images ({pct:.1f}%) have a near-duplicate in train")
        for cid, p, tcid, tp, dist in bad[:15]:
            tag = "  <-- LABEL CONFLICT" if cid != tcid else ""
            print(f"   {p}  ~  {tp}  (dist {dist}){tag}")
        if len(bad) > 15:
            print(f"   ... and {len(bad) - 15} more")
        if a.delete:
            for _, p, *_ in bad:
                os.remove(p)
            print(f"   deleted {len(bad)} files from {split}")

if __name__ == "__main__":
    main()
