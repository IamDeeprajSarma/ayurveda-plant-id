"""Evaluate a trained checkpoint on any folder laid out as <root>/<class_id>/*.jpg.

    python ml/evaluate.py --ckpt models/efficientnet_b0_best.pt --root data/processed/test --name test
    python ml/evaluate.py --ckpt models/efficientnet_b0_best.pt --root data/field_test    --name field

Writes reports/<name>_<arch>/: report.txt (top-1, top-3, per-class precision/recall/F1),
confusion_matrix.png, predictions.csv (every image, with top-3), thresholds.csv.
Folders whose name is NOT a trained class are treated as 'unknown plants': they are left out of
accuracy, and thresholds.csv shows how often the model confidently (wrongly) names them.
That table is how you choose the confidence cut-off for the app's 'Not sure' state.
"""
import argparse, csv, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import common

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--name", default="eval"); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    a = ap.parse_args()
    device = common.pick_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu")
    classes, arch, img = ck["classes"], ck["arch"], ck["img"]
    model, _ = common.build_model(arch, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()

    ds = common.EvalDataset(a.root, classes, common.eval_tf(img))
    if ds.unknown_folders:
        print("Treating as UNKNOWN plants (not trained classes):", ds.unknown_folders)
    dl = DataLoader(ds, a.bs, shuffle=False, num_workers=a.workers)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); labels += y.tolist()
    P, y = np.concatenate(probs), np.array(labels)
    paths = [p for p, _ in ds.items]
    top = np.argsort(-P, axis=1)[:, :3]
    known = y >= 0
    out_dir = os.path.join("reports", f"{a.name}_{arch}"); os.makedirs(out_dir, exist_ok=True)

    lines = [f"checkpoint={a.ckpt}  root={a.root}  images={len(y)}  known={known.sum()}  unknown={(~known).sum()}"]
    if known.any():
        top1 = (top[known, 0] == y[known]).mean(); top3 = (top[known] == y[known][:, None]).any(1).mean()
        present = sorted(set(y[known].tolist()))
        lines += [f"top-1 accuracy: {top1:.4f}", f"top-3 accuracy: {top3:.4f}", "",
                  classification_report(y[known], top[known, 0], labels=present,
                                        target_names=[classes[i] for i in present], zero_division=0)]
        cm = confusion_matrix(y[known], top[known, 0], labels=present).astype(float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(max(6, len(present) * 0.45), max(5, len(present) * 0.4)))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
        ax.set_xticklabels([classes[i] for i in present], rotation=90, fontsize=7)
        ax.set_yticklabels([classes[i] for i in present], fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{a.name}: row-normalised confusion matrix")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "true", "pred1", "conf1", "pred2", "conf2", "pred3", "conf3"])
        for i, p in enumerate(paths):
            row = [p, classes[y[i]] if y[i] >= 0 else "UNKNOWN"]
            for j in top[i]:
                row += [classes[j], f"{P[i, j]:.3f}"]
            w.writerow(row)

    maxp = P.max(1)
    with open(os.path.join(out_dir, "thresholds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "known_coverage", "known_acc_when_answered", "unknown_wrongly_accepted"])
        for t in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            acc_mask = known & (maxp >= t)
            cov = acc_mask.sum() / max(known.sum(), 1)
            acc = (top[acc_mask, 0] == y[acc_mask]).mean() if acc_mask.any() else float("nan")
            unk = ((~known) & (maxp >= t)).sum() / max((~known).sum(), 1) if (~known).any() else float("nan")
            w.writerow([t, f"{cov:.3f}", f"{acc:.3f}", f"{unk:.3f}"])
    print(f"\nreports written to {out_dir}/")

if __name__ == "__main__":
    main()
