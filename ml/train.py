"""Fine-tune an ImageNet-pretrained CNN on data/processed.

Smoke test first (1-2 minutes):
    python ml/train.py --arch efficientnet_b0 --epochs 1 --epoch-size 200
Real run:
    python ml/train.py --arch efficientnet_b0 --epochs 12
Architectures: resnet50, vgg16, inception_v3, efficientnet_b0, mobilenet_v3_large
(vgg16 is very heavy on a laptop - use Kaggle/Colab for it.)

Phase 1 trains only the new classifier head (--freeze-epochs), phase 2 fine-tunes everything.
Classes are balanced with a weighted sampler. The best epoch by validation macro-F1 is kept.
"""
import argparse, csv, json, os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

import common

def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n, ys, ps, top3 = 0.0, 0, [], [], 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y); n += len(y)
            k = min(3, out.shape[1])
            top3 += (out.topk(k, dim=1).indices == y[:, None]).any(dim=1).sum().item()
            ys += y.tolist(); ps += out.argmax(1).tolist()
    return dict(loss=total_loss / max(n, 1), acc=accuracy_score(ys, ps),
                f1=f1_score(ys, ps, average="macro", zero_division=0), top3=top3 / max(n, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed"); ap.add_argument("--out", default="models")
    ap.add_argument("--arch", default="efficientnet_b0", choices=list(common.ARCHS))
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--epoch-size", type=int, default=3000, help="images drawn per epoch (speeds up CPU runs)")
    ap.add_argument("--patience", type=int, default=4); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = common.pick_device(a.device); img = common.ARCHS[a.arch]
    print(f"device={device}  arch={a.arch}  input={img}px")

    tr = datasets.ImageFolder(os.path.join(a.data, "train"), common.train_tf(img))
    va = datasets.ImageFolder(os.path.join(a.data, "val"), common.eval_tf(img))
    te = datasets.ImageFolder(os.path.join(a.data, "test"), common.eval_tf(img))
    assert tr.classes == va.classes == te.classes, "train/val/test class folders differ"
    classes = tr.classes
    json.dump(classes, open(os.path.join(a.out, "classes.json"), "w"), indent=1)

    counts = np.bincount(tr.targets, minlength=len(classes))
    weights = 1.0 / counts[np.array(tr.targets)]
    sampler = WeightedRandomSampler(weights, num_samples=min(len(tr), a.epoch_size), replacement=True)
    kw = dict(num_workers=a.workers, persistent_workers=a.workers > 0)
    tr_dl = DataLoader(tr, a.bs, sampler=sampler, **kw)
    va_dl = DataLoader(va, a.bs, shuffle=False, **kw)
    te_dl = DataLoader(te, a.bs, shuffle=False, **kw)

    model, head = common.build_model(a.arch, len(classes), pretrained=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    ckpt_path = os.path.join(a.out, f"{a.arch}_best.pt")
    best_f1, bad, hist, t0 = -1.0, 0, [], time.time()
    opt = sched = None
    for epoch in range(a.epochs):
        if epoch == 0 and a.freeze_epochs > 0:
            common.set_backbone_trainable(model, head, False)
            opt = torch.optim.AdamW(head.parameters(), lr=a.lr_head, weight_decay=1e-4); phase = "head"
        if epoch == a.freeze_epochs:
            common.set_backbone_trainable(model, head, True)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs - a.freeze_epochs))
            phase = "full"
        t = run_epoch(model, tr_dl, device, criterion, opt)
        v = run_epoch(model, va_dl, device, criterion)
        if sched is not None and phase == "full":
            sched.step()
        hist.append(dict(epoch=epoch + 1, phase=phase, train_loss=t["loss"], train_acc=t["acc"],
                         val_loss=v["loss"], val_acc=v["acc"], val_f1=v["f1"], val_top3=v["top3"]))
        print(f"epoch {epoch+1:>2}/{a.epochs} [{phase}] train loss {t['loss']:.3f} acc {t['acc']:.3f} | "
              f"val loss {v['loss']:.3f} acc {v['acc']:.3f} macroF1 {v['f1']:.3f} top3 {v['top3']:.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        if v["f1"] > best_f1:
            best_f1, bad = v["f1"], 0
            torch.save(dict(state_dict=model.state_dict(), classes=classes, arch=a.arch, img=img), ckpt_path)
        elif phase == "full":
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    with open(os.path.join(a.out, f"{a.arch}_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test = run_epoch(model, te_dl, device, criterion)
    res = dict(arch=a.arch, classes=len(classes), best_val_macro_f1=best_f1, test_top1=test["acc"],
               test_top3=test["top3"], test_macro_f1=test["f1"], train_seconds=round(time.time() - t0))
    json.dump(res, open(os.path.join(a.out, f"{a.arch}_metrics.json"), "w"), indent=1)
    print("TEST (held-out MED117 blocks / Mendeley splits - optimistic, see report):", res)

if __name__ == "__main__":
    main()
