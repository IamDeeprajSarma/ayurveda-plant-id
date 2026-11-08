"""Build data/processed/{train,val,test}/<class_id>/ from data/raw using docs/class_map.csv.

    python ml/prepare_data.py --raw data/raw --map docs/class_map.csv --out data/processed

Design choices (explain these in the report):
* MED117 images are video frames named frameN.jpg with no video id. Neighbouring frames are
  near-duplicates, so we split by BLOCKS of consecutive frames (whole blocks go to one split)
  and drop a few frames at every block edge. This stops adjacent frames leaking across splits.
  LIMITATION: MED117 has only 2-3 leaves per species, so the same physical leaf can still appear
  in train and test. MED117 test accuracy is therefore optimistic; trust the FIELD test set.
* Classes are capped (default 400 train images) to limit imbalance. Real photos from the other
  sources are kept first; MED117 frames fill the remaining quota by even striding.
* Field photos in data/field_test/ are never used for training. This script only reports them.
* A manifest.csv records the split of every image (useful for auditing leakage).
"""
import argparse, csv, os, random, re, shutil, sys
from collections import defaultdict
from PIL import Image, ImageOps

IMG = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(folder):
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG and not f.startswith(".")]

def safe_name(text):
    """Keep only letters, digits, '_', '-', '.' (Kaggle/zip reject characters such as [ ] ( ) and accents)."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")

def save_resized(src, dst, max_side):
    """Apply EXIF rotation, convert to RGB, shrink so the longest side <= max_side, save as JPEG."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(dst, "JPEG", quality=92)

def frame_key(name):
    nums = re.findall(r"\d+", name)
    return (int(nums[-1]) if nums else 0, name)

def resolve(raw, source, match):
    base = os.path.join(raw, source)
    hits = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d)) and match.lower() in d.lower()]
    if len(hits) != 1:
        sys.exit(f"[map error] source={source!r} folder_match={match!r} matched {len(hits)} folders: {hits}")
    return os.path.join(base, hits[0])

def stride_pick(items, k):
    if k <= 0: return []
    if len(items) <= k: return list(items)
    return [items[int(i * len(items) / k)] for i in range(k)]

def split_ratio(n, rng, ratios=(0.7, 0.15, 0.15)):
    idx = list(range(n)); rng.shuffle(idx)
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]

def plant_split(files, rng):
    """Our own photos are named <class>_p<plant>_<n>.jpg. If there are >= 3 plants, give one whole plant
    to val, one to test and the rest to train, so no plant appears in two splits. Else return None."""
    groups = defaultdict(list)
    for f in files:
        m = re.search(r"_p(\d+)_", f)
        if not m:
            return None
        groups[m.group(1)].append(f)
    if len(groups) < 3:
        return None
    keys = sorted(groups); rng.shuffle(keys)
    return {"val": groups[keys[0]], "test": groups[keys[1]],
            "train": [f for k in keys[2:] for f in groups[k]]}

def block_split(files, rng, block, edge, ratios=(0.7, 0.15, 0.15)):
    files = sorted(files, key=frame_key)
    blocks = [files[i:i + block] for i in range(0, len(files), block)]
    if len(blocks) > 1 and len(blocks[-1]) < block // 2:   # merge a short tail so no split gets an empty block
        tail = blocks.pop()
        blocks[-1] = blocks[-1] + tail
    order = list(range(len(blocks))); rng.shuffle(order)
    nb = len(blocks)
    n_val = max(1, round(nb * ratios[1])) if nb >= 3 else 0
    n_test = max(1, round(nb * ratios[2])) if nb >= 3 else 0
    assign = {}
    for r, b in enumerate(order):
        assign[b] = "val" if r < n_val else "test" if r < n_val + n_test else "train"
    out = {"train": [], "val": [], "test": []}
    for b, blk in enumerate(blocks):
        core = blk[edge:len(blk) - edge] if len(blk) > 2 * edge + 3 else []
        out[assign[b]] += core
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--map", default="docs/class_map.csv")
    ap.add_argument("--out", default="data/processed"); ap.add_argument("--field", default="data/field_test")
    ap.add_argument("--cap-train", type=int, default=400); ap.add_argument("--cap-eval", type=int, default=60)
    ap.add_argument("--block", type=int, default=40); ap.add_argument("--edge", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max-side", type=int, default=512)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = [r for r in csv.DictReader(l for l in open(a.map, encoding="utf-8") if not l.startswith("#"))]
    classes = defaultdict(list)
    for r in rows: classes[r["class_id"]].append(r)
    # Auto-include our own photos: data/raw/own/<class_id>/ for every class in the map
    own_root = os.path.join(a.raw, "own")
    if os.path.isdir(own_root):
        for cid, entries in classes.items():
            has_own = any(e["source"] == "own" and e["folder_match"].lower() == cid.lower() for e in entries)
            if os.path.isdir(os.path.join(own_root, cid)) and not has_own:
                entries.append({"class_id": cid, "display_name": entries[0]["display_name"],
                                "source": "own", "folder_match": cid})

    if os.path.exists(a.out): shutil.rmtree(a.out)
    manifest, summary = [], []
    for cid, entries in classes.items():
        real = {"train": [], "val": [], "test": []}   # (srcpath, source)
        video = {"train": [], "val": [], "test": []}
        for e in entries:
            if e["source"] == "own":
                try:
                    folder = resolve(a.raw, e["source"], e["folder_match"])
                except (SystemExit, FileNotFoundError):
                    print(f"[note] no own photos yet for {cid} (data/raw/own/{e['folder_match']})"); continue
            else:
                folder = resolve(a.raw, e["source"], e["folder_match"])
            files = list_images(folder)
            if e["source"] == "med117":
                parts = block_split(files, rng, a.block, a.edge)
                for s in parts: video[s] += [(os.path.join(folder, f), "med117") for f in parts[s]]
            else:
                files.sort()
                by_plant = plant_split(files, rng) if e["source"] == "own" else None
                if by_plant is not None:
                    for s in by_plant:
                        real[s] += [(os.path.join(folder, f), e["source"]) for f in by_plant[s]]
                else:
                    tr, va, te = split_ratio(len(files), rng)
                    for s, ix in (("train", tr), ("val", va), ("test", te)):
                        real[s] += [(os.path.join(folder, files[i]), e["source"]) for i in ix]
        if not any(real[s] or video[s] for s in real):
            print(f"[skip] {cid}: no images found, class left out"); continue
        counts = {}
        for s, cap in (("train", a.cap_train), ("val", a.cap_eval), ("test", a.cap_eval)):
            r_items = stride_pick(sorted(real[s]), cap)
            v_items = stride_pick(video[s], cap - len(r_items))
            chosen = r_items + v_items
            os.makedirs(os.path.join(a.out, s, cid), exist_ok=True)
            for path, src in chosen:
                stem = os.path.splitext(os.path.basename(path))[0]
                dst_name = safe_name(f"{src}__{os.path.basename(os.path.dirname(path))[:20]}__{stem}") + ".jpg"
                dst = os.path.join(a.out, s, cid, dst_name)
                k = 1
                while os.path.exists(dst):                 # never overwrite on a name clash
                    dst = os.path.join(a.out, s, cid, safe_name(f"{src}__{os.path.basename(os.path.dirname(path))[:20]}__{stem}") + f"_{k}.jpg"); k += 1
                save_resized(path, dst, a.max_side)
                manifest.append((s, cid, src, path))
            counts[s] = (len(r_items), len(v_items))
        field_dir = os.path.join(a.field, cid)
        n_field = len(list_images(field_dir)) if os.path.isdir(field_dir) else 0
        summary.append((cid, counts, n_field))
        for s in ("val", "test"):
            if sum(counts[s]) == 0:
                print(f"[warning] {cid} has no {s} images (too few photos): training will fail until it has some")

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["split", "class_id", "source", "original_path"]); w.writerows(manifest)
    print(f"{'class':<14}{'train(real+med117)':>20}{'val':>12}{'test':>12}{'field':>8}")
    for cid, c, nf in summary:
        fmt = lambda t: f"{t[0]}+{t[1]}"
        print(f"{cid:<14}{fmt(c['train']):>20}{fmt(c['val']):>12}{fmt(c['test']):>12}{nf:>8}")
    print(f"\n{len(classes)} classes, {len(manifest)} images -> {a.out}  (manifest.csv written)")

if __name__ == "__main__":
    main()
