"""Pre-flight check of raw datasets BEFORE writing class_map.csv / running prepare_data.py.

    python ml/inspect_raw.py data/raw/kaggle48 data/raw/yasht26            # one or more dataset folders
    python ml/inspect_raw.py data/raw/kaggle48 data/raw/yasht26 --cap 0    # hash every image (slower, exact)

For each dataset it reports: how many class folders / images, whether it ships its own train/val/test split,
how many file names look augmented (aug, flip, rotate ...), the image-size mix, and the share of images that have a
near-duplicate in the same folder (frames, augmented copies; flips and 90-degree rotations are detected too).
With two or more datasets it also counts images that appear in both and prints which folders match which, which
is exactly what you need to merge overlapping species into one class_id and to find mislabelled folders.
By default only the first --cap images (sorted by name) of each folder are hashed, so rates are estimates.
"""
import argparse, collections, os, re
import numpy as np
from PIL import Image, ImageOps

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def hamming_to_all(h, hs):
    """Bit distance between hash h and every hash in the uint64 array hs."""
    x = np.ascontiguousarray(np.bitwise_xor(hs, h))
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)

IMG = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = {"train", "training", "val", "valid", "validation", "test", "testing"}
AUG = re.compile(r"aug|flip|rotat|mirror|shear|zoom|bright|noise|_copy|\(\d+\)", re.I)
TR = getattr(Image, "Transpose", Image)
TRANSFORMS = [None, TR.FLIP_LEFT_RIGHT, TR.FLIP_TOP_BOTTOM, TR.ROTATE_90, TR.ROTATE_180,
              TR.ROTATE_270, TR.TRANSPOSE, TR.TRANSVERSE]

def hashes8(path):
    """dHash of the image under all 8 flips/rotations (index 0 = as is), plus its original size."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        size = im.size
        im.thumbnail((256, 256))
        out = []
        for t in TRANSFORMS:
            v = im if t is None else im.transpose(t)
            px = np.asarray(v.resize((9, 8), Image.LANCZOS), dtype=np.int16)
            out.append(np.packbits((px[:, 1:] > px[:, :-1]).flatten()).view(np.uint64)[0])
    return np.array(out, dtype=np.uint64), size

def scan(root):
    """Every folder that directly holds images -> sorted list of image paths."""
    found = {}
    for d, _, files in os.walk(root):
        imgs = sorted(f for f in files if os.path.splitext(f)[1].lower() in IMG and not f.startswith("."))
        if imgs:
            found[os.path.relpath(d, root)] = [os.path.join(d, f) for f in imgs]
    return found

def min_dist_to(h8, ref):
    """Smallest Hamming distance between any of the 8 variants of one image and each hash in ref."""
    return np.min([hamming_to_all(v, ref) for v in h8], axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--cap", type=int, default=300, help="max images hashed per folder (0 = all)")
    ap.add_argument("--max-ham", type=int, default=6)
    a = ap.parse_args()

    pool = {}                                   # root -> list of (folder, h8)
    for root in a.roots:
        folders = scan(root)
        if not folders:
            print(f"\n=== {root}: no images found"); continue
        counts = sorted(len(v) for v in folders.values())
        allfiles = [p for v in folders.values() for p in v]
        presplit = sorted({part for f in folders for part in re.split(r"[\\/]", f) if part.lower() in SPLITS})
        aug = sum(bool(AUG.search(os.path.basename(p))) for p in allfiles)
        print(f"\n=== {root}")
        print(f"{len(folders)} image folders, {len(allfiles)} images; per folder min/median/max = "
              f"{counts[0]}/{counts[len(counts) // 2]}/{counts[-1]}")
        print("ships its own split folders:", presplit if presplit else "no")
        print(f"file names that look augmented: {aug} ({100 * aug / len(allfiles):.1f}%)")

        items, sizes, dup_imgs, hashed = [], collections.Counter(), 0, 0
        for folder, paths in sorted(folders.items()):
            paths = paths[:a.cap] if a.cap else paths
            hs = []
            for p in paths:
                try:
                    h8, size = hashes8(p)
                except Exception:
                    continue
                hs.append(h8); sizes[size] += 1; items.append((folder, h8))
            if len(hs) < 2:
                continue
            H = np.array(hs, dtype=np.uint64)           # (n, 8)
            for i in range(len(hs)):
                d = min_dist_to(H[i], H[:, 0]); d[i] = 99
                dup_imgs += int(d.min() <= a.max_ham)
            hashed += len(hs)
        pool[root] = items
        top = ", ".join(f"{w}x{h} ({100 * n / max(sum(sizes.values()), 1):.0f}%)" for (w, h), n in sizes.most_common(3))
        print(f"most common sizes: {top}")
        print(f"images with a near-duplicate in their own folder: {dup_imgs}/{hashed} "
              f"({100 * dup_imgs / max(hashed, 1):.1f}%)   <- high = frames or augmented copies; do NOT trust random splits")

    roots = [r for r in a.roots if pool.get(r)]
    for x in range(len(roots)):
        for y in range(x + 1, len(roots)):
            A, B = roots[x], roots[y]
            ref = np.array([h[0] for _, h in pool[A]], dtype=np.uint64)
            pairs, hit = collections.Counter(), 0
            for folder, h8 in pool[B]:
                d = min_dist_to(h8, ref)
                j = int(d.argmin())
                if d[j] <= a.max_ham:
                    hit += 1; pairs[(pool[A][j][0], folder)] += 1
            print(f"\n=== overlap {A}  vs  {B}")
            print(f"{hit} of {len(pool[B])} hashed images in B also exist in A ({100 * hit / max(len(pool[B]), 1):.1f}%)")
            for (fa, fb), n in pairs.most_common(25):
                print(f"   {n:>5}  A:{fa}   <->   B:{fb}")

if __name__ == "__main__":
    main()
