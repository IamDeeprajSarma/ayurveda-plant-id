"""Draft a class map from raw dataset folders (edit the result by hand afterwards).

    python ml/make_class_map.py data/raw/medicinal_leaf data/raw/leaves48 --out docs/class_map_draft.csv

Folders with the same name in different datasets (ignoring case, spaces, '_' and '-') become ONE class with one row per
source, so the same species from both datasets is pooled. The source column is the dataset folder's name under data/raw.
Check the printed lists for the same species spelled differently in two datasets (they would NOT be merged), delete the
classes you don't want, and swap in your own class_ids / display names. (Written with --out, not '>' redirection,
because Windows PowerShell's '>' saves UTF-16, which prepare_data cannot read.)
"""
import argparse, csv, os, re

IMG = {".jpg", ".jpeg", ".png", ".webp"}

def key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

def has_images(d):
    return any(os.path.splitext(f)[1].lower() in IMG for f in os.listdir(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+"); ap.add_argument("--out", default="docs/class_map_draft.csv")
    a = ap.parse_args()
    rows, names, sources = {}, {}, []
    for root in a.roots:
        src = os.path.basename(os.path.normpath(root)); sources.append(src)
        for d in sorted(os.listdir(root)):
            path = os.path.join(root, d)
            if os.path.isdir(path) and has_images(path):
                k = key(d); names.setdefault(k, d); rows.setdefault(k, []).append((src, d))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class_id", "display_name", "source", "folder_match"])
        for k, lst in sorted(rows.items()):
            for src, d in lst:
                w.writerow([k, names[k], src, d])
    both = [names[k] for k, l in rows.items() if len({s for s, _ in l}) == len(sources)]
    print(f"{len(rows)} classes written to {a.out}; in all datasets: {len(both)}")
    for src in sources:
        only = sorted(names[k] for k, l in rows.items() if {s for s, _ in l} == {src})
        print(f"only in {src} ({len(only)}): {', '.join(only)}")

if __name__ == "__main__":
    main()
