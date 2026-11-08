"""Match a downloaded image dataset's class folders against the shortlist.

    python ml/match_dataset_folders.py docs/species_candidates.csv /path/to/dataset_root
Prints, for each candidate, which folder (if any) looks like it, plus image counts.
Matching is fuzzy on scientific name, synonyms and common names - CHECK THE RESULT BY EYE.
"""
import csv, difflib, os, sys

def norm(s): return " ".join(s.lower().replace("_", " ").replace("-", " ").split())

def main(csv_path, root):
    folders = {d: len(os.listdir(os.path.join(root, d)))
               for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    keys = {norm(d): d for d in folders}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names = [row["scientific_name"]] + \
                    [n for n in row["source_name_variants"].replace("(same species)", "").split(";") if n] + \
                    [n for n in row["local_or_common_names"].split("/")]
            best = None
            for n in names:
                n = norm(n.split(" also listed")[0])
                if not n: continue
                hit = difflib.get_close_matches(n, keys, n=1, cutoff=0.8)
                if hit: best = keys[hit[0]]; break
                for k in keys:                      # substring match, e.g. 'tulsi' in 'tulsi (ocimum)'
                    if len(n) > 4 and n in k: best = keys[k]; break
                if best: break
            status = f"{best} ({folders[best]} files)" if best else "-- not found --"
            print(f"{row['priority']}  {row['scientific_name']:<28} {status}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
