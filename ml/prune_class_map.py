"""Remove non-medicinal classes (food crops / ornamentals) from a class map.

    python ml/prune_class_map.py --map docs/class_map_draft.csv --out docs/class_map_medicinal.csv
    python ml/prune_class_map.py ... --keep Rose --drop Papaya Lemon      # override the defaults for this run

A class is matched by class_id, display_name or folder_match (case, spaces, brackets and '_' ignored), so it works
before or after you fixed the spellings. The whole class (all its source rows) is removed. The default list below is a
judgment call: these are plants grown mainly as food or for decoration. Many have folk uses, so edit DROP to taste.
"""
import argparse, csv, os, re

# display name -> accepted spellings (normalised)
DROP = {
    "Tomato": ["tomato"],
    "Beans": ["beans"],
    "Chilly": ["chilly", "chili", "chilli"],
    "Avocado": ["avacado", "avocado"],
    "Sapota": ["sapota"],
    "Palak (Spinach)": ["palakspinach", "palak", "spinach"],
    "Malabar Spinach": ["malabarspinach"],
    "Jackfruit": ["jackfruit"],
    "Bamboo": ["bamboo"],
    "Marigold": ["marigold"],
    "Rose": ["rose"],
}

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="docs/class_map_draft.csv")
    ap.add_argument("--out", default="docs/class_map_medicinal.csv")
    ap.add_argument("--keep", nargs="*", default=[], help="names to protect from the default drop list")
    ap.add_argument("--drop", nargs="*", default=[], help="extra names to remove")
    a = ap.parse_args()

    keep = {norm(k) for k in a.keep}
    groups = {name: {norm(x) for x in alts} - keep for name, alts in DROP.items()}
    groups.update({n: {norm(n)} for n in a.drop})
    with open(a.map, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(l for l in f if not l.startswith("#"))
        fields, rows = reader.fieldnames, list(reader)

    by_class = {}
    for r in rows:
        by_class.setdefault(r["class_id"], []).append(r)
    dropped, matched = {}, set()
    for cid, rs in by_class.items():
        names = {norm(x) for r in rs for x in (r["class_id"], r["display_name"], r["folder_match"])}
        for label, alts in groups.items():
            if names & alts:
                dropped[cid] = rs[0]["display_name"]; matched.add(label); break

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows(r for r in rows if r["class_id"] not in dropped)
    print(f"{len(by_class)} classes in, {len(dropped)} removed, {len(by_class) - len(dropped)} kept -> {a.out}")
    print("removed:", ", ".join(sorted(dropped.values())))
    missing = [n for n in groups if n not in matched and groups[n]]
    if missing:
        print("not found in the map (spelling?):", ", ".join(missing))

if __name__ == "__main__":
    main()
