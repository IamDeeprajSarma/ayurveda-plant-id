"""Count research-grade iNaturalist observations per candidate species,
inside a Tripura bounding box and a wider North-East India box.

Run on your own machine (needs internet):
    python ml/check_inaturalist.py docs/species_candidates.csv docs/inat_counts.csv
Uses only the standard library. Be polite: ~1 request/second.
"""
import csv, json, sys, time, urllib.parse, urllib.request

API = "https://api.inaturalist.org/v1/observations"
BOXES = {
    "tripura": dict(swlat=22.9, swlng=91.1, nelat=24.55, nelng=92.35),
    "ne_india": dict(swlat=21.9, swlng=89.7, nelat=29.5, nelng=97.4),
}

def count(name, box):
    params = dict(taxon_name=name, quality_grade="research", per_page=0, **box)
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ayurveda-plant-id-student-project"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["total_results"]

def main(src, dst):
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        name = row["scientific_name"]
        rec = {"priority": row["priority"], "scientific_name": name}
        for label, box in BOXES.items():
            try:
                rec[label] = count(name, box)
            except Exception as e:
                rec[label] = f"error: {e}"
            time.sleep(1)
        print(rec)
        out.append(rec)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["priority", "scientific_name", "tripura", "ne_india"])
        w.writeheader(); w.writerows(out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
