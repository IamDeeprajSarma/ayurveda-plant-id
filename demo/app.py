"""Fallback demo server: the same API contract as the Django backend, plus the web page.

Run from the repo root:   python demo/app.py     then open http://localhost:8000
Settings (environment variables): PREDICTOR, MODEL_PATH, CONFIDENCE_THRESHOLD (default 0.60), PORT.
"""
import io
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import predictor` works however the script is started

import predictor  # noqa: E402
ROOT = HERE.parent
MEDIA = HERE / "media"
DB = HERE / "demo.sqlite3"
MAX_BYTES = 8 * 1024 * 1024
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))

app = Flask(__name__, static_folder=None)
MEDIA.mkdir(exist_ok=True)
P = predictor.load()


def load_plants():
    for p in (ROOT / "backend" / "identify" / "plants.json", HERE / "plants_starter.json"):
        if p.exists():
            data = json.load(open(p, encoding="utf-8"))
            recs = data if isinstance(data, list) else list(data.values())
            return {r["species_id"]: r for r in recs}, p
    return {}, None


PLANTS, PLANTS_FILE = load_plants()


def display_name(sid):
    r = PLANTS.get(sid)
    if not r:
        return sid.replace("_", " ").title()
    names = r.get("common_names") or []
    if not names or names[0].lower() == r["scientific_name"].lower():
        return r["scientific_name"]
    return f"{names[0]} ({r['scientific_name']})"


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS prediction (id INTEGER PRIMARY KEY AUTOINCREMENT,
        created REAL, status TEXT, top3 TEXT, lat REAL, lng REAL, image TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT,
        created REAL, prediction_id INTEGER, is_correct INTEGER, correct_species_id TEXT)""")
    return con


def err(msg, code=400):
    return jsonify({"error": msg}), code


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@app.get("/")
def index():
    return send_from_directory(HERE / "static", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "predictor": P.kind, "classes": len(P.classes),
                    "threshold": THRESHOLD, "plants_file": PLANTS_FILE.name if PLANTS_FILE else None})


@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return err("No image uploaded.")
    data = f.read()
    if len(data) > MAX_BYTES:
        return err("Image is larger than 8 MB.")
    try:
        image = predictor.open_image(io.BytesIO(data))
    except Exception:
        return err("Could not read this image. Please use a JPEG or PNG photo.")
    try:
        top3 = P.predict(image)
    except Exception as e:  # keep the demo alive, show a clear message
        return err(f"The model failed: {e}", 500)
    status = predictor.status_for(top3, THRESHOLD)
    path = MEDIA / f"{int(time.time() * 1000)}.jpg"
    image.save(path, "JPEG", quality=90)
    con = db()
    cur = con.execute("INSERT INTO prediction (created, status, top3, lat, lng, image) VALUES (?,?,?,?,?,?)",
                      (time.time(), status, json.dumps(top3), num(request.form.get("lat")),
                       num(request.form.get("lng")), path.name))
    con.commit()
    rid = cur.lastrowid
    con.close()
    return jsonify({"request_id": rid, "status": status, "predictions": [
        {"species_id": s, "name": display_name(s), "confidence": round(p, 4)} for s, p in top3]})


@app.get("/api/plants/")
def plants_list():
    ids = list(P.classes)
    return jsonify(sorted(({"species_id": s, "name": display_name(s)} for s in ids), key=lambda x: x["name"]))


@app.get("/api/plants/<species_id>")
def plant_detail(species_id):
    if species_id not in PLANTS and species_id not in P.classes:
        return err("Unknown species.", 404)
    rec = dict(PLANTS.get(species_id, {"species_id": species_id, "scientific_name": species_id}))
    rec["name"] = display_name(species_id)
    return jsonify(rec)


@app.post("/api/feedback")
def feedback():
    body = request.get_json(silent=True) or {}
    rid, ok = body.get("request_id"), body.get("is_correct")
    if not isinstance(rid, int) or not isinstance(ok, bool):
        return err("request_id (int) and is_correct (bool) are required.")
    con = db()
    if con.execute("SELECT 1 FROM prediction WHERE id=?", (rid,)).fetchone() is None:
        con.close()
        return err("Unknown request_id.", 404)
    con.execute("INSERT INTO feedback (created, prediction_id, is_correct, correct_species_id) VALUES (?,?,?,?)",
                (time.time(), rid, int(ok), body.get("correct_species_id")))
    con.commit()
    con.close()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  Predictor: {P.kind} ({len(P.classes)} classes)   Plant info: {PLANTS_FILE}")
    if P.kind == "dummy":
        print("  NOTE: placeholder answers - no trained model found at models/efficientnet_b0_best.pt")
    print(f"  Open http://localhost:{port}   (same Wi-Fi phone: http://<this computer's IP>:{port})\n")
    app.run(host="0.0.0.0", port=port, debug=False)
