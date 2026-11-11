# Fallback demo (Flask + one HTML page)

A complete, working prototype of the app: photo in, top 3 matches out, plant details and a feedback button.
It uses the **same API contract** as the Django backend in the handoff doc, so the team's React frontend
and Django backend can replace it, and this can stay as a backup.

## Run it (from the repo root)
```bash
conda activate plantid
pip install flask
python demo/app.py
```
Open http://localhost:8000.

* **With the trained model:** put the checkpoint at `models/efficientnet_b0_best.pt` (from the Kaggle
  results). The page badge then says "Model: N species". Needs the `plantid` environment (PyTorch).
* **Without it:** the app starts in *demo mode* and shows a visible badge, "Demo mode: placeholder answers".
  Those answers are NOT predictions. Never present them as results.
* **Phone on the same Wi-Fi:** open `http://<your computer's IP>:8000`. The camera button works. The
  "share location" option only works on HTTPS or localhost.

## Files
| File | What it does |
|---|---|
| `app.py` | Flask server: `/api/predict`, `/api/plants/`, `/api/plants/<id>`, `/api/feedback`, `/api/health` |
| `predictor.py` | Loads the trained model once (uses `ml/common.py`), or the placeholder predictor |
| `static/index.html` | The mobile-first page (no build step, no npm) |
| `plants_starter.json` | Names, families and cautions for the species. Uses are still to be added from sources |

## Settings (environment variables)
`PREDICTOR` = `auto` (default), `torch` or `dummy` · `MODEL_PATH` · `CONFIDENCE_THRESHOLD` (default 0.60) · `PORT` (default 8000)

If `backend/identify/plants.json` exists (written by the data teammate), it is used instead of `plants_starter.json`.

## Data it saves
Photos go to `demo/media/` and the log to `demo/demo.sqlite3`. Both are ignored by Git.
