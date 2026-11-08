# Tripura Ayurvedic Plant Identification (prototype)

Web app that identifies Ayurvedic medicinal plants of Tripura from a photo. Final-year project.
For learning only: never eat or use a plant based on this app.

## Parts
| Folder | What it is | How to run |
|---|---|---|
| `ml/` | data preparation, training, evaluation (PyTorch) | see `ml/` scripts; training runs on Kaggle (`notebooks/kaggle_train.ipynb`) |
| `backend/` | Django REST API; loads the trained model | _TODO: backend developer_ |
| `frontend/` | React web app | _TODO: frontend developer_ |
| `docs/` | species list, class map, handoff notes | |

## Data and models (not in Git)
Images, trained weights and field photos live on Google Drive: _TODO: add link_.
Class ids are defined in `docs/class_map.csv`.

## ML quick start
```bash
conda create -n plantid python=3.11 -y && conda activate plantid
pip install -r ml/requirements.txt
python ml/prepare_data.py --raw data/raw --map docs/class_map.csv --out data/processed
python ml/train.py --arch efficientnet_b0 --epochs 12      # or use the Kaggle notebook
python ml/evaluate.py --ckpt models/efficientnet_b0_best.pt --root data/field_test --name field
```

## Team
_TODO: names and areas_ (see `CONTRIBUTING.md`).
