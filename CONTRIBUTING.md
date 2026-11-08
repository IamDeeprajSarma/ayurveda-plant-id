# How we work on GitHub

One repo, four people, four areas. Most conflicts come from two people editing the same file, so each area has an owner and we keep changes small.

## Areas and owners
| Folder | Owner | Notes |
|---|---|---|
| `ml/`, `notebooks/`, `docs/class_map.csv` | Model owner | data prep, training, evaluation |
| `backend/` | Backend developer | Django API; `identify/predictor.py` is the model glue |
| `frontend/` | Frontend developer | React app |
| `docs/` (except `class_map.csv`), `backend/identify/plants.json` | Data and report owner | species info, report |

If you need a change in someone else's folder, open an issue or a PR and tag the owner. Don't edit it directly on `main`.

## Daily routine
```bash
git checkout main && git pull                  # start from the latest main
git checkout -b backend/predict-endpoint       # one branch per task: area/short-name
# ... work, then commit small and often ...
git add <files> && git commit -m "backend: add /api/predict with dummy predictor"
git pull --rebase origin main                  # bring in others' changes BEFORE pushing
git push -u origin backend/predict-endpoint
```
Then open a Pull Request into `main`. Another person reads it and clicks Approve before it is merged. Merge at least once a day, so branches never drift far.

## Rules
1. **Never commit directly to `main`.** Turn on branch protection: require a pull request and one approval.
2. **Never commit data, model weights, `.env`, `db.sqlite3` or uploads.** `.gitignore` already blocks them. Share big files through Google Drive and say where in the PR.
3. **Commit `classes.json`, `plants.json`, `docs/class_map.csv` and the API contract**, because they are small and everyone depends on them.
4. **Changing the API contract** (`docs/` handoff doc) needs both developers to agree before the PR is merged.
5. **Photo files** go in Drive, never in Git. Name them `<class_id>_p<plant number>_<nnn>.jpg`, for example `tulsi_p03_012.jpg`.

## Resolving a conflict
1. `git pull --rebase origin main`. Git stops and lists the conflicted files.
2. Open each file and look for `<<<<<<<`, `=======` and `>>>>>>>`. Keep the right lines (often both), and delete the markers.
3. `git add <file>` then `git rebase --continue`.
4. Run the code once, then push. If it all goes wrong, `git rebase --abort` returns you to where you started.
5. Two people changed the same lines and can't tell which is right? Ask them both, in the PR comments.

## Seeing how it all works
- **Issues and a Project board** (GitHub Projects): one issue per task, with columns To do, Doing, Review, Done.
- **README** says how to run each part. If you change how something runs, update the README in the same PR.
- **PRs are the record.** Say in each PR what changed and how you tested it.
