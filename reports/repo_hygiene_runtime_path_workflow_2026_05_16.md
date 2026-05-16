# P0/P1 Repo Hygiene Runtime Path & Workflow Report — 2026-05-16

## Scope
First-batch hygiene fix only. This blocks future runtime/data pollution paths and leaves second-batch tracked debug cleanup (`data/debug`, `data/ai_phase_results`) for a separate PR.

## Changes
- Sanitized local `origin` remote URL to tokenless HTTPS.
- Updated `.github/workflows/predict.yml`:
  - Run `python scripts/main.py` from repo root.
  - Removed `git pull --rebase --autostash origin main || true`; pull/rebase now fails hard.
  - Removed `rsync -a --delete scripts/data/ data/`.
  - Removed wholesale `git add data/`.
  - Allowlist data commit to `data/predictions.json` and latest `data/history_*_today_*.json` only.
  - Runtime debug/cache dirs are removed before commit and uploaded as artifacts only where applicable.
- Updated `scripts/main.py` output path:
  - Writes canonical frontend data to repo-root `data/` regardless of caller CWD.
- Updated `index.html` recommendation semantics:
  - `精选` only for top4 / strict top-pick flags.
  - `recommend_gate_pass` / `recommendation.is_recommended` outside top4 displays as `可关注`.
- Updated `.gitignore` for runtime/debug/cache output:
  - `scripts/data/`
  - `data/debug/`
  - `data/debug_v20/`
  - `*.log`
- Removed untracked local pollution:
  - `reports/hotfix_audit/tail_guard_shadow_compare_latest.json`
  - `reports/live_predictions/parser.py`
  - `scripts/data/`
  - `tests/test_predict_patch.py`
  - local `__pycache__`, `.pyc`, `.patch`, `.rej`, `.orig` leftovers.

## Validation
- Remote URL PAT check: PASS (`origin` no longer contains GitHub token pattern locally).
- `scripts/predict.py` diff check: PASS (not modified).
- Workflow dangerous-pattern scan: PASS (no `rsync -a --delete`, no `git add data/`, no `origin main || true`).
- `node --check` on `index.html` script: PASS.
- `python3 -m py_compile scripts/main.py`: PASS.
- Token scan on tracked non-data/non-report files: PASS.
- Pytest subset: PASS (`12 passed`).

## Deferred
- Do not mix in second-batch cleanup here.
- Later PR should decide whether to `git rm --cached data/debug data/ai_phase_results` and move AI phase/debug snapshots to artifacts only.
