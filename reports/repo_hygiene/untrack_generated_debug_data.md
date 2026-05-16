# Repo Hygiene: Untrack Generated Debug Data

## Purpose
Remove generated AI/debug/runtime artifacts from Git tracking while preserving local runtime files.

## Changes
- Untracked data/debug*
- Untracked data/ai_phase_results*
- Untracked data/ai_cache if present
- Kept frontend data files such as data/predictions.json and data/history_*.json

## Non-goals
- No prediction logic changes.
- No history rewrite.
- No deletion of local runtime files.
- No force push.
