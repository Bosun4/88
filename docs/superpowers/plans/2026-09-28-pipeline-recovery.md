# Football pipeline recovery implementation plan

**Goal:** Restore the original analyst → referee framework and an observable, bounded production run.

**Architecture:** Keep the collector, evidence compiler, protocol adapter, immutable ledger and static Pages. Add a focused panel orchestrator using the existing transport and strict normalization; retain single_pass as an explicit option.

**Tech Stack:** Python 3.12, aiohttp, pytest, dependency-free HTML/CSS/JavaScript, GitHub Actions.

**Spec:** ../specs/2026-09-28-pipeline-recovery-design.md

**Execution:** Main agent edits and verifies; scouts are read-only per user instructions. Autonomous implementation and GitHub validation are already authorized.

- [x] Freeze the legacy sample for three tests. Reproduce current CI failures, replace live-data inputs with explicit fixtures, run the affected tests.
- [x] Add regression tests for preserved odds, unknown injuries and current business-day scope. Edit fetch_data.py, keeping fixture identity and kickoff gates.
- [x] Add tests for GPT/Grok → Gemini ordering, no referee with zero analysts, three-call budget reservation, cache reuse/invalidation and complete/partial/failed results. Implement scripts/panel.py and wire predict.py/main.py.
- [x] Use strict JSON and streaming transport for bounded panel stages, without endpoint retry multiplication; preserve per-model status. Default review remains deterministic and zero AI calls.
- [x] Test source provenance and exclude synthetic OU odds. Improve prompts without manufacturing evidence or changing model scores locally.
- [x] Restore dark dashboard, model roles and run coverage; add frontend behavior checks and visually verify desktop/mobile.
- [x] Update workflow secrets/configuration, diagnostics and README. Full offline regression: 374 Python tests and 15 frontend tests passed; pip check clean; no known runtime dependency vulnerabilities found.
- [x] Commit, publish and validate a real manual run and Pages with the authenticated GitHub session.

## Production verification — 2026-09-28

- PR #88 merged as `86c83d6c4b03062cdc3d23cc40350aade93548a6`; the uploaded code tree matched the locally tested implementation. Main Offline CI and Pages deployment passed.
- Football AI Predict #692 (`36345963414`) completed with 1/1 valid final predictions: GPT succeeded, Grok 4.6 returned HTTP 503 channel saturation, and Gemini succeeded. Three requests, zero cache hits; no retries.
- Model directory diagnostic (`36346293789`) listed three Grok models. #693 (`36346450956`) checked 4.5: HTTP 503 channel saturation. #694 (`36346594668`) checked 4.2 Fast: HTTP 422, provider reported its internal mapped model `grok-420-fast` did not exist. Both workflows and their Pages deployments succeeded with the valid GPT/Gemini result; each made one new request and reused two cached results.
- Restored Repository Variable `GROK_MODEL` to `熊猫-A-10-grok-4.6`. Further Grok availability depends on the provider repairing its channel; no claim of successful three-model coverage is made.
- Published result at 04:03:49 Beijing time: Columbus vs Inter Miami, main score 1-2, risk scenarios 1-1 / 2-2 / 1-0, tier D, observe only. The business date is 2026-09-27 because of the 11-hour boundary.
- Final regression on published source: 374 Python tests, 15 frontend tests, `pip check`, and `git diff --check` passed. The runtime dependency audit reported no known vulnerabilities. Browser console errors were empty; the 390 px mobile viewport had 375 px document width.
- Pages JSON matched the current main payload. The ledger retained 43 entries across all three runs, with unchanged first-lock bytes and a valid hash chain. The committed original snapshot hash matched its ledger entry; Windows checkout CRLF conversion was accounted for by hashing the Git blob.
- Current Pages: https://bosun4.github.io/88/
- Latest verified prediction run: https://github.com/Bosun4/88/actions/runs/36346594668

Successful execution and protocol validation do not establish a higher prediction win rate. Evaluate that separately using the preserved pre-match ledger and verified results.
