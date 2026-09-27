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
- [ ] Commit, publish and validate a real manual run and Pages with the authenticated GitHub session.
