"""Guard against accidental paid AI runs or unsafe CI permission changes."""
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    # BaseLoader keeps GitHub's YAML `on` key and string env values intact.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_paid_predictions_are_manual_and_bounded():
    doc = workflow("predict.yml")
    assert set(doc["on"]) == {"workflow_dispatch"}
    job = doc["jobs"]["predict"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    prediction = next(step for step in job["steps"] if step.get("name") == "Run Predictions")
    env = prediction["env"]
    expected = {
        "AI_RUN_MODE": "single_pass", "AI_PRIMARY_MODEL": "gpt",
        "AI_BATCH_SIZE": "6", "AI_CHUNK_CONCURRENCY": "2", "AI_MODEL_CONCURRENCY": "2",
        "AI_SINGLE_PASS_MAX_CALLS": "12", "AI_CONNECT_TIMEOUT": "20",
        "AI_READ_TIMEOUT": "180", "AI_HTTP_TOTAL_TIMEOUT": "180",
        "AI_PHASE1_RETRY_MAX": "0", "AI_FINAL_RETRY_MAX": "0",
        "AI_ENDPOINT_FAILOVER": "false", "AI_ENABLE_CROSS_EXAM": "false",
        "AI_ENABLE_CONSISTENCY_JUDGE": "false", "AI_ENABLE_FALLBACK_REFEREE": "false",
        "AI_ENABLE_FAMILY_DEBATE_REFEREE": "false",
        "AI_PERSISTENT_CACHE_ENABLED": "true", "AI_DECISION_CACHE_TTL": "1800",
    }
    assert {key: env.get(key) for key in expected} == expected
    assert "AI_MAX_REQUESTS_PER_AI" not in env
    assert "AI_CHUNK_SIZE" not in env
    assert env["GPT_MODEL"] == "${{ vars.GPT_MODEL }}"
    assert not any(key.startswith(("GROK_", "GEMINI_", "CLAUDE_")) for key in env)
    assert doc["jobs"]["deploy_pages"]["needs"] == "predict"


def test_automatic_ci_has_no_credentials_or_prediction_entrypoint():
    doc = workflow("ci.yml")
    assert set(doc["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert doc["permissions"] == {"contents": "read"}
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "secrets." not in text
    assert "scripts/main.py" not in text
    assert "--disable-socket --allow-unix-socket" in text
    assert "continue-on-error" not in text
    assert "pip_audit -r requirements.txt" in text
    for job in doc["jobs"].values():
        assert job.get("permissions", {"contents": "read"}) == {"contents": "read"}


def test_pages_only_packages_static_public_files():
    doc = workflow("pages.yml")
    assert "schedule" not in doc["on"]
    text = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "secrets." not in text and "scripts/main.py" not in text
    assert "path: _site" in text
    assert "cp index.html .nojekyll _site/" in text
    assert "cp -R data _site" not in text  # would publish cache/debug artifacts


def test_runtime_dependency_graph_is_exactly_pinned():
    requirements = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+]+", line) for line in requirements)
    names = [line.split("==")[0].lower().replace("_", "-") for line in requirements]
    assert len(names) == len(set(names))
    assert "deep-translator" not in names
    assert "aiohttp==3.14.3" in requirements


def test_publication_stages_and_validates_persistent_forward_ledger():
    doc = workflow("predict.yml")
    job = doc["jobs"]["predict"]
    publish = next(step for step in job['steps'] if step.get('name') == 'Push data updates')['run']
    assert 'read_verified_entries(ledger)' in publish
    assert '["git", "add", "--", "data/forward_ledger.jsonl"]' in publish
    artifact = next(step for step in job['steps'] if step.get('uses') == 'actions/upload-artifact@v4')
    assert 'data/forward_ledger.jsonl' in artifact['with']['path']
