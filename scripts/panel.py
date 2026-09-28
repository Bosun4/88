"""Bounded GPT/Grok analysts followed by a Gemini referee, one fixture at a time."""
from __future__ import annotations

import asyncio
import copy
import os
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

try:
    from .single_pass import compact_evidence, _digest, _load_cache, _save_cache
    from .score_policy import SYSTEM, INSTRUCTIONS, score_only_row
    from .prematch_guard import prematch_status
except ImportError:
    from single_pass import compact_evidence, _digest, _load_cache, _save_cache
    from score_policy import SYSTEM, INSTRUCTIONS, score_only_row
    from prematch_guard import prematch_status

VERSION = 'panel-v24.0'
ROLES = {
    'gpt': '你负责比分情景初审：先审球队与赛程证据，再交叉让球、总进球及相邻比分，给出主线和最强反证。',
    'grok': '你负责独立反证：检验热门兑现、平局与反向路径、高比分尾部和轮换缺口；不得为博冷而博冷。',
    'gemini': '你是终审：先读原始证据，再比较初审的分歧及依据，独立裁决。初审缺席必须标注；不可按多数票或最高信心机械选分。',
}


def semantic_evidence(value):
    """Ignore collector clock churn; retain quote/published/kickoff times."""
    if isinstance(value, dict):
        return {k: semantic_evidence(v) for k, v in value.items() if k not in {'captured_at', 'fetched_at'}}
    if isinstance(value, list):
        return [semantic_evidence(v) for v in value]
    return value


def verified_sources(row, evidence):
    """Only exact supplied fact records are allowed to claim external support.

    Merely citing the URL of a provided market is not evidence for an invented
    lineup/news claim. The transport provides no retrieval tool.
    """
    supplied = []
    def visit(value):
        if isinstance(value, dict):
            url = value.get('source_url') or value.get('url')
            claim = value.get('claim')
            if isinstance(url, str) and claim and urlsplit(url).scheme in {'http', 'https'} and urlsplit(url).hostname:
                supplied.append((url, claim))
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
    visit(evidence)
    raw = row.get('raw_item') if isinstance(row.get('raw_item'), dict) else row
    untrusted_facts = raw.get('external_fact_table')
    facts = untrusted_facts if isinstance(untrusted_facts, list) else []
    accepted = [f for f in facts if isinstance(f, dict) and
                (f.get('source_url') or f.get('url'), f.get('claim')) in supplied]
    research = raw.get('web_research')
    claimed = bool(untrusted_facts or (research.get('sources') if isinstance(research, dict) else research))
    for target in (row, raw):
        target['external_fact_table'] = accepted
        target['web_research'] = {'used': False, 'sources': [], 'failure_reason': '本轮只分析采集证据，未执行网页检索'}
    if claimed and (len(accepted) != len(facts) or not accepted):
        row.setdefault('validation_warnings', []).append('unverified_model_sources_removed')
    return row


async def run_panel(engine, evidence):
    lock = engine.__dict__.setdefault('_panel_run_lock', threading.Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError('panel run already active')
    try:
        engine.AI_CALL_STATUS.clear()
        engine.AI_RESULT_FILES.clear()
        engine._LAST_AI_RUN_METADATA = {'run_mode': 'panel', 'run_status': 'running'}
        return await _run_panel(engine, evidence)
    except BaseException:
        engine._LAST_AI_RUN_METADATA['run_status'] = 'failed'
        raise
    finally:
        lock.release()


async def _run_panel(engine, evidence):
    started = time.monotonic()
    unique = {}
    for entry in evidence:
        idx = entry.get('match')
        if type(idx) is not int or idx <= 0:
            raise ValueError('evidence match must be a positive integer')
        if idx in unique and unique[idx] != entry:
            raise ValueError(f'conflicting evidence for match {idx}')
        unique[idx] = entry
    max_calls = max(0, engine._env_int('AI_PANEL_MAX_CALLS', 180))
    max_seconds = max(1, engine._env_int('AI_PANEL_MAX_SECONDS', 5400))
    ttl = max(1, engine._env_int('AI_DECISION_CACHE_TTL', 1800))
    use_cache = engine._env_bool('AI_PERSISTENT_CACHE_ENABLED', True) and not engine.AI_MOCK_MODE
    directory = Path(os.environ.get('AI_CACHE_DIR', 'data/ai_cache')) / VERSION
    request_slots = asyncio.Semaphore(max(1, engine.AI_MODEL_CONCURRENCY))
    fixture_slots = asyncio.Semaphore(max(1, engine.AI_CHUNK_CONCURRENCY))
    request_count = cache_hits = reserved = 0
    stages = []
    unavailable_origins = {}
    run_id = engine._make_run_id(evidence)
    session = None
    if engine.aiohttp is not None and not engine.AI_MOCK_MODE:
        session = engine.aiohttp.ClientSession(trust_env=True,
                    connector=engine.aiohttp.TCPConnector(limit=max(1, engine.AI_MODEL_CONCURRENCY)))

    async def call(name, phase, entry, analysts=None):
        nonlocal request_count, cache_hits
        idx = entry['match']
        endpoint = urlsplit(engine.get_url_for_ai(name))
        origin = (endpoint.scheme, endpoint.hostname, endpoint.port)
        packet = {'evidence': entry, 'analyst_outputs': analysts or {}}
        instructions = ROLES[name] + '\n' + INSTRUCTIONS
        key = _digest({'version': VERSION, 'model': engine._model_for(name),
                       'endpoint': engine.get_url_for_ai(name), 'phase': phase,
                       'system': SYSTEM, 'instructions': instructions,
                       'max_tokens': engine.AI_MAX_OUTPUT_TOKENS,
                       'temperature': engine.AI_TEMPERATURE_FINAL if phase == 'panel_final' else engine.AI_TEMPERATURE_PHASE1,
                       'response_format': engine.AI_USE_RESPONSE_FORMAT,
                       'stream': engine._env_bool('AI_STREAM', True),
                       'evidence': semantic_evidence(packet)})
        path = directory / f'{key}.json'
        lock_path = path.with_suffix('.lock')
        owned = False
        obj, status = {}, {}
        async with request_slots:
            reason = 'run_deadline_exhausted' if time.monotonic() - started >= max_seconds else ''
            if not reason and isinstance(entry.get('identity'), dict) and not engine.AI_MOCK_MODE:
                eligibility = prematch_status(entry['identity'])
                reason = eligibility if eligibility != 'eligible' else ''
            cached = _load_cache(path, ttl) if use_cache and not reason else None
            if reason:
                status = {'ok': False, 'status': reason}
            elif cached:
                obj, status = cached['object'], {**cached['status'], 'cache_hit': True}
                cache_hits += 1
            elif origin in unavailable_origins:
                status = {'ok': False, 'status': 'endpoint_unavailable',
                          'blocked_cause': unavailable_origins[origin]}
            else:
                if use_cache:
                    directory.mkdir(parents=True, exist_ok=True)
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        owned = True
                    except FileExistsError:
                        status = {'ok': False, 'status': 'duplicate_inflight_or_uncertain'}
                try:
                    cached = _load_cache(path, ttl) if owned else None
                    if cached:
                        obj, status = cached['object'], {**cached['status'], 'cache_hit': True}
                        cache_hits += 1
                    if not status:
                        request_count += 1
                        if use_cache:
                            _save_cache(path, {}, {'ok': False, 'status': 'request_inflight_or_uncertain'})
                        prompt = instructions + '\n指定match编号: ' + str([idx]) + '\n' + engine._safe_json_line(packet)
                        _, obj, status = await engine.async_call_ai_json(session, name, SYSTEM, prompt, phase, [idx])
                        status = {**status, 'cache_hit': False}
                        if status.get('status') in {'tls_handshake_error', 'tls_certificate_error', 'dns_error', 'connect_error'}:
                            unavailable_origins[origin] = status['status']
                        if use_cache:
                            _save_cache(path, obj, status)
                finally:
                    if owned:
                        lock_path.unlink(missing_ok=True)
        items = engine._unwrap_predictions(obj) if status.get('ok') else []
        matched = [r for r in items if isinstance(r, dict) and type(r.get('match')) in (int, str) and str(r['match']) == str(idx)]
        rows = engine.normalize_ai_predictions({'predictions': [{**matched[0], 'match': idx}]}, [idx], name, phase) if len(matched) == 1 else {}
        row = rows.get(idx)
        if row:
            row = score_only_row(verified_sources(row, entry))
        valid = bool(row and row.get('final_direction') in {'home', 'draw', 'away'} and not row.get('is_abstain'))
        status = {**status, 'ai_name': name, 'phase': phase, 'match_ids': [idx],
                  'model': engine._model_for(name), 'evidence_hash': key,
                  'row_status': 'ok' if valid else 'abstain'}
        stages.append(copy.deepcopy(status))
        print(f"  [PANEL {idx}] {name.upper()} {phase}: {status.get('status', 'ok')} / {status['row_status']}")
        return row if valid else None, status

    async def fixture(entry):
        nonlocal reserved
        idx = entry['match']
        # Reserve before scheduling analysts, so other matches cannot consume
        # the final referee's budget. Cache hits conservatively keep reservation.
        if reserved + 3 > max_calls:
            return idx, engine._abstain_ai_prediction(idx, 'run_call_budget_exhausted')
        reserved += 3
        entry = compact_evidence(entry)
        async with fixture_slots:
            analyst_tasks = [asyncio.create_task(call(n, 'panel_analysis', entry)) for n in ('gpt', 'grok')]
            try:
                initial = await asyncio.gather(*analyst_tasks)
            finally:
                for task in analyst_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*analyst_tasks, return_exceptions=True)
            analysts = {n: {**engine._short_prediction_for_prompt(r),
                            'reading_summary': r.get('reading_summary', {}),
                            'market_risk_audit': r.get('market_risk_audit', {})}
                        for n, (r, s) in zip(('gpt', 'grok'), initial) if r}
            statuses = {n: {'phase1': s} for n, (r, s) in zip(('gpt', 'grok'), initial)}
            if analysts:
                final, status = await call('gemini', 'panel_final', entry, analysts)
                statuses['gemini'] = {'final': status}
                final = final or engine._abstain_ai_prediction(idx, 'final_referee_failed')
            else:
                final = engine._abstain_ai_prediction(idx, 'all_analysts_failed')
            final.update(phase1_model_outputs=analysts, ai_call_status=statuses,
                         prediction_completed_at=datetime.now(timezone.utc).isoformat(),
                         evidence_hash=_digest(semantic_evidence(entry)))
            engine._save_snapshot(run_id, f'match{idx}_panel', {'final': final, 'status': statuses})
            return idx, final

    tasks = [asyncio.create_task(fixture(e)) for e in unique.values()]
    try:
        result = dict(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if session:
            await session.close()
    successful = sum(r.get('final_direction') in {'home', 'draw', 'away'} and not r.get('is_abstain') for r in result.values())
    engine._LAST_AI_RUN_METADATA = {
        'run_id': run_id, 'run_mode': 'panel', 'engine_version': VERSION,
        'run_status': 'completed' if successful == len(result) and result else 'partial' if successful else 'failed',
        'total_matches': len(result), 'successful_matches': successful, 'abstained_matches': len(result)-successful,
        'request_count': request_count, 'max_calls': max_calls, 'cache_hits': cache_hits,
        'max_seconds': max_seconds,
        'call_order': ['GPT + Grok independent analysis', 'Gemini final review'],
        'mock_mode': engine.AI_MOCK_MODE, 'retries': 0, 'batches': stages,
        'failure_summary': dict(Counter(s.get('status', 'unknown') for s in stages if not s.get('ok'))),
        'elapsed_seconds': round(time.monotonic()-started, 3),
    }
    return result
