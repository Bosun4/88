'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const context = vm.createContext({ console, Date, Intl });
vm.runInContext(fs.readFileSync(path.join(root, 'assets/dashboard.js'), 'utf8'), context);
const ui = context.Dashboard;
const now = Date.parse('2026-09-19T08:00:00Z');
const snapshot = { update_time: '2026-09-19 15:30:00', scope: 'today_only' };
const match = { id: 1, league: '英超', home_team: '主队', away_team: '客队', date: '2026-09-19', kickoff_at: '2026-09-19T12:00:00Z', prediction: { predicted_score: '2-1', final_direction: 'home', recommendation_tier: 'D', recommend_gate_pass: false, risk_score_candidates: [{ score: '1-2', reason: '反击', risk_type: '风险D' }], reason: '副文保留2-2风险' } };
const clone = (v) => JSON.parse(JSON.stringify(v));
const view = (m = match, d = snapshot, at = now) => ui.normalizeMatch(m, d, at, 0);

test('real snapshot: all 26 matches retained, today bucket does not mean today', () => {
  const d = JSON.parse(fs.readFileSync(path.join(root, 'data/predictions.json'), 'utf8'));
  const rows = ui.getMatches(d).map((m, i) => ui.normalizeMatch(m, d, now, i));
  assert.equal(rows.length, 26);
  assert.equal(rows.filter(m => m.today).length, 0);
  assert.equal(rows.filter(m => m.eligible).length, 0);
  assert.equal(rows.filter(m => m.state === 'stale').length, 26);
});

test('stale snapshot, started game, missing timezone and future snapshot all block recommendations', () => {
  const m = clone(match);
  m.prediction.recommend_gate_pass = true;
  m.prediction.recommendation = { is_recommended: true, bet_action: 'main' };
  assert.equal(view(m).eligible, true);
  assert.equal(view(m, { update_time: '2026-09-17 12:00:00' }).eligible, false);
  assert.equal(view(m, snapshot, Date.parse(m.kickoff_at)).eligible, false);
  delete m.kickoff_at;
  assert.equal(view(m).state, 'unknown');
  assert.equal(view(m).eligible, false);
  m.kickoff_at = '2026-09-19T20:00:00';
  assert.equal(view(m).eligible, false);
  m.kickoff_at = match.kickoff_at;
  assert.equal(view(m, { update_time: '2026-09-20 12:00:00' }).eligible, false);
  m.prediction.prematch_status = 'already_started';
  assert.equal(view(m).eligible, false);
});

test('Beijing date follows aware kickoff across midnight, not bucket or conflicting date', () => {
  const m = clone(match);
  m.date = '2026-09-18';
  m.kickoff_at = '2026-09-18T16:30:00Z';
  assert.equal(view(m).today, true);
});

test('unknown probabilities stay missing; zeros kept; percentages are not renormalized', () => {
  for (const p of [{}, { home_win_pct: null, draw_pct: null, away_win_pct: null }, { direction_probs: { home: 50, draw: 50 } }, { direction_probs: { home: '', draw: false, away: 100 } }, { direction_probs: { home: -5, draw: 25, away: 80 } }, { direction_probs: { home: 0, draw: 0, away: 0 } }]) assert.equal(ui.probabilities(p), null);
  assert.equal(JSON.stringify(ui.probabilities({ direction_probs: { home: 0, draw: 25, away: 75 } })), '[0,25,75]');
  assert.equal(JSON.stringify(ui.probabilities({ home_win_pct: 45, draw_pct: 28, away_win_pct: 27 })), '[45,28,27]');
  assert.equal(ui.probabilities({ direction_probs: { home: null, draw: null, away: null }, home_win_pct: 33, draw_pct: 33, away_win_pct: 34 }), null);
  assert.match(ui.renderMatch(view()), /胜平负概率未提供/);
});

test('failed/missing models never inherit final score, even if critic succeeded', () => {
  const p = { predicted_score: '3-1', gpt_score: '3-1', gpt_analysis: '该模型本轮未返回可展示分析', ai_call_status: { gpt: { phase1: { ok: false, status: 'http_524' }, critic: { ok: true, status: 'ok' }, last_status: 'ok' } } };
  const models = ui.models(p);
  assert.equal(models.find(m => m.name === 'gpt').score, null);
  assert.equal(models.find(m => m.name === 'grok').score, null);
  assert.equal(models.find(m => m.name === 'gemini').score, null);
  const html = ui.renderModels(p);
  assert.doesNotMatch(html, /model-score[^>]*>3-1/);
  assert.match(html, /http_524/);
});

test('single_pass reports model, cache, row abstention and budget status truthfully', () => {
  const p = { predicted_score: '2-1', final_direction: 'home', ai_call_status: { gemini: { single_pass: { ok: true, status: 'ok', model: 'fixture-model', cache_hit: true, row_status: 'ok' } } } };
  assert.match(ui.renderModels(p), /单模型单次分析/);
  assert.match(ui.renderModels(p), /fixture-model/);
  assert.match(ui.renderModels(p), /缓存命中/);
  assert.doesNotMatch(ui.renderModels(p), /三模型共识/);
  assert.equal(ui.models(p).find(m => m.name === 'gemini').score, '2-1');
  p.ai_call_status.gemini.single_pass.row_status = 'abstain';
  assert.equal(ui.models(p).find(m => m.name === 'gemini').score, null);
  p.ai_call_status.gemini.single_pass = { ok: false, status: 'run_call_budget_exhausted', row_status: 'abstain' };
  assert.match(ui.renderModels(p), /run_call_budget_exhausted/);
});

test('D tier, risk candidates, prose side scores survive combined filtering', () => {
  const rows = [view(), view({ ...match, id: 2, league: '德甲', prediction: { predicted_score: '1-0', recommendation_tier: 'A' } })];
  const filtered = ui.filterMatches(rows, { league: '英超', tier: 'D', status: 'risk' });
  assert.equal(filtered.length, 1);
  const html = ui.renderMatch(filtered[0]);
  for (const s of ['2-1', '1-2', '2-2', '副文保留', '风险 D']) assert.ok(html.includes(s), s);
  assert.equal(ui.filterMatches(rows, { search: '主队', tier: 'D' }).length, 1);
});

test('league_context contract renders fact envelopes, both teams, signed gaps and provenance', () => {
  const fact = (value, status = 'observed') => ({ value, status, reason: null, sources: [{ source: 'fixture-source', source_url: 'https://example.test/table', captured_at: '2026-09-19T06:00:00Z', league_id: 39, season: 2026 }] });
  const lc = { version: 1, league: '英超', season: fact(2026), stage: fact('early', 'derived'), round: fact(4), total_rounds: fact(38), home: { rank: fact(2), points: fact(9), remaining_matches: fact(35), title_gap: fact(0), europe_gap: fact(-3), relegation_cushion: fact(8), rest_days: fact(3.5), rotation: { value: 'unknown', status: 'unknown', sources: [], reason: 'no_sourced_rotation_evidence' } }, away: { points: fact(4), motivation: { value: 'unknown', status: 'unknown', sources: [], reason: 'points_do_not_prove_intent' } }, note: '分差不代表战意' };
  const html = ui.renderLeagueContext(lc);
  for (const s of ['赛季', '赛季初段', '主队', '客队', '欧战分差', '-3', 'fixture-source', '2026-09-19T06:00:00Z', '未知', '分差不代表战意']) assert.ok(html.includes(s), s);
  assert.doesNotMatch(html, /\[object Object\]/);
});

test('untrusted text is escaped and core audit/odds/reasons are retained', () => {
  const m = clone(match);
  m.home_team = '<img src=x onerror=alert(1)>';
  m.prediction.anchor_audit = { zero_zero: '零比零审计' };
  m.prediction.bookmaker_cross_audit = { bookmaker_intent: '公司交叉审计' };
  m.prediction.top3 = [{ score: '2-2', prob: null, logic: '候选比分理由' }];
  const html = ui.renderMatch(view(m));
  assert.doesNotMatch(html, /<img/);
  for (const s of ['&lt;img', '零比零审计', '公司交叉审计', '候选比分理由']) assert.ok(html.includes(s));
});
