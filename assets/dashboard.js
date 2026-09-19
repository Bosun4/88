'use strict';
/* Dependency-free renderer; exported helpers also run in offline contract tests. */
var Dashboard = (() => {
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text = v => typeof v === 'object' && v !== null ? JSON.stringify(v, null, 2) : String(v ?? '');
  const present = v => v !== null && v !== undefined && v !== '';
  const num = v => typeof v === 'number' && Number.isFinite(v) ? v : null;
  const awareTime = v => typeof v === 'string' && /(?:Z|[+-]\d\d:\d\d)$/.test(v) ? Date.parse(v) : NaN;
  const snapshotTime = d => {
    const v = d.update_time || d.generated_at || d.updated_at || '';
    return /^(\d{4}-\d\d-\d\d)[ T]\d\d:\d\d(?::\d\d)?$/.test(v) ? Date.parse(v.replace(' ', 'T') + '+08:00') : awareTime(v);
  };
  const day = t => Number.isFinite(t) ? new Intl.DateTimeFormat('en-CA', {timeZone:'Asia/Shanghai', year:'numeric',month:'2-digit',day:'2-digit'}).format(t) : '';
  const dateTime = t => Number.isFinite(t) ? new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai', dateStyle:'medium',timeStyle:'short'}).format(t) : '时间待核验';
  function getMatches(d) {
    if (Array.isArray(d.matches)) return d.matches;
    if (d.matches && typeof d.matches === 'object') return Object.values(d.matches).filter(Array.isArray).flat();
    return Array.isArray(d.predictions) ? d.predictions : [];
  }
  function probabilities(p) {
    const a = Object.prototype.hasOwnProperty.call(p, 'direction_probs') ? ['home','draw','away'].map(k => p.direction_probs?.[k]) : [p.home_win_pct,p.draw_pct,p.away_win_pct];
    if (!a.every(v => num(v) !== null && v >= 0 && v <= 100)) return null;
    const sum = a.reduce((s,v) => s + v, 0);
    return sum >= 98 && sum <= 102 ? a : null;
  }
  function riskScores(p) {
    const result = [];
    const add = (score, reason, label) => {
      if (!/^\d{1,2}[-:]\d{1,2}$/.test(String(score || ''))) return;
      score = score.replace(':','-');
      if (!result.some(r => r.score === score && (r.label === '副文比分') === (label === '副文比分'))) result.push({score, reason:text(reason), label});
    };
    const candidates = p.risk_score_candidates || [];
    for (const r of Array.isArray(candidates) ? candidates : []) add(typeof r === 'string' ? r : r.score, r.reason || r.logic || '', r.risk_type || '风险 D');
    for (const k of ['tail_scores','mid_tail_scores','upset_scores','all_tail_scores']) for (const r of Array.isArray(p[k]) ? p[k] : []) add(typeof r === 'string' ? r : r.score, r.reason || k, '尾部风险');
    const prose = [p.reason,p.ai_native_reason,p.final_ai_analysis,p.contextual_logic,p.tail_risk,p.grok_analysis].map(text).join('\n');
    for (const hit of prose.matchAll(/\b(\d{1,2}[-:]\d{1,2})\b/g)) if (hit[1] !== p.predicted_score) add(hit[1], '来自原文提及；未必是明确推荐', '副文比分');
    return result;
  }
  function normalizeMatch(m, d, now, index) {
    const p = m.prediction || m;
    const kickoff = awareTime(m.kickoff_at || m.kickoff || m.commence_time);
    const stamp = snapshotTime(d);
    const invalidSnapshot = !Number.isFinite(stamp) || stamp > now + 300000;
    const expired = Number.isFinite(stamp) && now - stamp > 86400000;
    const matchDay = Number.isFinite(kickoff) ? day(kickoff) : String(m.date || '').slice(0,10);
    const state = expired || (matchDay && matchDay < day(now)) || (Number.isFinite(kickoff) && kickoff <= now) ? 'stale' : invalidSnapshot || !Number.isFinite(kickoff) ? 'unknown' : 'fresh';
    const abstain = p.is_abstain === true || p.final_direction === 'abstain' || !/^\d+[-:]\d+$/.test(String(p.predicted_score || ''));
    const eligible = state === 'fresh' && !abstain && p.recommend_gate_pass === true && p.recommendation?.is_recommended === true && (!p.prematch_status || p.prematch_status === 'eligible');
    return {raw:m,p,index,key:'match-'+index,home:m.home_team || m.home || '主队',away:m.away_team || m.guest || '客队',league:m.league || '未知联赛',tier:p.recommendation_tier || p.recommendation?.tier || 'unknown',kickoff,state,eligible,abstain,today:matchDay === day(now),risks:riskScores(p)};
  }
  function models(p) {
    return ['gpt','grok','gemini','claude'].map(name => {
      const st = p.ai_call_status?.[name] || {};
      const call = st.single_pass || st.phase1;
      const single = Boolean(st.single_pass);
      const analysis = p[name+'_analysis'];
      const valid = call?.ok === true && call.row_status !== 'abstain' && !(single && p.final_direction === 'abstain');
      const score = valid ? (single ? p.predicted_score : p[name+'_score']) || null : null;
      return {name,single,score,analysis:valid ? analysis : null,call};
    });
  }
  function renderModels(p) {
    return '<div class="model-grid">' + models(p).map(m => `<section class="model-card"><h3>${esc(m.name.toUpperCase())}</h3><p>${m.single ? '单模型单次分析' : '模型初审'} · ${esc(m.call?.model || '本轮未提供')}</p><strong class="model-score">${esc(m.score || '未返回可用预测')}</strong><p>${esc(m.call?.status || 'not_called')}${m.call?.cache_hit ? ' · 缓存命中' : ''}${m.call?.row_status === 'abstain' ? ' · 本场弃权' : ''}</p>${m.analysis ? `<p class="analysis-text">${esc(text(m.analysis))}</p>` : ''}</section>`).join('') + '</div>';
  }
  const factLabels = {season:'赛季',stage:'赛季阶段',round:'轮次',total_rounds:'总轮次',rank:'排名',points:'积分',played:'已赛',remaining_matches:'剩余比赛',title_gap:'争冠分差',europe_gap:'欧战分差',relegation_cushion:'保级缓冲',rest_days:'距上一场 / 天',next_match_in_days:'距下一场 / 天',rotation:'轮换证据',motivation:'战意证据',goals_for:'进球',goals_against:'失球',goal_difference:'净胜球',schedule_density:'赛程密度'};
  function factRow(k, f) {
    const v = f && typeof f === 'object' && 'value' in f ? f.value : f;
    const unknown = v === 'unknown' || !present(v) || f?.status === 'unknown';
    const translated = {early:'赛季初段',middle:'赛季中段',late:'赛季末段'};
    const source = (Array.isArray(f?.sources) ? f.sources : []).map(s => `${s.source || '未知来源'} · ${s.captured_at || '抓取时间未知'}`).join('；');
    return `<div class="fact-row"><strong>${esc(factLabels[k] || k)}</strong><span>${unknown ? '未知' : esc(translated[v] || text(v))}</span><small>${esc([f?.status,source,f?.reason].filter(Boolean).join(' · '))}</small></div>`;
  }
  function renderLeagueContext(lc) {
    if (!lc || typeof lc !== 'object') return '<p class="muted">联赛、积分与轮换证据未提供。</p>';
    const base = Object.entries(lc).filter(([k]) => ['season','stage','round','total_rounds'].includes(k)).map(([k,v]) => factRow(k,v)).join('');
    return `<div class="league-context">${base}${['home','away'].map(side => `<section><h3>${side === 'home' ? '主队' : '客队'}</h3>${Object.entries(lc[side] || {}).map(([k,v]) => factRow(k,v)).join('') || '<p>未知</p>'}</section>`).join('')}<p class="muted">${esc(lc.note || '积分差不证明战意；未知轮换不能当成既定事实。')}</p></div>`;
  }
  const fieldLabels = {...factLabels, score:'比分',prob:'模型概率（%）',probability:'模型概率（%）',logic:'依据',reason:'理由',home:'主胜',draw:'平局',away:'客胜',zero_zero:'0-0 路径',one_one:'1-1 路径',high_score_tail:'高比分风险',handicap_cover:'让球兑现',one_x_two:'胜平负',handicap:'让球',correct_score:'比分赔率',total_goals:'总进球',external_market:'外部市场',selected_cluster:'选择的比分组',why_selected_score:'选择理由',adjacent_scores_checked:'相邻比分比较',league_style:'联赛特点',team_style:'球队特点',tempo:'比赛节奏',score_shape:'比分形态',btts_likelihood:'双方进球',rotation_risk:'轮换风险',sharp_money_direction:'资金方向判断',public_money_direction:'公众倾向',evidence:'依据',bookmaker_intent:'公司报价解读',exclusion:'排除理由',available:'是否有数据',missing:'缺失信息',source:'来源',source_url:'来源地址',captured_at:'采集时间',has_conflict:'是否存在冲突',conflicts:'冲突说明',why_this_can_fail:'可能失效的条件',why_recommended:'判断依据',risk_type:'风险情景',raw_packet_quality:'原始数据质量'};
  const valueLabels = {unknown:'未知',unclear:'尚不明确',yes:'是',no:'否',high:'高',medium:'中',low:'低',home:'主队',away:'客队',draw:'平局',observed:'来源事实',derived:'根据事实计算'};
  function readable(value) {
    if (!present(value)) return '<span class="muted">未提供</span>';
    if (Array.isArray(value)) return '<ul class="readable-list">'+value.map(v=>'<li>'+readable(v)+'</li>').join('')+'</ul>';
    if (typeof value === 'object') return '<dl class="readable-fields">'+Object.entries(value).map(([k,v])=>`<div><dt>${esc(fieldLabels[k] || k)}</dt><dd>${readable(v)}</dd></div>`).join('')+'</dl>';
    return `<span class="analysis-text">${esc(typeof value === 'boolean' ? (value ? '是' : '否') : valueLabels[value] || String(value))}</span>`;
  }
  function detail(title, value) {
    if (!present(value) || (typeof value === 'object' && Object.keys(value).length === 0)) return '';
    return `<section class="evidence-section"><h3>${esc(title)}</h3>${readable(value)}</section>`;
  }
  function renderMatch(m) {
    const p = m.p, r = m.raw;
    const probs = probabilities(p);
    const candidates = p.top3 || p.top_score_candidates || p.top_scores || [];
    const reason = p.reason || p.ai_native_reason || p.final_ai_analysis || p.contextual_logic || '本轮未提供主线文字分析';
    const blocks = [
      ['候选比分及理由',candidates],['锚点审计',p.anchor_audit],['公司交叉审计',p.bookmaker_cross_audit],
      ['盘口解读',p.market_interpretation],['盘口与市场来源',r.global_odds || r.international_odds || r.odds_movement],
      ['资金流分析（仅按来源陈述）',p.money_flow],['战术与节奏',p.tempo_xg_tactical_audit],['赛前因素',p.pre_match_factor_audit],
      ['比分排除与相邻比分比较',p.score_elimination_audit || p.score_cluster_audit],['反向风险',p.upset_evidence],
      ['外部事实与来源',p.external_fact_table],['来源冲突',p.source_conflict_audit],['证据缺口',p.minimum_evidence_needed],
      ['质量检查',p.data_quality],['原始副文',p.contextual_logic],['校验提示',p.validation_warnings],
      ['推荐闸门',p.recommend_gate_reasons || p.recommendation_downgrade_reasons],['原始玩法分析',p.bet_recommendation]
    ].map(([k,v]) => detail(k,v)).join('');
    const states = {fresh:'赛前记录',stale:'历史记录',unknown:'时间待核验'};
    const scoreChips = items => items.map(x=>`<span class="score-chip" title="${esc(x.label+' · '+x.reason)}">${esc(x.score)}</span>`).join('');
    const risk = m.risks.filter(x=>x.label !== '副文比分');
    const prose = m.risks.filter(x=>x.label === '副文比分');
    const reasonText = typeof reason === 'string' ? reason : Object.values(reason).filter(v=>typeof v==='string').join('；');
    const timeLabel = Number.isFinite(m.kickoff) ? dateTime(m.kickoff) : /^\d{4}-\d\d-\d\d$/.test(r.date || '') ? r.date+' · 时间待核验' : '时间待核验';
    const direction = {home:'主胜',draw:'平局',away:'客胜'}[p.final_direction] || '';
    const riskDetails = m.risks.map(x=>`<div class="risk-explanation"><strong>${esc(x.score)}</strong><div><span class="badge">${esc(x.label)}</span><p>${esc(x.reason)}</p></div></div>`).join('') || '<p class="muted">未提供风险比分</p>';
    return `<article class="match-card" id="${m.key}">
      <header class="match-head"><div class="match-meta"><span class="league-name">${esc(m.league)}</span><span>${esc(r.match_num || r.match_id || r.id || '')}</span><time>${esc(timeLabel)}</time></div><div class="match-badges"><span class="badge tier-${esc(m.tier)}">评级 ${esc(m.tier === 'unknown' ? '未知' : m.tier)}</span><span class="record-state">${states[m.state]}</span></div></header>
      <div class="match-summary">
        <div class="teams"><div class="team-name">${esc(m.home)}<small>主</small></div><span class="vs">VS</span><div class="team-name">${esc(m.away)}<small>客</small></div></div>
        <section class="primary-score"><span class="column-label">${m.state === 'stale' ? '原主线比分' : '主线比分'}</span><div class="score-row"><strong class="score ${m.abstain ? 'missing' : ''}">${esc(m.abstain ? '未给出' : p.predicted_score)}</strong><span class="direction">${esc(m.abstain ? '弃权' : direction)}</span></div></section>
        <section class="risk-summary"><span class="column-label">风险比分 <small>独立于评级</small></span><div class="score-chips">${scoreChips(risk) || '<span class="muted">未提供</span>'}</div></section>
        <section class="prose-summary"><span class="column-label">副文提及 <small>不等于推荐</small></span><div class="score-chips">${scoreChips(prose) || '<span class="muted">无额外比分</span>'}</div></section>
      </div>
      <div class="reason-preview"><span>核心判断</span><p>${esc(reasonText)}</p></div>
      <div class="match-bottom"><span class="probabilities">${probs ? probs.map((v,i)=>`${['主','平','客'][i]} ${v}%`).join('　') : '胜平负概率未提供'}</span><span class="availability">${m.eligible ? '赛前可关注' : m.state === 'stale' ? '保留原判断 · 不作当前推荐' : '仅供分析'}</span></div>
      <details class="analysis-details"><summary>展开完整分析 <span>主线依据 · 风险情景 · 赔率证据 <i aria-hidden="true">＋</i></span></summary><div class="detail-content">
        <div class="reading-columns">${detail('主线完整依据',reason)}<section class="evidence-section"><h3>风险 D / 副文比分 · 情景依据</h3>${riskDetails}</section></div>
        ${detail('总进球 / 双方进球',[p.goal_band,p.btts,p.goal_range].filter(present).join(' / '))}
        <div class="evidence-grid">${blocks}</div>
        <section class="evidence-section league-section"><h3>联赛与赛程证据</h3>${renderLeagueContext(p.league_context || r.league_context)}</section>
        <details class="technical-details"><summary>数据来源与模型记录</summary><div class="technical-content">${renderModels(p)}${detail('原始胜平负报价',{home:r.sp_home ?? null,draw:r.sp_draw ?? null,away:r.sp_away ?? null})}<details><summary>查看原始数据</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></div></details>
      </div></details>
    </article>`;
  }
  const structuredRiskCount = m => m.risks.filter(x=>x.label !== '副文比分').length;
  function filterMatches(rows, f) {
    const out = rows.filter(m => (!f.league || m.league === f.league) && (!f.tier || m.tier === f.tier) && (!f.search || [m.home,m.away,m.league,m.raw.match_num,m.raw.id].join(' ').toLowerCase().includes(f.search.toLowerCase())) && (!f.status || (f.status === 'risk' ? structuredRiskCount(m) > 0 : ['today','eligible','abstain'].includes(f.status) ? m[f.status] : m.state === f.status)));
    if (f.sort === 'kickoff') out.sort((a,b) => (a.kickoff || Infinity)-(b.kickoff || Infinity));
    if (f.sort === 'tier') out.sort((a,b) => 'SABCD'.indexOf(a.tier) - 'SABCD'.indexOf(b.tier));
    if (f.sort === 'risk') out.sort((a,b) => structuredRiskCount(b) - structuredRiskCount(a));
    return out;
  }
  async function start() {
    const $ = id => document.getElementById(id);
    let rows = [], data = {}, refreshTimer;
    function render() {
      rows = getMatches(data).map((m,i) => normalizeMatch(m,data,Date.now(),i));
      const filtered = filterMatches(rows,Object.fromEntries(['search','league','tier','status','sort'].map(k => [k,$(k).value])));
      $('matches').innerHTML = filtered.map(renderMatch).join('') || '<div class="empty">当前条件下没有比赛。可重置筛选查看全部快照。</div>';
      $('result-count').textContent = `${filtered.length} / ${rows.length} 场`;
      document.querySelectorAll('#matches .analysis-details').forEach(d => d.open = $('expand-all').checked);
      const metrics = [['比赛记录',rows.length],['当前可关注',rows.filter(m=>m.eligible).length]];
      const counts = {all:rows.length,d:rows.filter(m=>m.tier==='D').length,risk:rows.filter(m=>structuredRiskCount(m)).length,today:rows.filter(m=>m.today).length};
      document.querySelectorAll('[data-count]').forEach(e=>e.textContent=counts[e.dataset.count]);
      document.querySelectorAll('[data-view]').forEach(e=>e.setAttribute('aria-pressed',String(e.dataset.view === ($('tier').value==='D' ? 'd' : ['risk','today'].includes($('status').value) ? $('status').value : !$('tier').value && !$('status').value ? 'all' : ''))));
      $('metrics').innerHTML = metrics.map(([k,v])=>`<div class="metric"><strong>${v}</strong><span>${k}</span></div>`).join('');
      const stamp = snapshotTime(data), stale = !Number.isFinite(stamp) || Date.now()-stamp>86400000 || stamp>Date.now()+300000;
      $('snapshot-status').className = 'snapshot-banner'+(stale?' stale':'');
      $('snapshot-status').innerHTML = `<strong>${stale ? '历史快照' : '赛前快照'}</strong><p>${esc(dateTime(stamp))} 更新${stale ? ' · 仅回看原判断，暂无当前推荐' : ' · 北京时间'}</p>`;
      $('version').textContent = '页面 v21 · 数据：'+(data.engine_version || data.runtime?.ai_run?.engine_version || '历史版本');
      const eligible = rows.filter(m=>m.eligible);
      $('watchlist').hidden = eligible.length === 0;
      $('watchlist').innerHTML = eligible.length ? `<h2>当前可关注</h2><div class="watchlinks">${eligible.map(m=>`<a href="#${m.key}">${esc(m.home)} vs ${esc(m.away)}</a>`).join('')}</div>` : '';
    }
    async function load() {
      $('refresh').disabled = true;
      try {
        const response = await fetch('data/predictions.json?t='+Date.now(),{cache:'no-store'});
        if (!response.ok) throw Error('HTTP '+response.status);
        data = await response.json();
        const selected = $('league').value;
        $('league').innerHTML = '<option value="">全部联赛</option>'+[...new Set(getMatches(data).map(m=>m.league || '未知联赛'))].sort().map(l=>`<option value="${esc(l)}">${esc(l)}</option>`).join('');
        $('league').value = selected;
        render();
        clearInterval(refreshTimer); refreshTimer = setInterval(render,60000);
      } catch(e) {
        $('snapshot-status').className = 'snapshot-banner error';
        $('snapshot-status').innerHTML = `<strong>快照载入失败</strong><p>${esc(e.message)}，请点击刷新重试。</p>`;
        $('watchlist').innerHTML = '';
        $('matches').innerHTML = '<div class="empty">当前数据不可用，推荐已关闭。</div>';
        clearInterval(refreshTimer);
      } finally { $('refresh').disabled = false; }
    }
    for (const k of ['search','league','tier','status','sort']) $(k).addEventListener(k==='search'?'input':'change',render);
    $('expand-all').addEventListener('change',()=>document.querySelectorAll('#matches .analysis-details').forEach(d=>d.open=$('expand-all').checked));
    document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
      const view=button.dataset.view; $('tier').value=view==='d' ? 'D' : '';
      $('status').value=['risk','today'].includes(view) ? view : ''; render();
    }));
    $('reset').addEventListener('click',()=>{for(const k of ['search','league','tier','status']) $(k).value='';$('sort').value='source';render();});
    $('refresh').addEventListener('click',load);
    await load();
  }
  return {getMatches,probabilities,normalizeMatch,models,renderModels,renderLeagueContext,renderMatch,filterMatches,start};
})();
if (typeof document !== 'undefined') Dashboard.start();
