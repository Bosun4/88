<!DOCTYPE html>

<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>QUANT FOOTBALL TERMINAL</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
:root{--bg:#08090e;--s1:#0e1118;--s2:#141822;--bd:#1e2536;--tx:#b8c4d6;--t2:#6b7a90;--t3:#3d4a5e;--w:#eaf0f8;--red:#ff4d5a;--gn:#00e08e;--bl:#4d8aff;--am:#ffb020;--pr:#b366ff;--cy:#00d4e0;--pk:#ff5ca0;--or:#ff8030;--y:#ffe040}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans SC','IBM Plex Sans',sans-serif;background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;font-size:13px}
.m{font-family:'IBM Plex Mono',monospace}
/* Header */
.hdr{position:sticky;top:0;z-index:99;background:rgba(8,9,14,.96);backdrop-filter:blur(16px);border-bottom:1px solid var(--bd);height:44px;display:flex;align-items:center;padding:0 16px}
.hdr-w{max-width:1280px;margin:0 auto;width:100%;display:flex;justify-content:space-between;align-items:center}
.hdr-l{display:flex;align-items:center;gap:8px}
.hdr-logo{width:6px;height:20px;background:var(--gn);border-radius:1px}
.hdr-t{font-family:'IBM Plex Sans';font-size:14px;font-weight:800;color:var(--w);letter-spacing:1px}
.hdr-v{font-size:9px;color:var(--t3);background:var(--s2);padding:2px 6px;border-radius:3px;font-family:'IBM Plex Mono';border:1px solid var(--bd)}
.hdr-r{display:flex;align-items:center;gap:10px}
.live{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--t2)}
.live-d{width:5px;height:5px;border-radius:50%;background:var(--gn);animation:p 2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.2}}
/* Main */
.wrap{max-width:1280px;margin:0 auto;padding:10px 12px}
/* Dashboard */
.dash{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin-bottom:14px}
.dc{background:var(--s1);border:1px solid var(--bd);border-radius:6px;padding:8px 10px}
.dc-v{font-family:'IBM Plex Mono';font-size:20px;font-weight:700}.dc-l{font-size:8px;color:var(--t3);margin-top:1px;text-transform:uppercase;letter-spacing:.5px}
/* Section */
.sec{display:flex;align-items:center;gap:6px;margin-bottom:8px}
.sec-t{font-size:13px;font-weight:800;color:var(--w);letter-spacing:.3px}
.sec-b{background:var(--am);color:#000;font-size:7px;font-weight:800;padding:2px 5px;border-radius:2px;letter-spacing:.5px}
/* TOP4 */
.t4g{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:5px;margin-bottom:14px}
.t4{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:12px;position:relative;cursor:pointer;transition:.15s;overflow:hidden}
.t4:hover{border-color:var(--am);box-shadow:0 0 20px rgba(255,176,32,.06)}
.t4-bar{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--am),var(--red))}
.t4-rank{position:absolute;top:8px;right:8px;font-family:'IBM Plex Mono';font-size:18px;font-weight:700;color:var(--am);opacity:.3}
.t4-lg{font-size:9px;color:var(--am);font-weight:700;margin-bottom:4px}
.t4-teams{display:flex;align-items:center;gap:6px;margin-bottom:8px}
.t4-tn{font-size:12px;font-weight:700;color:var(--w);flex:1}.t4-tn:last-child{text-align:right}
.t4-vs{font-size:8px;color:var(--t3);font-weight:800}
.t4-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px}
.t4-c{text-align:center;background:var(--bg);border-radius:3px;padding:4px}
.t4-cl{font-size:7px;color:var(--t3)}.t4-cv{font-family:'IBM Plex Mono';font-size:11px;font-weight:700;color:var(--w)}
.t4-cv.g{color:var(--gn)}.t4-cv.a{color:var(--am)}
.t4-ft{font-size:8px;color:var(--or);margin-top:4px}
/* Tabs */
.tabs{display:flex;gap:2px;background:var(--s1);border:1px solid var(--bd);border-radius:4px;padding:2px;max-width:320px;margin:0 auto 14px}
.tab{flex:1;text-align:center;padding:5px;border-radius:3px;font-size:10px;font-weight:700;color:var(--t3);cursor:pointer;border:none;background:none;transition:.15s}
.tab.ac{background:var(--bl);color:#fff}
/* Match Card */
.mc{background:var(--s1);border:1px solid var(--bd);border-radius:8px;overflow:hidden;margin-bottom:5px}
.mc.rec{border-color:var(--am)}
.mc-h{display:flex;justify-content:space-between;align-items:center;padding:6px 12px;background:var(--bg);border-bottom:1px solid var(--bd)}
.mc-lg{font-size:10px;color:var(--bl);font-weight:700}
.mc-badge{background:var(--am);color:#000;font-size:7px;font-weight:800;padding:1px 4px;border-radius:2px;margin-left:4px}
.mc-time{font-size:9px;color:var(--t3)}
.mc-b{padding:12px}
/* Teams row */
.teams{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.tm{width:35%;text-align:center}.tm-n{font-size:13px;font-weight:700;color:var(--w)}
.sc-c{text-align:center}
.sc-big{font-family:'IBM Plex Mono';font-size:28px;font-weight:700;color:var(--gn)}
.res-tag{display:inline-block;font-size:8px;padding:2px 6px;border-radius:2px;font-weight:700;margin-top:2px}
.res-tag.h{background:#ff4d5a18;color:var(--red)}.res-tag.d{background:#ffb02018;color:var(--am)}.res-tag.a{background:#4d8aff18;color:var(--bl)}
/* Stats row */
.sr{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;margin-bottom:8px}
.sb{background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:5px;text-align:center}
.sb-l{font-size:7px;color:var(--t3);text-transform:uppercase}.sb-v{font-family:'IBM Plex Mono';font-size:12px;font-weight:700}
/* Probability bar */
.prob{margin-bottom:8px;background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:6px 10px}
.prob-h{display:flex;justify-content:space-between;font-size:9px;font-weight:700;margin-bottom:3px}
.prob-bar{display:flex;height:3px;border-radius:2px;overflow:hidden;background:var(--s2)}
.prob-bar .h{background:var(--red)}.prob-bar .d{background:#64748b}.prob-bar .a{background:var(--bl)}
/* Signals */
.sig{padding:5px 8px;border-radius:4px;font-size:9px;font-weight:600;margin-bottom:3px;display:flex;align-items:center;gap:4px}
.sig-r{background:#ff4d5a0a;border:1px solid #ff4d5a20;color:var(--red)}
.sig-a{background:#ffb0200a;border:1px solid #ffb02020;color:var(--am)}
.sig-g{background:#00e08e0a;border:1px solid #00e08e20;color:var(--gn)}
.sig-b{background:#4d8aff0a;border:1px solid #4d8aff20;color:var(--bl)}
.sig-p{background:#b366ff0a;border:1px solid #b366ff20;color:var(--pr)}
/* Experience panel */
.exp{border:1px solid var(--or);border-radius:6px;overflow:hidden;margin-bottom:5px;background:linear-gradient(135deg,rgba(255,128,48,.04),rgba(255,80,90,.02))}
.exp-h{padding:5px 10px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid rgba(255,128,48,.15)}
.exp-t{font-size:10px;font-weight:800;color:var(--or)}.exp-s{font-family:'IBM Plex Mono';font-size:12px;font-weight:700;color:var(--w);background:var(--s2);padding:1px 5px;border-radius:3px}
.exp-b{padding:5px 10px;font-size:9px;line-height:1.5}
.exp-rule{display:flex;gap:4px;padding:1px 0}
.exp-id{font-family:'IBM Plex Mono';font-size:7px;color:var(--or);background:#ff803010;padding:0 3px;border-radius:2px}
.exp-rn{color:var(--w);font-weight:600;font-size:9px}
.exp-rec{margin-top:3px;padding:3px 6px;border-radius:3px;font-size:9px;font-weight:700}
.exp-rec.hi{background:#ff4d5a12;color:var(--red)}.exp-rec.md{background:#ffb02012;color:var(--am)}.exp-rec.lo{background:#00e08e12;color:var(--gn)}
/* Toggle */
.tbtn{width:100%;padding:5px;background:none;border:1px solid var(--bd);border-radius:4px;color:var(--t2);font-size:10px;cursor:pointer;font-weight:600;transition:.15s;font-family:'IBM Plex Sans','Noto Sans SC'}.tbtn:hover{background:var(--s2);color:var(--w)}
.det{display:none;padding:0 12px 12px}.det.open{display:block}
/* Model grid */
.mg{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px}
.mb{background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:6px;position:relative;overflow:hidden}
.mb::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.mb.c1::before{background:var(--gn)}.mb.c2::before{background:var(--bl)}.mb.c3::before{background:var(--pr)}.mb.c4::before{background:var(--am)}.mb.c5::before{background:var(--cy)}.mb.c6::before{background:var(--pk)}.mb.c7::before{background:var(--or)}
.mb-t{font-size:8px;font-weight:700;margin-bottom:2px;text-transform:uppercase;letter-spacing:.3px}
.mb-t.c1{color:var(--gn)}.mb-t.c2{color:var(--bl)}.mb-t.c3{color:var(--pr)}.mb-t.c4{color:var(--am)}.mb-t.c5{color:var(--cy)}.mb-t.c6{color:var(--pk)}.mb-t.c7{color:var(--or)}
.mb-r{display:flex;justify-content:space-between;font-size:8px;padding:1px 0}.mb-l{color:var(--t3)}.mb-v{font-family:'IBM Plex Mono';font-weight:600;color:var(--w);font-size:8px}
.chip{display:inline-block;background:var(--s2);border:1px solid var(--bd);padding:1px 4px;border-radius:2px;font-size:7px;font-family:'IBM Plex Mono';color:var(--t3);margin:1px}.chip.top{border-color:var(--gn);color:var(--gn)}
/* Section labels */
.sl{font-size:8px;color:var(--t3);font-weight:700;margin:6px 0 3px;padding-bottom:2px;border-bottom:1px solid var(--bd);text-transform:uppercase;letter-spacing:.5px}
/* Asian handicap grid */
.ahg{display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-bottom:5px}
.ahc{background:var(--bg);border:1px solid var(--bd);border-radius:3px;padding:4px;text-align:center}
.ahc-l{font-size:6px;color:var(--t3);text-transform:uppercase}.ahc-v{font-family:'IBM Plex Mono';font-size:10px;font-weight:700;color:var(--w)}.ahc-v.hot{color:var(--gn)}
/* Implied xG */
.ixg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;margin-bottom:5px}
.ixg-c{background:var(--bg);border:1px solid var(--bd);border-radius:3px;padding:5px;text-align:center}
.ixg-l{font-size:6px;color:var(--t3);text-transform:uppercase}.ixg-v{font-family:'IBM Plex Mono';font-size:14px;font-weight:700}
/* AI section */
.ai-box{border:1px solid var(--am);border-radius:6px;overflow:hidden;margin-top:5px;background:linear-gradient(135deg,rgba(255,176,32,.04),rgba(255,80,48,.02))}
.ai-h{padding:6px 10px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid rgba(255,176,32,.15)}
.ai-t{font-size:11px;font-weight:800;color:var(--am)}.ai-s{font-family:'IBM Plex Mono';font-size:14px;font-weight:700;color:var(--w);background:var(--s2);padding:1px 6px;border-radius:3px}
.ai-x{font-size:10px;color:var(--tx);line-height:1.4;padding:8px 10px}
.ai3{display:grid;grid-template-columns:1fr 1fr 1fr;border:1px solid var(--bd);border-radius:6px;overflow:hidden;margin-top:5px}
.ai3-c{padding:7px;background:var(--bg);border-right:1px solid var(--bd)}.ai3-c:last-child{border:none}
.ai3-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px}
.ai3-t{font-size:8px;font-weight:700}.ai3-t.gpt{color:var(--gn)}.ai3-t.grk{color:var(--cy)}.ai3-t.gem{color:var(--bl)}
.ai3-s{font-family:'IBM Plex Mono';font-size:9px;font-weight:700;color:var(--w);background:var(--s2);padding:1px 3px;border-radius:2px}
.ai3-x{font-size:7px;color:var(--t3);line-height:1.3}
/* Form/Stats */
.fg{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:5px}
.fb{background:var(--bg);border-radius:4px;padding:6px;border:1px solid var(--bd)}
.fb-t{font-size:8px;font-weight:700;color:var(--t2);margin-bottom:2px;text-transform:uppercase}
.fb-r{display:flex;justify-content:space-between;font-size:8px;padding:1px 0}
.fb-l{color:var(--t3)}.fb-v{font-family:'IBM Plex Mono';font-weight:600;color:var(--w);font-size:8px}
/* Footer */
.foot{text-align:center;padding:14px;color:var(--t3);font-size:7px;border-top:1px solid var(--bd);margin-top:16px;letter-spacing:.3px}
.empty{text-align:center;padding:40px;color:var(--t3)}
/* MC bar */
.mc-bar{display:flex;gap:2px;margin:3px 0}
.mc-bar span{height:3px;border-radius:1px}
@media(max-width:640px){.dash{grid-template-columns:repeat(2,1fr)}.mg,.fg{grid-template-columns:1fr}.ai3{grid-template-columns:1fr}.ai3-c{border-right:none;border-bottom:1px solid var(--bd)}.t4g{grid-template-columns:1fr}.ahg{grid-template-columns:repeat(2,1fr)}.sr{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="hdr"><div class="hdr-w"><div class="hdr-l"><div class="hdr-logo"></div><div class="hdr-t">QUANT FOOTBALL</div><div class="hdr-v" id="ver">vMAX</div></div><div class="hdr-r"><div class="live"><div class="live-d"></div><span class="m" id="ut">...</span></div></div></div></div>
<div class="wrap">
<div class="dash" id="db"></div>
<div class="sec"><div class="sec-t">ELITE PICKS</div><div class="sec-b">TOP 4</div></div>
<div class="t4g" id="t4"></div>
<div class="tabs"><button class="tab" onclick="sw('yesterday')">昨日</button><button class="tab ac" onclick="sw('today')">今日</button><button class="tab" onclick="sw('tomorrow')">明日</button></div>
<div id="ms"></div>
<div class="foot">ZI-BivariatePoisson · Dixon-Coles · MonteCarlo5K · Poisson · ELO · RF · GB · NN · SVM · KNN · ProOverround · FLB · CLV · SteamDetect · AdaptiveCalibrator · SmartFilter<br>GPT-5.4 · Grok-4.2 · Gemini-3.1 · Claude-Opus-4.6 · 58条经验规则 · 历史赔率匹配 · 亚盘概率 · 蒙特卡洛模拟 · 价值投注引擎 | vMAX</div>
</div>
<script>
var D=null,CT='today';
function cl(t){return t?String(t).replace(/```json/g,'').replace(/```/g,'').trim():''}
function fm(f){if(!f)return'?';return f.split('').map(c=>c==='W'?'<span style="color:var(--gn)">W</span>':c==='D'?'<span style="color:var(--am)">D</span>':c==='L'?'<span style="color:var(--red)">L</span>':c).join('')}
function P(v){return v!=null?(parseFloat(v)*100).toFixed(1)+'%':'?'}
function n(v,d){return v!=null&&v!=='?'?v:d||'?'}

fetch(‘data/predictions.json?t=’+Date.now()).then(r=>r.json()).then(d=>{D=d;document.getElementById(‘ut’).textContent=d.update_time||’’;document.getElementById(‘ver’).textContent=d.version||‘vMAX’;rdb();rt4(d.top4||[]);rm()}).catch(()=>{document.getElementById(‘ms’).innerHTML=’<div class="empty">AWAITING DATA FEED…</div>’});
function sw(t){CT=t;document.querySelectorAll(’.tab’).forEach(b=>b.classList.remove(‘ac’));event.target.classList.add(‘ac’);rm()}

function rdb(){var ms=D.matches||{},tt=0,rc=0,cf=0,expC=0,vb=0;[‘yesterday’,‘today’,‘tomorrow’].forEach(k=>{var a=ms[k]||[];tt+=a.length;a.forEach(m=>{var p=m.prediction||{};cf+=p.confidence||0;if(p.model_agreement)rc++;var ex=p.experience_analysis||{};if(ex.triggered_count>=2)expC++;if(p.value_bets&&p.value_bets.length)vb+=p.value_bets.length})});
var av=tt?Math.round(cf/tt):0;
document.getElementById(‘db’).innerHTML=`<div class="dc"><div class="dc-v" style="color:var(--am)">${tt}</div><div class="dc-l">MATCHES</div></div><div class="dc"><div class="dc-v" style="color:var(--gn)">${D.top4?D.top4.length:0}</div><div class="dc-l">ELITE PICKS</div></div><div class="dc"><div class="dc-v" style="color:var(--or)">${expC}</div><div class="dc-l">EXP TRIGGERS</div></div><div class="dc"><div class="dc-v" style="color:var(--pr)">${av}%</div><div class="dc-l">AVG CONF</div></div><div class="dc"><div class="dc-v" style="color:var(--gn)">${vb}</div><div class="dc-l">VALUE BETS</div></div>`}

function rt4(t4){var el=document.getElementById(‘t4’);if(!t4||!t4.length){el.innerHTML=’<div class="empty">NO DATA</div>’;return}
el.innerHTML=t4.map((m,i)=>{var p=m.prediction||{},cc=p.confidence>=70?‘g’:p.confidence>=50?‘a’:’’,ex=p.experience_analysis||{},ps=p.predictability_score||0;
var ft=’’;if(ex.triggered_count>0)ft+=`📋${ex.triggered_count}条经验`;if(ps>0)ft+=` · 可预测:${ps}`;
return`<div class="t4" onclick="go('m-${m.id}')"><div class="t4-bar"></div><div class="t4-rank">#${i+1}</div><div class="t4-lg">${m.league||''} ${m.match_num||''}</div><div class="t4-teams"><span class="t4-tn">${m.home_team}</span><span class="t4-vs">VS</span><span class="t4-tn">${m.away_team}</span></div><div class="t4-grid"><div class="t4-c"><div class="t4-cl">SCORE</div><div class="t4-cv g">${p.predicted_score||'?'}</div></div><div class="t4-c"><div class="t4-cl">RESULT</div><div class="t4-cv">${p.result||'?'}</div></div><div class="t4-c"><div class="t4-cl">CONF</div><div class="t4-cv ${cc}">${p.confidence||0}%</div></div></div>${ft?`<div class="t4-ft">${ft}</div>`:''}</div>`}).join(’’)}

function rm(){var el=document.getElementById(‘ms’),ms=(D.matches||{})[CT]||[];if(!ms.length){el.innerHTML=’<div class="empty">NO MATCHES</div>’;return}
el.innerHTML=ms.map(m=>{var p=m.prediction||{},hp=p.home_win_pct||33,dp=p.draw_pct||33,ap=p.away_win_pct||34;
var rc=p.result===‘主胜’?‘h’:p.result===‘平局’?‘d’:‘a’;
var poi=p.poisson||{},bvp=p.bivariate_poisson||{},elo=p.elo||{},rf=p.random_forest||{},gb=p.gradient_boost||{},nn=p.neural_net||{},dc=p.dixon_coles||{},mc=p.monte_carlo||{};
var hf=p.home_form||{},af=p.away_form||{},hs=m.home_stats||{},as2=m.away_stats||{};
var ts=p.top_scores||[],ex=p.experience_analysis||{},ahp=p.asian_handicap_probs||{},po=p.pro_odds||{};
var ixg_h=p.bookmaker_implied_home_xg,ixg_a=p.bookmaker_implied_away_xg;
var fw=p.fusion_weights||{},ps=p.predictability_score||0;
var ora=p.odds_range_analysis||{};

// Signals
var sig=’’;var ss=p.smart_signals||[];
var ew=p.extreme_warning||’’;if(ew&&ew!==‘无’&&ew!==’’)sig+=`<div class="sig sig-r">${ew}</div>`;
ss.forEach(s=>{
if(s.includes(‘🔥’)||s.includes(‘CLV’))sig+=`<div class="sig sig-g">${s}</div>`;
else if(s.includes(‘💎’)||s.includes(‘价值’))sig+=`<div class="sig sig-p">${s}</div>`;
else if(s.includes(‘📊’)||s.includes(‘Steam’))sig+=`<div class="sig sig-b">${s}</div>`;
else if(s.includes(‘🚨’))sig+=`<div class="sig sig-r">${s}</div>`;
else sig+=`<div class="sig sig-a">${s}</div>`});

// Value bets
var vbH=’’;var vbs=p.value_bets||p.value_bets_summary||[];
if(vbs.length){vbH=’<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:4px">’;
if(typeof vbs[0]===‘object’){vbs.forEach(v=>{vbH+=`<span style="background:#b366ff10;color:var(--pr);font-size:8px;padding:2px 5px;border-radius:2px;border:1px solid #b366ff20;font-family:IBM Plex Mono">💎${v.direction} EV+${v.ev}% @${v.odds}</span>`})}
else{vbs.forEach(v=>{vbH+=`<span style="background:#ffb02010;color:var(--am);font-size:8px;padding:2px 5px;border-radius:2px;border:1px solid #ffb02020;font-family:IBM Plex Mono">💰${v}</span>`})}
vbH+=’</div>’}

// Experience
var expH=’’;if(ex.triggered_count>0){var rc2=ex.total_score>=20?‘hi’:ex.total_score>=10?‘md’:‘lo’;
expH=`<div class="exp"><div class="exp-h"><div class="exp-t">📋 EXP ENGINE</div><div class="exp-s">${ex.triggered_count}/${ex.total_score}</div></div><div class="exp-b">`;
(ex.rules||[]).forEach(r=>{expH+=`<div class="exp-rule"><span class="exp-id">${r.id}</span><span class="exp-rn">${r.name}</span></div>`});
expH+=`<div class="exp-rec ${rc2}">${ex.recommendation||''}</div></div></div>`}

// Expand content
// Models
var mdl=’<div class="sl">MODELS</div><div class="mg">’;
mdl+=`<div class="mb c1"><div class="mb-t c1">POISSON</div><div class="mb-r"><span class="mb-l">Score</span><span class="mb-v">${n(poi.predicted_score)}</span></div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(poi.home_win)}/${n(poi.draw)}/${n(poi.away_win)}</span></div><div class="mb-r"><span class="mb-l">xG</span><span class="mb-v">${n(poi.home_xg)}-${n(poi.away_xg)}</span></div>${ts.length?'<div style="margin-top:2px">'+ts.slice(0,5).map((s,i)=>`<span class="chip${i<1?' top':''}">${s.score} ${s.prob}%</span>`).join('')+'</div>':''}</div>`;
mdl+=`<div class="mb c6"><div class="mb-t c6">ZI-BIVARIATE POISSON</div><div class="mb-r"><span class="mb-l">Score</span><span class="mb-v">${n(bvp.predicted_score)}</span></div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(bvp.home_win)}/${n(bvp.draw)}/${n(bvp.away_win)}</span></div><div class="mb-r"><span class="mb-l">ρ</span><span class="mb-v">${n(bvp.correlation)} p0:${n(bvp.p0_inflate)}</span></div></div>`;
mdl+=`<div class="mb c5"><div class="mb-t c5">MONTE CARLO 5K</div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(mc.home_win)}/${n(mc.draw)}/${n(mc.away_win)}</span></div><div class="mb-r"><span class="mb-l">Avg Goals</span><span class="mb-v">${n(mc.avg_goals)}</span></div></div>`;
mdl+=`<div class="mb c2"><div class="mb-t c2">ELO</div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(elo.home_win)}/${n(elo.draw)}/${n(elo.away_win)}</span></div><div class="mb-r"><span class="mb-l">Δ</span><span class="mb-v" style="color:${parseFloat(elo.elo_diff||0)>0?'var(--gn)':'var(--red)'}">${elo.elo_diff>0?'+':''}${n(elo.elo_diff)}</span></div></div>`;
if(po.true_home){mdl+=`<div class="mb c7"><div class="mb-t c7">PRO OVERROUND</div><div class="mb-r"><span class="mb-l">True H/D/A</span><span class="mb-v">${po.true_home}/${po.true_draw}/${po.true_away}</span></div><div class="mb-r"><span class="mb-l">Method</span><span class="mb-v">Shin+Pwr+Mult</span></div></div>`}
else{mdl+=`<div class="mb c7"><div class="mb-t c7">RANDOM FOREST</div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(rf.home_win)}/${n(rf.draw)}/${n(rf.away_win)}</span></div></div>`}
mdl+=`<div class="mb c3"><div class="mb-t c3">NEURAL NET</div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(nn.home_win)}/${n(nn.draw)}/${n(nn.away_win)}</span></div></div>`;
mdl+=`<div class="mb c4"><div class="mb-t c4">GRADIENT BOOST</div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(gb.home_win)}/${n(gb.draw)}/${n(gb.away_win)}</span></div></div>`;
mdl+=`<div class="mb c1"><div class="mb-t c1">DIXON-COLES</div><div class="mb-r"><span class="mb-l">H/D/A</span><span class="mb-v">${n(dc.home_win)}/${n(dc.draw)}/${n(dc.away_win)}</span></div></div>`;
mdl+=’</div>’;

// Implied xG
var ixgH=’’;if(ixg_h&&ixg_h!==’?’){ixgH=`<div class="sl">BOOKMAKER IMPLIED xG</div><div class="ixg"><div class="ixg-c"><div class="ixg-l">HOME xG</div><div class="ixg-v" style="color:var(--am)">${ixg_h}</div></div><div class="ixg-c"><div class="ixg-l">AWAY xG</div><div class="ixg-v" style="color:var(--cy)">${ixg_a}</div></div><div class="ixg-c"><div class="ixg-l">TOTAL</div><div class="ixg-v" style="color:var(--gn)">${p.expected_total_goals?parseFloat(p.expected_total_goals).toFixed(2):'?'}</div></div></div>`}

// Asian Handicap
var ahH=’’;if(ahp[‘ah_0’]||ahp[‘ou_2.5’]){
var ah0=ahp[‘ah_0’]||[0,0,0],ah05=ahp[‘ah_0.5’]||[0,0,0],ou25=ahp[‘ou_2.5’]||[0,0,0],bt=ahp.btts||0;
ahH=`<div class="sl">ASIAN HANDICAP & O/U</div><div class="ahg"><div class="ahc"><div class="ahc-l">AH0 Home</div><div class="ahc-v">${P(ah0[0])}</div></div><div class="ahc"><div class="ahc-l">AH0 Push</div><div class="ahc-v">${P(ah0[1])}</div></div><div class="ahc"><div class="ahc-l">AH0 Away</div><div class="ahc-v">${P(ah0[2])}</div></div><div class="ahc"><div class="ahc-l">AH-0.5 H</div><div class="ahc-v${ah05[0]>0.55?' hot':''}">${P(ah05[0])}</div></div><div class="ahc"><div class="ahc-l">O 2.5</div><div class="ahc-v${ou25[0]>0.55?' hot':''}">${P(ou25[0])}</div></div><div class="ahc"><div class="ahc-l">BTTS</div><div class="ahc-v${bt>0.55?' hot':''}">${P(bt)}</div></div></div>`}

// Odds Range
var orH=’’;if(ora.home){orH=`<div class="sl">ODDS RANGE ANALYSIS</div><div class="ahg"><div class="ahc"><div class="ahc-l">Home ${ora.home.hist_rate}%</div><div class="ahc-v" style="font-size:7px;color:var(--t2)">${ora.home.desc}</div></div><div class="ahc"><div class="ahc-l">Draw ${ora.draw.hist_rate}%</div><div class="ahc-v" style="font-size:7px;color:var(--t2)">${ora.draw.desc}</div></div><div class="ahc"><div class="ahc-l">Away ${ora.away.hist_rate}%</div><div class="ahc-v" style="font-size:7px;color:var(--t2)">${ora.away.desc}</div></div></div>`}

// Form
var frm=`<div class="sl">FORM & STATUS</div><div class="fg"><div class="fb"><div class="fb-t">${m.home_team}</div><div class="fb-r"><span class="fb-l">Trend</span><span class="fb-v">${n(hf.trend)}</span></div><div class="fb-r"><span class="fb-l">Score</span><span class="fb-v">${n(hf.score)}</span></div><div class="fb-r"><span class="fb-l">Form</span><span class="fb-v">${fm(hs.form)}</span></div><div class="fb-r"><span class="fb-l">Record</span><span class="fb-v">${n(hs.wins)}W${n(hs.draws)}D${n(hs.losses)}L</span></div><div class="fb-r"><span class="fb-l">Goals</span><span class="fb-v">${n(hs.goals_for)}/${n(hs.goals_against)}</span></div></div><div class="fb"><div class="fb-t">${m.away_team}</div><div class="fb-r"><span class="fb-l">Trend</span><span class="fb-v">${n(af.trend)}</span></div><div class="fb-r"><span class="fb-l">Score</span><span class="fb-v">${n(af.score)}</span></div><div class="fb-r"><span class="fb-l">Form</span><span class="fb-v">${fm(as2.form)}</span></div><div class="fb-r"><span class="fb-l">Record</span><span class="fb-v">${n(as2.wins)}W${n(as2.draws)}D${n(as2.losses)}L</span></div><div class="fb-r"><span class="fb-l">Goals</span><span class="fb-v">${n(as2.goals_for)}/${n(as2.goals_against)}</span></div></div></div>`;

// AI
var ai=`<div class="sl">AI MATRIX</div><div class="ai-box"><div class="ai-h"><div class="ai-t">👑 CLAUDE</div><div class="ai-s">${n(p.claude_score,'-')}</div></div><div class="ai-x">${cl(p.claude_analysis)||'N/A'}</div></div><div class="ai3"><div class="ai3-c"><div class="ai3-h"><span class="ai3-t gpt">GPT</span><span class="ai3-s">${n(p.gpt_score,'-')}</span></div><div class="ai3-x">${cl(p.gpt_analysis)||'N/A'}</div></div><div class="ai3-c"><div class="ai3-h"><span class="ai3-t grk">GROK</span><span class="ai3-s">${n(p.grok_score,'-')}</span></div><div class="ai3-x">${cl(p.grok_analysis)||'N/A'}</div></div><div class="ai3-c"><div class="ai3-h"><span class="ai3-t gem">GEMINI</span><span class="ai3-s">${n(p.gemini_score,'-')}</span></div><div class="ai3-x">${cl(p.gemini_analysis)||'N/A'}</div></div></div>`;

// Meta info
var meta=`<div style="text-align:center;margin-top:5px;font-size:8px;color:var(--t3);font-family:IBM Plex Mono">${p.model_agreement?'✅ CONSENSUS':'⚠️ DIVERGENCE'} | ${p.model_consensus||0}/${p.total_models||11} models | Predict:${ps} | Fusion:${fw.market||'?'}/${fw.model||'?'}</div>`;

return`<div class="mc ${m.is_recommended?'rec':''}" id="m-${m.id}"><div class="mc-h"><div class="mc-lg">${m.league||''} <span class="m">${m.match_num||''}</span>${m.is_recommended?'<span class="mc-badge">ELITE</span>':''}</div><div class="mc-time">${m.match_time||''}</div></div><div class="mc-b"><div class="teams"><div class="tm"><div class="tm-n">${m.home_team}</div></div><div class="sc-c"><div class="sc-big">${p.predicted_score||'?'}</div><span class="res-tag ${rc}">${p.result||'?'}</span></div><div class="tm"><div class="tm-n">${m.away_team}</div></div></div><div class="sr"><div class="sb"><div class="sb-l">TOTAL</div><div class="sb-v" style="color:var(--gn)">${p.expected_total_goals?parseFloat(p.expected_total_goals).toFixed(1):'?'}</div></div><div class="sb"><div class="sb-l">O2.5</div><div class="sb-v" style="color:var(--pr)">${n(p.over_2_5)}%</div></div><div class="sb"><div class="sb-l">BTTS</div><div class="sb-v" style="color:var(--cy)">${n(p.btts)}%</div></div><div class="sb"><div class="sb-l">CONF</div><div class="sb-v" style="color:${p.confidence>=70?'var(--gn)':p.confidence>=50?'var(--am)':'var(--red)'}">${p.confidence||0}%</div></div></div><div class="prob"><div class="prob-h"><span style="color:var(--red)">H ${hp}%</span><span style="color:var(--t2)">D ${dp}%</span><span style="color:var(--bl)">A ${ap}%</span></div><div class="prob-bar"><div class="h" style="width:${hp}%"></div><div class="d" style="width:${dp}%"></div><div class="a" style="width:${ap}%"></div></div></div>${sig}${vbH}${expH}<button class="tbtn" onclick="tg(this,'d-${m.id}')">EXPAND ANALYSIS ▼</button></div><div class="det" id="d-${m.id}">${mdl}${ixgH}${ahH}${orH}${frm}${ai}${meta}</div></div>`}).join(’’)}

function tg(b,id){var e=document.getElementById(id);if(e.classList.contains(‘open’)){e.classList.remove(‘open’);b.textContent=‘EXPAND ANALYSIS ▼’}else{e.classList.add(‘open’);b.textContent=‘COLLAPSE ▲’}}
function go(id){var e=document.getElementById(id);if(e){e.scrollIntoView({behavior:‘smooth’,block:‘center’});var d=document.getElementById(‘d-’+id.replace(‘m-’,’’));if(d&&!d.classList.contains(‘open’))d.classList.add(‘open’)}}
</script>

</body>
</html>