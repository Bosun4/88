# 项目88 审计修复 · 前后对比报告 (2026-06-21)

> 分支: `fix/score-shape-and-draw-guard-20260621` (off `fix/divergence-source-overdamp-v2`)
> 备份: `/root/backups/88_predict.py.20260621_000643`
> 触发审计: `reports/audit_real_results_live_20260620.md` (现网 6/20 完赛4场, 方向3/4准但比分0/4, 错误全是"给负方安慰球" + 唯一翻车落在低信心平局)
> 落地军规: 已 pytest 全回归 (126 passed)；commit/push **待用户单独批准**，未提交。

## 一、本轮三项修复 (用户裁决: 都要 / Q1=C Q2=C Q3=A Q4=skip Q5=safe)

| # | 修复 | 类型 | 位置 |
|---|---|---|---|
| Fix1 | 负方安慰球压缩 (强弱悬殊+高信心胜场, 4-1→4-0/3-1→3-0) | **确定性后处理 gate** | `_apply_lopsided_consolation_goal_gate` |
| Fix2 | AI 提示词新增"负方安慰球审计"规则 (默认负方0球, 有破门证据才留1球) | 提示词 (非确定性) | 两处 final-judge prompt 块 |
| Fix3 | 低信心平局降 observe (draw 且 conf<45) | **确定性硬闸** (复用 `_cap_to_observe`) | `_apply_low_confidence_draw_guard` |

三个 gate 均接入编排链 (contrarian gate 之后、selection layer 之前)，各带 try/except 隔离，异常只写 warning 不中断。

### Fix1 触发条件 (保守, 防 n=4 过拟合)
- final_direction ∈ {home, away} 且 胜方信心 ≥ 70
- 负方进球 == 1 且 胜方进球 ≥ 3 (净胜≥2 的"安慰球"形态)
- 动作: 负方 1→0；**绝不改方向** (改后方向变了则放弃)；同步 goal_band/btts/top3[0]；写 `validation_warnings` 溯源
- **不动推荐等级** (只改比分形态)

### Fix2 提示词规则要点
- 胜方信心≥70且净胜≥2 → 默认负方进球=0 (优先 N-0 而非 N-1)
- 仅当负方有独立破门证据(对攻战意/快反/定位球/客场不弃赛+xG)才保留1球
- 明确**不与大球带冲突**: 只压"负方安慰球", 不阻止双方对攻的对称大比分(2-2/3-2/3-3)

### Fix3 触发条件
- final_direction == draw 且 draw 信心 < 45 → `_cap_to_observe` (tier 封 C, 不推荐, bet_action=observe, gate 不过)

## 二、前后对比 (同一批现网预测 × 6/20 真实赛果, n=4)

| 指标 | BEFORE | AFTER | 变化 |
|---|---|---|---|
| 方向命中 | 3/4 | 3/4 | 持平 (修复不动方向, 符合设计) |
| 精确比分 | 0/4 | 0/4 | 持平 |
| 进球带命中 | 1/4 | 1/4 | 持平 |
| **BTTS 命中** | 0/4 | **1/4** | **+1** ✅ |
| 翻车计入实单 | 是(土耳其) | **否(→observe)** | ✅ 风控生效 |

### 逐场 AFTER
| 场次 | 预测 | 实际 | 方向 | 比分 | 动作 |
|---|---|---|---|---|---|
| 美国vs澳 | 2-1主 | 2-0主 | ✓ | ✗ | 未触发(conf55<70, 保守不误伤) |
| 苏格兰vs摩 | 1-2客 | 0-1客 | ✓ | ✗ | 未触发 |
| 巴西vs海 | **4-0**主(原4-1) | 3-0主 | ✓ | ✗ | **比分校准** (conf82, 净胜3) |
| 土耳其vs巴 | 1-1平 | 0-1客 | ✗ | ✗ | **→observe** (conf41<45) |

## 三、诚实结论 (不夸大)
- **真实收益**: ①BTTS 0/4→1/4 (不再虚判海地破门); ②比分形态更贴近现实(4-1→4-0, 离3-0更近); ③唯一方向翻车(土耳其)从"实单"剔除为观察。
- **精确比分仍 0/4**: 巴西校准后 4-0 vs 实 3-0, gate 只压了负方安慰球、没动胜方虚高进球(动胜方侧会在 n=4 上过拟合, 故意不做)。比分形态偏差需更大样本才值得进一步调参。
- **方向 3/4 不变**: 三项修复刻意都不改 final_direction, 零方向回归风险。
- **样本仅 4 场**: 所有数字为指示性, 待 033-035 及后续完赛扩样本复核。

## 四、回归验证
- `python3 -m py_compile scripts/predict.py` ✅
- `PYTHONPATH=. .venv/bin/pytest tests/ -q` → **126 passed in 0.78s** ✅ (无新增失败)

## 五、改动文件清单 (待 commit, 仅精确 add)
- `scripts/predict.py` (生产码: 3处新增, 编排链接入)
- 报告/数据 (untracked, 按需 add): `reports/audit_fix_before_after_20260621.json`, 本报告

## 六、指纹浏览器说明
本轮按裁决 **Q4=skip**。修复为离线改 Python+核验赛果, 指纹浏览器(AdsPower, 用于打开网站环境)与此任务无关; 且其 Local API 当前 down (ECONNREFUSED 127.0.0.1:50325), 即使想用也不可用。
