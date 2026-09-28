# Football AI · v23

足球赛前证据整理、GPT/Grok 独立初审、Gemini 终审与赛后独立统计工具。Python 生成数据，静态网页读取 `data/predictions.json`。默认只处理当前竞彩业务日；北京时间减去 11 小时确定业务日。

## 运行流程

**赛前证据 → GPT + Grok 并行初审 → Gemini 终审 → 协议校验 → 主线 / 风险线 → 锁档 → 赛后独立统计**

1. 收集赛程、赔率和可获得的赛前资料；缺失、时间不明或相互冲突的资料要明确标注。
2. 默认 `panel` 每场 GPT 先分析球队、赛程与比分情景，Grok 独立审查反向、平局与高比分风险；两者并行。再对照让球、总进球、正确比分与半全场报价。至少一份初审有效才交给 Gemini 终审，两份都失败则直接弃权。终审失败保留初审观点，但不将初审比分冒充最终预测。
3. 协议层检查返回结构、场次对应和可展示字段；协议通过不等于预测准确或可盈利。
4. 主线保存本次方向和比分；风险线单独呈现不同走势与候选比分，不把风险命中混算为主线命中。
5. 写入实时文件、按业务日历史文件和独立时间戳快照。赛前锁档保留源文件哈希；后续赛果不得反写原预测。
6. 赛后用实际赛果单独评分，区分方向、比分、风险覆盖与弃权，避免同一比赛跨快照重复计数。

本次历史依据与设计见 [恢复设计](docs/superpowers/specs/2026-09-28-pipeline-recovery-design.md)。旧版单模型 `single_pass` 模式仍可显式启用，v21 记录保留在 [docs/UPGRADE_V21.md](docs/UPGRADE_V21.md)。

## v23 比分与报价风险边界

生产 `panel` 不向模型发送 1X2 逐方向赔率、去水/Shin 概率、泊松结果、旧版专家推荐或经验比分锚，也不经过旧版弱主队尾部改写。主平客方向仅由终审比分派生；主平客概率与比分概率保持空值，不生成投注金额建议。

三项报价只参与整组成本计算，对调主客赔率不改变比分证据。完整互斥报价的 `overround = Σ(1/赔率) − 1`，理论留存率 `hold = 1 − 1/Σ(1/赔率)`；两者不同，也不等于机构实际利润。逐场、逐玩法计算，不套固定 13%。缺完整选项时保持未知，正确比分表缺“其他”结果时不估整盘抽水。

所谓诱盘、过热只能作为需要反证的假设：记录可观察报价、其他合理解释、反证和待确认信息。同一机构不同玩法不是独立多源确认；变化码不等于连续报价或资金流。总进球 7 选项按 7+ 理解，竞彩让球按主队 90 分钟进球加让球值后结算。

原配置保持：GPT `gpt-5.6-sol`、Grok `熊猫-A-10-grok-4.6`、Gemini `熊猫-顶级特供-X-17-gemini-3.1-pro-preview-联网`；URL/KEY 继续读取原 Secrets，不增加备用端点。

## 安装与离线检查

使用 **Python 3.12**。以下命令从仓库根目录运行，Shell 示例采用 Bash（Windows Git Bash 可用）。

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows Git Bash 改为:
# source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install pytest==9.1.1 pytest-socket==0.8.1 PyYAML==6.0.3 pip-audit==2.10.1
python -m pip check
python -m pytest -q --disable-socket --allow-unix-socket
python -m pip_audit -r requirements.txt
```

Windows 的 asyncio 使用回环 TCP 创建内部管道，本机测试改用：

```bash
python -m pytest -q --allow-hosts=127.0.0.1,::1 --allow-unix-socket
```

测试使用固定输入或模拟调用，不需要 AI 凭证。Linux CI 禁止测试打开 TCP socket；依赖安装与漏洞库查询阶段需要联网。`requirements.txt` 固定全部运行时依赖版本，移除了 `deep-translator`；未命中队名显式映射时保留原名称，不依赖隐式在线翻译。

## 手动预测

先通过安全的环境配置设置凭证，切勿提交密钥：

| 环境变量 | 用途 |
| --- | --- |
| `WENCAI_AUTHORIZATION` | 当前主赛程接口认证 |
| `GPT_API_URL`、`GPT_API_KEY` | 主线初审接口与密钥 |
| `GROK_API_URL`、`GROK_API_KEY` | 独立反证初审接口与密钥 |
| `GEMINI_API_URL`、`GEMINI_API_KEY` | 终审接口与密钥 |
| `GPT_MODEL`、`GROK_MODEL`、`GEMINI_MODEL` | 可选模型名称；空值沿用 `scripts/predict.py` 中 endpoint slot 默认值 |
| `API_FOOTBALL_KEY`、`FOOTBALL_DATA_KEY`、`ODDS_API_KEY` | 对应数据源凭证；缺失可能降低证据完整度 |

```bash
AI_RUN_MODE=panel \
AI_BATCH_SIZE=1 \
AI_CHUNK_CONCURRENCY=2 \
AI_MODEL_CONCURRENCY=4 \
AI_PANEL_MAX_CALLS=180 \
AI_PANEL_MAX_SECONDS=5400 \
AI_CONNECT_TIMEOUT=20 \
AI_READ_TIMEOUT=180 \
AI_FINAL_READ_TIMEOUT=180 \
AI_STREAM=true \
AI_HTTP_TOTAL_TIMEOUT=180 \
VMAX_FETCH_DAYS_AHEAD=0 \
AI_DECISION_CACHE_TTL=1800 \
AI_PERSISTENT_CACHE_ENABLED=true \
VMAX_ALLOW_AUTO_INSTALL=false \
python scripts/main.py
```

该命令会抓取实时数据并可能产生 API 费用。`panel` 每场最多三次调用，按整场预留预算，默认最多 60 场、180 次请求、两个场次与四个模型请求并发。90 分钟后停止新阶段，已发请求仍受 180 秒总超时限制。每阶段调用前重新核验开赛时间。单次失败不重试、不切换备用端点、不追加修复或备用裁判。赛后复盘只做客观对账，不暗中调用模型。

`runtime.ai_run` 和 `data/ai_phase_results/last_run.json` 记录有效终审数、弃权数、调用次数、缓存命中及阶段状态。全部失败时工作流报错并保留上一份线上预测；部分成功时明确展示完成率。页面中的模型分歧、主线、风险候选和盘口推导尾部各自标明。

TLS 握手、证书验证、DNS 与连接失败分别分类；相同接口地址发生连接故障后，停止本轮尚未发出的同域请求，最多保留已在途请求。HTTP 模型错误不触发整站停用，避免单个 Grok 服务故障阻断 GPT/Gemini。GitHub 运行摘要与 artifact 提供失败分类和次数；证书验证始终开启。2026-09-28 的 #695 在成功采集 8 场后，16 次初审全部在 TLS 握手失败，尚未进入模型推理，此类外部故障不能靠更换预测公式修复。

缓存位于 `data/ai_cache/`，按语义证据、模型、接口、提示与生成参数的哈希及 1800 秒 TTL 决定复用。重复采集时钟不使缓存失效，报价和开赛时间改变会失效；失败也短期缓存，避免重复扣费。缓存不提交、不发布到 Pages。缺失伤停保持未知，模型自述网页搜索不视为已检索事实；没有真实大小球报价时，不从总进球选项合成可交易赔率。

## GitHub Actions

- **Football AI Predict**：只接受手动 `workflow_dispatch`，仅允许 `main`；先测试，再预测、校验 JSON 并推送本次历史和快照。没有定时器，push/PR 不会触发付费 AI。
- **Offline CI**：push、PR 或手动执行，只有 `contents: read`，不注入 Secrets；运行禁网测试、`pip check` 和 `pip-audit`，失败会阻断作业。
- **Deploy Pages**：仅打包 `index.html`、`assets/` 和公开 JSON 数据；不运行预测。手动预测成功后显式调用部署，因为 `GITHUB_TOKEN` 的推送不会再触发 push 工作流。

在仓库 **Settings → Secrets and variables → Actions** 添加上述 Secrets，模型覆盖使用对应 Repository Variables。在 **Settings → Pages** 将部署来源设为 **GitHub Actions**，并确保 `github-pages` environment 允许 `main` 部署。测试全部使用模拟接口；真实模型可用性与当日采集情况以手动运行日志和 `runtime.ai_run` 为准。完整流程运行成功不代表预测准确率或收益改善，须继续用赛前锁档样本评估。

## 锁档与赛后验证

`main.py` 发布路径保存在 `runtime.history_path` 与 `runtime.snapshot_path`。历史文件同一天同一时段可能更新；独立快照用于保留各次原始输出。

锁档模块的输入要求及参数以 CLI 为准：

```bash
python -m forward_ledger.cli lock --help
python -m forward_ledger.cli score --help
python scripts/post_review.py --help
```

对经过格式核对的赛前预测文件创建新的专属台账路径，再以同一台账评分：

```bash
python -m forward_ledger.cli lock --pred path/to/prematch.json --out path/to/new-ledger.jsonl
python -m forward_ledger.cli score --ledger path/to/new-ledger.jsonl --actual path/to/verified-actuals.csv --out_csv path/to/scored.csv --out_md path/to/scored.md
python scripts/post_review.py path/to/prematch-snapshot.json path/to/verified-actuals.json --output path/to/review.json
```

这些是命令格式示例，不代表仓库自带对应新赛季赛果。使用新的输出路径，先确认比赛 ID、开赛时间、赛季和赛果来源匹配。历史原型研究与回填结果不能代替前瞻样本成绩。

## 目录

| 路径 | 内容 |
| --- | --- |
| `scripts/` | 取证、预测、适配和赛后审计入口 |
| `forward_ledger/` | 赛前锁档与独立评分 |
| `index.html`、`assets/` | 静态看板 |
| `data/` | 当前预测、历史与快照，保留用户历史记录 |
| `tests/` | 离线回归与工作流安全约束 |
| `docs/security/` | 本次依赖审计原始 JSON |
| `legacy/archive-20260919/` | 旧备份和世界杯研究原件、归档哈希清单 |

旧世界杯赛前资料只作历史证据保存，不应将旧赛事阶段、阵容或固定日期假设应用到当前比赛。
