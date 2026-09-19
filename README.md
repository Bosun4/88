# Football AI · v21

足球赛前证据整理、单轮批量 AI 分析与赛后独立统计工具。Python 生成数据，静态网页读取 `data/predictions.json`。默认只处理当前竞彩业务日；北京时间减去 11 小时确定业务日。

## 运行流程

**赛前证据 → 单轮批量 AI → 协议校验 → 主线 / 风险线 → 锁档 → 赛后独立统计**

1. 收集赛程、赔率和可获得的赛前资料；缺失、时间不明或相互冲突的资料要明确标注。
2. `single_pass` 按每批最多 6 场调用一个主模型。每批只调用一次，不互评、不追加修复或备用裁判调用。超时、无有效响应或达到调用上限时显式弃权。
3. 协议层检查返回结构、场次对应和可展示字段；协议通过不等于预测准确或可盈利。
4. 主线保存本次方向和比分；风险线单独呈现不同走势与候选比分，不把风险命中混算为主线命中。
5. 写入实时文件、按业务日历史文件和独立时间戳快照。赛前锁档保留源文件哈希；后续赛果不得反写原预测。
6. 赛后用实际赛果单独评分，区分方向、比分、风险覆盖与弃权，避免同一比赛跨快照重复计数。

升级改动与实际验证边界见 [docs/UPGRADE_V21.md](docs/UPGRADE_V21.md)。

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
| `GPT_API_URL`、`GPT_API_KEY` | 主模型接口与密钥 |
| `GPT_MODEL` | 可选，覆盖已有模型名称；空值沿用代码默认 |
| `API_FOOTBALL_KEY`、`FOOTBALL_DATA_KEY`、`ODDS_API_KEY` | 对应数据源凭证；缺失可能降低证据完整度 |

```bash
AI_RUN_MODE=single_pass \
AI_PRIMARY_MODEL=gpt \
AI_BATCH_SIZE=6 \
AI_CHUNK_CONCURRENCY=2 \
AI_MODEL_CONCURRENCY=2 \
AI_SINGLE_PASS_MAX_CALLS=12 \
AI_CONNECT_TIMEOUT=20 \
AI_READ_TIMEOUT=180 \
AI_STREAM=false \
AI_HTTP_TOTAL_TIMEOUT=180 \
AI_DECISION_CACHE_TTL=1800 \
AI_PERSISTENT_CACHE_ENABLED=true \
VMAX_ALLOW_AUTO_INSTALL=false \
python scripts/main.py
```

该命令会抓取实时数据并可能产生 API 费用。`AI_BATCH_SIZE` 是每批场数，同时受证据字符预算约束，超大批次会拆分；两个并发变量控制在途批次和模型请求。`AI_SINGLE_PASS_MAX_CALLS=12` 是每次运行的硬调用上限；超过上限的待预测场次必须弃权。单次失败不重试、不切换备用端点。旧 `AI_MAX_REQUESTS_PER_AI` 未在旧引擎执行，v21 不再将它作为费用保护。

缓存位于 `data/ai_cache/`，按预测证据哈希和 1800 秒 TTL 决定复用；更新代码或证据后不应复用旧决策。缓存不提交、不发布到 Pages。缓存命中或无赛事不保证一定发起 API 调用。

## GitHub Actions

- **Football AI Predict**：只接受手动 `workflow_dispatch`，仅允许 `main`；先测试，再预测、校验 JSON 并推送本次历史和快照。没有定时器，push/PR 不会触发付费 AI。
- **Offline CI**：push、PR 或手动执行，只有 `contents: read`，不注入 Secrets；运行禁网测试、`pip check` 和 `pip-audit`，失败会阻断作业。
- **Deploy Pages**：仅打包 `index.html`、`assets/` 和公开 JSON 数据；不运行预测。手动预测成功后显式调用部署，因为 `GITHUB_TOKEN` 的推送不会再触发 push 工作流。

在仓库 **Settings → Secrets and variables → Actions** 添加上述 Secrets，模型覆盖使用 Repository Variable `GPT_MODEL`。在 **Settings → Pages** 将部署来源设为 **GitHub Actions**，并确保 `github-pages` environment 允许 `main` 部署。本次本地升级没有修改这些远端设置，也没有触发线上预测或部署。

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
