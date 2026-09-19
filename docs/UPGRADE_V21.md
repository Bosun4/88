# v21 升级说明（2026-09-19）

## 默认调用约束

工作流与主代码按同一组配置对齐：

| 配置 | 默认值 / 语义 |
| --- | --- |
| `AI_RUN_MODE` | `single_pass`，单阶段分析 |
| `AI_PRIMARY_MODEL` | `gpt` |
| `AI_BATCH_SIZE` | `6`，真正的每批场数 |
| `AI_CHUNK_CONCURRENCY` | `2`，并行批次上限 |
| `AI_MODEL_CONCURRENCY` | `2`，模型请求并发上限 |
| `AI_SINGLE_PASS_MAX_CALLS` | `12`，每次运行硬调用上限；超出显式弃权 |
| `AI_CONNECT_TIMEOUT` | `20` 秒 |
| `AI_READ_TIMEOUT` / `AI_HTTP_TOTAL_TIMEOUT` | `180` / `180` 秒 |
| `AI_PHASE1_RETRY_MAX` / `AI_FINAL_RETRY_MAX` | `0` / `0` |
| `AI_ENDPOINT_FAILOVER` | `false`，单次失败不换备用端点 |
| `AI_PERSISTENT_CACHE_ENABLED` | `true` |
| `AI_DECISION_CACHE_TTL` | `1800` 秒，证据哈希命中才复用 |
| `GPT_MODEL` | 可由环境变量 / Repository Variable 覆盖，空值沿用原模型名 |

不再配置旧 `AI_MAX_REQUESTS_PER_AI`：旧代码只记录该变量，没有执行请求限流，不能据此声称能控制费用。单轮模式不互评、不追加一致性裁判、不调用备用裁判、不循环修复 JSON。

Actions 缓存按源码和依赖哈希分组，每次运行保存新 cache key；数据提交改变 Git SHA 不会直接令同一代码的缓存失效。最终是否命中仍由本地证据哈希、模型、端点、提示版本和 TTL 判断。缓存及调试文件不提交，也不打包到 Pages。

## 依赖安全

运行时由 5 个直接依赖和 15 个间接依赖组成，全部使用精确版本：

- `aiohttp 3.13.5 → 3.14.3`，PyPI 实际可安装；修复范围以 [上游变更记录](https://github.com/aio-libs/aiohttp/blob/master/CHANGES.rst) 与 [GHSA-cq5v-8q36-5273](https://github.com/advisories/GHSA-cq5v-8q36-5273) 为参考。
- 保留并锁定 `structlog 25.5.0`、`requests 2.34.2`、`numpy 2.5.3`、`pandas 3.0.6`；这些是本次解析、安装、检查过的版本。
- 移除 `deep-translator`，同时不再安装它引入的 HTML 解析依赖。原抓取模块的可选回退导入会在缺包时返回原队名；已映射队名不受影响。未知队名需要维护显式映射。

原始审计结果保存在 [升级前](security/pip-audit-before-20260919.json) 和 [升级后](security/pip-audit-after-20260919.json)。工具版本为 `pip-audit 2.10.1`，使用默认漏洞服务，不忽略漏洞：

| 扫描 | 解析包数 | 输出漏洞条目 | 按包名 + 漏洞 ID 去重 |
| --- | ---: | ---: | ---: |
| 原 requirements | 23 | 29 | 15（aiohttp 14、deep-translator 1） |
| 升级后完整锁定 | 20 | 0 | 0 |

升级前数据库响应包含重复编号，故 29 条不代表 29 个不同漏洞。复扫为检查当时“未发现已知漏洞”，不是安全无缺陷保证。

实际执行：全新 Windows Python 3.12 虚拟环境安装成功，`pip check` 返回 `No broken requirements found.`；Linux CPython 3.12 的全部运行时 wheel 已下载解析成功。wheel 下载仅证明依赖在目标平台可解析，不等于已在 Linux 上执行测试。

## GitHub Actions 边界

- `predict.yml` 仍只接受手动触发，并限制 `main` 分支；测试成功后才能执行付费调用。
- `ci.yml` 增加只读、无 Secrets 的 push / PR 测试与依赖审计。测试阶段禁 TCP socket；失败不会 `continue-on-error`。
- `pages.yml` 只发布静态网页、前端资源和公开 JSON。既可由 main 的页面/数据变动触发，也供手动预测成功后复用调用；不运行 AI。
- 发布时校验本次 `runtime.history_path` 和 `runtime.snapshot_path` 指向的确切文件，拒绝仓库外路径、冲突标记、非标准 JSON；同时用 Node `JSON.parse` 验证浏览器兼容。远端 main 已前进时拒绝发布旧基底结果。
- Pages 必须在远端设置为 GitHub Actions 来源。前向账本 `data/forward_ledger.jsonl` 随预测产物校验、提交，并纳入故障 artifact；不对外打包到 Pages。远端验收状态见下文。

## 归档

将两个顶层 `.backup*` 目录及 `reports/wc_research/` 共 24 个文件迁到 [legacy/archive-20260919](../legacy/archive-20260919/README.md)。逐文件 SHA-256 与字节数写入 [manifest.json](../legacy/archive-20260919/manifest.json)，移动后校验全部一致。

没有移动或删除 `data/` 的预测历史、快照、校准台账和用户数据。研究脚本保留原字节，因此内部旧相对路径仍存在；从归档根目录运行前须重新核验数据源与日期。旧资料不是当前世界杯阶段或实时队伍状态的依据。

## 集成验收与边界

最终集成验收记录见 [VALIDATION_V21.md](VALIDATION_V21.md)，覆盖后端、前端、浏览器、工作流、依赖及发布到结算的闭环。旧并行过程的失败数量不代表最终状态。

Windows 使用 `--allow-hosts=127.0.0.1,::1` 允许 asyncio 内部回环管道；Linux CI 保持 `--disable-socket --allow-unix-socket`。离线测试不发送付费 AI 请求。

单轮输入保留完整比分报价、胜平负、总进球、联赛与来源，删除重复的相邻比分表格；相邻比分推理在公共指令中明确要求。默认每批最多 6 场，另受 60000 字符限制。模型输出必须是完整严格 JSON；截断不修补，不额外调用。可用 `AI_STREAM=true` 使用流式响应，只有正常结束的 assistant content 才能进入校验，推理片段不会混入结果。

比分绝对概率的单位始终为 0–100，包括小于 1 的值；未知保存为 null，不补为零。胜平负历史小数输入只按完整向量推断单位。风险 D、主比分及副文风险分别保留和评估。

联赛战意模块仅以有来源的当前赛季积分、轮次、分差、赛程/休息及明确轮换证据为依据；未知保持 unknown。默认单轮不使用世界杯规则；历史路径单独保留。
