# 三家 AI 的单一配置入口

每家只使用一个模型和一组无编号的 URL/KEY，三家可以指向同一个服务商地址。相同地址不会额外生成模型槽位。

| AI | GitHub Secret：地址 | GitHub Secret：密钥 | 默认模型 |
| --- | --- | --- | --- |
| GPT | `GPT_API_URL` | `GPT_API_KEY` | `熊猫-按量-gpt-6-astra` |
| Gemini | `GEMINI_API_URL` | `GEMINI_API_KEY` | `gemini-3.8-flash-high` |
| Grok | `GROK_API_URL` | `GROK_API_KEY` | `grok-4.7` |

模型默认值统一在 `scripts/config.py` 的 `DEFAULT_MODELS`。现有可选 Repository Variables `GPT_MODEL`、`GEMINI_MODEL`、`GROK_MODEL` 分别覆盖该家的唯一模型；未设置或为空白时使用默认值，不要求新增配置。

旧的 `*_API_URL_2` / `*_API_URL2`、`*_API_KEY_2` / `*_API_KEY2` 等编号配置以及 `*_MODEL_1` 至 `*_MODEL_5` 均不读取。旧 `AI_ENDPOINT_*` 轮询、备用接口和槽位队列已移除，保留在后台的旧变量不会生效。

GPT 和 Grok 独立并行初审，Gemini 随后终审。当前工作流仍每场最多三次请求、无自动重试；并发由场次和请求上限控制，不再由五个接口槽位控制。缺少某家的 URL 或 KEY 时明确记录缺失，绝不借用另一家的配置或旧编号槽位。

这项配置清理不改变服务商对模型的路由和 TLS 连接状态；接口返回 `model_not_found` 或握手失败时仍保留真实故障。历史数据、已保存比分和前端布局不受本次配置整理影响。

## 手动运行与凌晨赛程

在 Actions → Football AI Predict → Run workflow 运行。`target_date` 可填写 `YYYY-MM-DD` 格式的竞彩业务日；留空时按北京时间减 11 小时计算当前业务日。

默认 `date_mode=next_available`：优先选择当前业务日的未开赛比赛；该日没有可用比赛时，选择源数据中未来 7 天内最近的一个业务日，每次仍只分析一天。选择 `current` 则严格保留当前业务日。显式填写 `target_date` 时不自动换日。已开赛比赛和自动选择模式下缺少可靠开赛时间的比赛不进入 AI。

实际选定日期会传递到比赛、输出 `target_date`、历史文件和不可变快照。Actions 摘要会显示请求日期、实际日期、源状态和可用场数；artifact 的 `ai_phase_results/schedule.json` 保存逐场开赛时间、业务日及筛选原因。

源错误或没有可用比赛时仍保留旧线上数据并报错，不以空发布模拟预测成功。接口根地址应包含服务商要求的版本路径，例如 `https://api.luminai.cc/v1`，程序随后追加 `/chat/completions`。

更新密钥、模型权限或服务商通道后，在 Run workflow 勾选“重新请求三家 AI”（`refresh_ai`）。普通运行会在 30 分钟内复用相同证据的结果，包括已记录的失败；密钥值不参与证据缓存标识，因此保存新密钥后直接重跑可能仍显示旧错误。勾选后本轮跳过 AI 持久化缓存，重新请求 GPT、Grok 和 Gemini，仍遵守每场最多三次调用、无自动重试的限制。正常日常运行保持不勾选以节约请求；运行摘要中的实际请求数与缓存数用于区分真实调用和复用。
