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
