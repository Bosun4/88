# v21 集成验收（2026-09-19）

基底：`79264aa120f9a64ec064cbbcb9a91261d1f6715a`。机器可读证据摘要：[validation-v21.json](validation-v21.json)。

## 已实测通过

- Windows Python 3.12：`python -m pytest tests -q --allow-hosts=127.0.0.1,::1 --tb=short`，349 passed，无跳过或预期失败。
- Node：`node tests/frontend_fixtures.cjs`，9 passed。
- Chromium 桌面 1280px / 手机视口：26场完整显示、D筛选6场；历史快照不计今日、无有效推荐；筛选、重置、展开可用，无横向溢出和页面脚本错误。
- `pip check` 无依赖冲突；`pip-audit` 扫描20个锁定依赖，未发现已知漏洞。
- actionlint 对三个工作流退出0；本机未启用 shellcheck/pyflakes，发布步骤内嵌 Python 另行语法检查通过。
- 24个归档文件逐一核验 SHA-256；Git 内容核验仅涉及 Windows 换行差异，预测历史 data/ 未修改。
- 发布→首份锁档→重复发布→结算集成测试通过；同赛事重复发布不会覆盖账本；主线和副文分别计算。

## 14项审计修复映射

| 编号 | 修复 | 验证 |
|---|---|---|
| 01 | 以赛事身份去重；无法确定身份不凭队名合并或结算 | test_evaluation_identity / test_actual_binding |
| 02 | 来源命名空间、赛季、开赛时间和主客身份校验 | test_actual_binding / test_evaluation_identity |
| 03 | 缺报价不计算可执行ROI，严格前向须有有效报价时间 | test_metrics_ledger / test_forward_scoring_audit |
| 04 | 推理前及发布前复查可验证开赛时间 | test_single_pass |
| 05 | 删除估算赛季统计，缺失保持unknown | test_fetch_provenance |
| 06 | 国际赔率须绑定确切事件、时间及原始来源 | test_global_odds_identity |
| 07 | 完整方向概率与比分绝对概率传递；0.5保持0.5%，null保持未知 | test_single_pass |
| 08 | 识别 matches.today，哈希链首份锁档并随CI数据持久化 | test_forward_lock_audit / test_workflow_safety |
| 09 | 缺席模型不继承终审预测；显示真实批次和行状态 | test_single_pass / frontend_fixtures |
| 10 | 运行互斥、按批隔离状态、按证据缓存、硬预算 | test_single_pass |
| 11 | 今日按北京时间实际日期计算，旧快照禁推荐 | frontend_fixtures / 浏览器 |
| 12 | 前端未知概率不补33/33/34，保留真实零 | frontend_fixtures |
| 13 | self_learn支持真实matches结构，赛果精确绑定 | test_review_audit |
| 14 | 升级aiohttp并移除deep-translator，锁定全依赖树 | pip check / pip-audit / test_workflow_safety |

联赛证据另由 test_league_context 覆盖赛季、积分差、轮次、争冠/欧战/保级、休息时间、密集赛程、轮换来源和过期拒绝。默认单轮路径不应用世界杯规则；风险D及副文分析继续保留。

独立代码复核提出两项问题：CI未持久化账本、比分百分比再次放大。两项均已修复并增加回归；概率回归在修复前9项失败，修复后全部通过。最终限定范围独立复核通过；另以独立进程发现并修复旧fast_batch模式初始化NameError，四种模式导入回归均通过。归档的CRLF原字节由.gitattributes保留，暂存Git内容逐文件哈希与清单一致；活跃代码diff检查通过。

## 真实接口：未通过成功生成验收

使用本机已有 `https://aihub.top/v1` / `gpt-6-astra`，明确以历史两场数据验证传输，未改为今日预测或发布。

| 有效认证实验 | 请求 | 结果 | 重跑请求 |
|---|---:|---|---:|
| 非流式原证据 | 1 | HTTP 524，125.766秒 | 0，缓存命中 |
| 非流式压缩证据 | 1 | HTTP 524，125.578秒 | 0，缓存命中 |
| 流式压缩证据 | 1 | 180.625秒超时 | 0，缓存命中 |

此前测试脚本未解析环境变量引用导致401，已更正；401不视为模型接口成功。零重试和失败缓存已真实验证，**有效预测响应、实际成功耗时尚未验证**。流式为可选配置，默认关闭。没有伪造响应、没有将失败历史回放发布成今日预测。

## 数据与远端边界

- 本机未发现问财认证配置，实时赛程抓取未验收；测试夹具验证不等于实时服务可用。
- 历史基准实跑：1480条输入版本；在缺少可验证身份时保留原记录，范围内720条未结算，已结算0，严格前向样本0。当前不能据此声明命中率、ROI或盈利能力。
- GitHub HTTPS凭据缺失，SSH认证失败；非交互push预检失败。未运行远端CI或部署Pages。
- 只读核验远端main仍为上述基底，线上页面HTTP 200且尚无新dashboard脚本。**本地测试通过不代表线上已更新。**

## 复核命令

```bash
python -m pytest tests -q --allow-hosts=127.0.0.1,::1 --tb=short
node tests/frontend_fixtures.cjs
python -m pip check
python -m pip_audit -r requirements.txt
```

GitHub Actions采用Linux禁网回归，仅手动触发预测入口会调用付费AI；页面部署入口不调用AI。
