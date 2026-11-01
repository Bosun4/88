# GPT 竞彩足球升级审计与验收

日期：2026-11-01。基于最新 `main` 的 `ef07722536a8bdab7887c273dc15f0c48babc12c`，分支 `feat/gpt-jingcai-research`。重新 fetch 后确认 main 未前移。本次仅修改代码、测试、文档及研究报告，没有执行生产预测、合并或部署。

## 已实现

- 活动配置及传输只允许 GPT；入口仅 panel。两个独立分析角色和一次终审分开保存状态、输出和缓存。失败不伪造角色成功，不用初审顶替失败终审。
- 每场预留3次、每轮硬上限180次、阶段启动5400秒、单请求180秒；保留2场/4请求并发，不自动重试。刷新缓存入口跳过读取并保存新响应。开赛前/发布前重检，全轮失败保留线上旧数据。
- 五玩法原始报价及完整市场成本核算，所有预测概率保持null；比分缺“其他”不算完整抽水。开盘/庄家视角须有替代解释、反证和失效条件。
- 独立报价索引、来源/时间边界、首采与初盘区分、回溯导入、精确赛事去重、相似历史正反例检索。统计只作历史比较，不冒称真实赛前成绩。
- API-Football 精确赛事匹配后补充近30天/近5/10场（最多90天）、主客场与对手、90分钟比分、赛程间隔、伤停首发和常规时间红牌。缺密钥、字段、匹配、来源权限均有诊断。
- 深色页面保留列表、主线、风险和五步摘要；按三个 GPT 角色展示真实调用状态与缺口，折叠报价/历史/来源/诊断。历史旧模型记录使用兼容读取。

## 运行审计

[失败运行 38746022926](https://github.com/Bosun4/88/actions/runs/38746022926) 的公开状态只能确认失败发生于 Run Predictions。未取得授权后的完整步骤日志；不能认定根因是密钥、模型、TLS、JSON或预算。本次实现补充明确诊断，但没有宣称已复现原故障。

`predict.py` 保留历史兼容代码和多次定义的旧函数。清单见 [legacy-definitions-v25.json](legacy-definitions-v25.json)。当前最终绑定的 `build_evidence_packet` 在 panel 下直接走 `score_policy.build_evidence`；`run_ai_native_web` 仅走 `panel.run_panel`；score-first 输出直接走 `score_policy.adapt_prediction`。回归设置旧数学编译器为禁止调用，核验当前链路未被同名定义覆盖。未进行大规模删除旧函数，以免破坏离线历史回归；旧提示内容不作为当前分析提示，也不存在非 GPT 传输后备。

角色隔离覆盖传输 phase、缓存 role、role_reports，避免同一个 provider 下互相覆盖。传输异常只保留类型/阶段；HTTP错误正文和解析片段不输出到日志；配置日志不再显示密钥片段或含查询串地址。请求不跟随重定向。网页文字转义，来源链接只允许无内嵌凭据的HTTP(S)，并带noopener/noreferrer。

## 数据覆盖及限制

整理214份归档（118 history + 96 snapshots），有效报价150场/827条。文件日期范围2026-05-06至2026-10-13，但可用报价赛事只覆盖2026-09-19至2026-10-13。独立可核验赛果0场，明确初盘0条，真实可用历史比较案例0；没有伪造案例或补充“全历史开盘规律”。逐文件缺口见 [history-coverage-v25.json](history-coverage-v25.json)。

公开 Sporttery 读取探测返回HTTP567，失败未形成成功检查点，详见 [public-source-v25.json](public-source-v25.json)。已提供带来源/时间的文件导入和按日期续传；公开字段适配只通过离线夹具，真实当前接口格式及覆盖范围仍未验证。API-Football 的真实付费权限/赛事覆盖、本项目的真实 GPT 服务商响应亦未测试。

`complete_jingcai_history=false`。本次没有宣称预测准确率提升。模拟输出只验证工程流程；历史补录、案例比较与前向锁档成绩各自保留统计口径。

## 验收

环境：Windows、Python3.12.14；运行依赖均按 requirements.txt 锁定。测试禁止外网，仅允许本地事件循环所需回环。

```sh
python -m pytest -q --disable-socket --allow-hosts=127.0.0.1,::1
node tests/frontend_fixtures.cjs
python -m compileall -q scripts forward_ledger
python -m pip check
python -m pip_audit -r requirements.txt --no-deps --disable-pip --format json
git diff --check
```

结果：**453项Python测试、19项前端测试通过**；编译、依赖一致性、工作流断言及差异空白检查通过。Linux CI 使用现有 `--allow-unix-socket` 配置；远程 CI 结果须在 PR 提交后确认。

覆盖 GPT-only 配置、角色隔离、三次调用、整场预算预留、上限不可被环境变量提高、部分失败/全部失败/终审失败、请求异常/超时、截断及非法JSON、刷新缓存、分析中开赛；数据覆盖跨日、同名异场、重复/冲突赛果、未来证据、缺失初盘、乱序报价、整数竞彩让球、90分钟口径、7+、半全场及比分其他。

浏览器使用隔离目录的明确标注演示数据，未改线上预测：1440×1000 桌面、390×844 手机通过；内容宽度未超视口，无控制台错误；筛选由2场变1场、明细折叠、来源链接及缺失角色展示正常。既有18项前端历史行为测试继续通过。

依赖初扫有3个包的已知漏洞记录，已更新 aiohttp 3.14.4、multidict 6.9.1、certifi 2026.10.19，复查无已知漏洞。依据：[aiohttp发布](https://github.com/aio-libs/aiohttp/releases/tag/v3.14.4)、[multidict发布](https://github.com/aio-libs/multidict/releases/tag/v6.9.1)、[certifi公告](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2)。原始审计结果见 docs/security 下20261101报告。

历史数据与账本的Git内容哈希对比见 [preservation-v25.json](preservation-v25.json)，变更数为0。测试和预览均使用临时目录。研究索引被.gitignore排除，Pages不打包该目录或模型缓存。

## 发布前仍需完成

代码通过功能分支 PR 供审阅，远程 CI 状态以 PR 的检查结果为准。真实接口试运行应由用户另行手动触发，验收响应格式、权限、延迟、数据覆盖与成本后再评估上线效果。本升级不合并、不触发生产预测或部署。
