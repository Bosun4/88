# 小鱼儿 · GPT 竞彩足球研究

v25 使用同一 GPT 进行盘口/历史和近况/风险两组独立分析，再由 GPT 终审。覆盖胜平负、竞彩让球、比分、总进球和半全场；保留原始报价与证据缺口，不输出预测概率或数学比分模型。

- [配置与运行](docs/AI_CONFIGURATION.md)
- [来源、时间边界与历史导入](docs/RESEARCH_DATA.md)
- [升级审计及验收](docs/AUDIT_V25.md)

Python 3.12：`python -m pip install -r requirements.txt`。离线回归：`python -m pytest -q --disable-socket --allow-unix-socket`（Windows用`--allow-hosts=127.0.0.1,::1`），前端：`node tests/frontend_fixtures.cjs`。测试工具版本见 `.github/workflows/ci.yml`。

预测工作流仅允许手动在 main 运行，会使用现有密钥并发布数据。功能分支和 PR 不运行预测、不部署。真实 GPT 接口与公开历史源的未验证范围在审计报告中明确列出。
