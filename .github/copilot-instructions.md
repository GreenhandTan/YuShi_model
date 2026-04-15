# VS Code Copilot Project Instructions

## Git 提交与推送规范
- **全中文提交**：每次导出并推送代码到 GitHub 修改时，Git 提交信息（Commit Message）**必须使用中文**。
- **标准前缀约定 (Conventional Commits)**：
  - `feat:` 新增功能（如导出ONNX、新增训练集）
  - `fix:` 修复 Bug（如修复路径判定残留）
  - `docs:` 文档更新（如 README、规则更改）
  - `chore:` 构建过程或辅助工具变动
- **代理推送规范**：由于本地网络原因，向远程仓库推送时，必须带有本地终端代理环境变量配置（即 `$env:http_proxy='http://127.0.0.1:7897'; $env:https_proxy='http://127.0.0.1:7897';` 链式命令组合）。
