# Cookbook 模型测试（BW1100）

本分支镜像构建并推送成功后，`release-docker-image.yml` 的 `test-image` job 将本次日期镜像传给 `cookbook-deployment-smoke.yml`，调用 `guochenyang99/ci-model-test` 的主线控制器，测试 Cookbook 中对应版本的 BW1100 单机部署。

模型测试与 SGLang 共用服务启动、HTTP 问答、资源检查、容器清理、Allure / Markdown 报告及 DashHub 上报。只进行部署与接口测试，不运行旧 CItest 的精度评分，也不发送旧飞书通知。

旧 `daily-test-docker-image.yml` 仍接受 `docker_image` 输入，转调同一新流程。手动预检查应使用 Cookbook 入口的 `mode: preflight`，确认后再 `execute`；`cases` 留空测试本版本范围内的全部单机部署。

`v0.25.1` 分支固定测试 0.25.1，`v0.28.1-dev` 分支固定测试 0.28.1。两个分支都只使用 `bw1100` Runner，暂不接入 BW1000。

仓库需授权 `MODEL_SMOKE_REPO_TOKEN`（只读拉取测试仓库）与 `DASHHUB_CI_TOKEN`（结果上报）。构建步骤、框架版本、DTK / Python 配置、定时调度和 `validate-image` 镜像校验保持原样。

验收时核对本次镜像地址、报告中的组件版本、模型状态，以及 DashHub 上报步骤的 HTTP 成功返回。已有运行不会自动切换到新流程，历史结果也不会自动重传。
