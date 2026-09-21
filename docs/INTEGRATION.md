# 团队接入说明

## 推荐集成边界

团队数据管线优先把命令行作为稳定边界，而不是直接依赖内部 Python 函数。核心命令统一由
`neuro-dataset-factory` 提供；下载、完成态交付和 release 管理命令使用
`neuro-dataset-*` 前缀。

核心命令正常时向 stdout 输出 JSON report：

```json
{
  "schema_version": 2,
  "ok": true,
  "stats": {},
  "issues": []
}
```

- `ok=true` 时退出码为 `0`。
- 质量门未通过时 `ok=false`，退出码为非零。
- 参数错误、输入损坏或缺少必需依赖同样返回非零。
- `issues[].code` 适合机器统计；不要依赖人类可读的 `message` 做分支判断。

部分 `ops` 命令会输出它们自己的版本化 report。调度器应以退出码和明确的状态字段为准，
不要用日志字符串判断成功。

## 安装组合

| 使用场景 | 安装命令 | 额外系统要求 |
|---|---|---|
| 本地 profile、打包、audit | `pip install .` | 无 |
| S3 论文语料 | `pip install '.[s3]'` | 可访问对应 endpoint |
| payload 下载 | `pip install .` | `curl` |
| 开发与 CI | `pip install '.[s3,test]'` | 无 |

生产环境应记录 Python 版本、本包版本和最终依赖解析结果。当前包版本为 `0.4.0`，核心数据
契约 `SCHEMA_VERSION=2`；下载器运行状态独立使用 schema version 1。

## 配置与凭据

将 `.env.example` 复制为 `.env`。可用变量：

| 变量 | 使用阶段 | 是否必需 |
|---|---|---|
| `API_BASE`、`API_KEY` | LLM 任务/图片映射 | 对这些阶段必需 |
| `TASKGEN_MODEL` | LLM 阶段默认模型 | 可用 `--model` 替代 |
| `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY` | 私有 S3 语料 | 私有语料必需 |
| `AWS_SESSION_TOKEN` | 临时 AWS 凭据 | 视凭据类型 |
| `SERPER_KEY`、`JINA_API_KEY` | 公开发现 | 可选 |

代码不会自动猜测 S3 endpoint、bucket 或输出根目录。它们必须由团队调度配置显式传入。
推荐由 secret manager 注入 `.env` 或进程环境，不要把真实 `.env` 放入任务镜像或数据交付物。

## 状态机

```text
unverified link
  -> verified_downloadable locator
  -> awaiting_data_download provisional task
  -> downloaded_and_verified payload
  -> reference_pending
  -> reference verified
  -> canonical/audited
  -> enabled
```

格式 audit 通过不代表数据和 reference 已完成。调度器至少同时检查：

1. 数据记录是否为 `verified_downloadable`；
2. payload 下载报告是否 `complete=true`；
3. task/reference 是否 `verified`；
4. canonical package audit 是否 `ok=true`；
5. 任务是否显式 `enabled=true`。

任何缺失或未知状态都应 fail closed。

## 幂等与断点续跑

- 核心 JSON/JSONL 写入使用同目录临时文件和原子替换。
- LLM 生成可通过 `--cache-dir` 和默认 resume 行为恢复；`--refresh-cache` 会重新调用模型。
- 远程验证的临时网络失败记录为 `verification_deferred_network`，应重试而非当作 rejected。
- 下载器使用 `.part` 断点文件和每数据集原子状态报告。
- 下载器的 `--skip-completed-reports` 只做快速恢复；最终发布仍要运行完成态构建/audit。
- 构建完成态交付时拒绝覆盖已存在的输出根目录，调度器应为每次发布使用新目录。

## 文件系统约束

- 输入和输出目录必须由调度器显式创建和挂载。
- canonical BrainArena 的公开 registry 使用相对路径。
- 完成态 payload 交付可能创建指向共享 payload store 的符号链接；打包或跨机器复制时要决定
  是保留链接还是物化链接目标。
- 不要将工作目录、缓存目录和最终发布目录设为同一路径。
- 不要让多个 writer 同时写同一个输出 JSONL；批次可分片，完成后用 merge 工具合并。

## 安全边界

`run-references` 会执行候选中声明的命令，因此默认拒绝执行。只有人工审查命令和代码后，
才允许传入 `--allow-execution`。执行器使用 `shell=False`、独立输出目录和超时，并从子进程
环境移除 API key、token、password、secret 和代理变量。

团队调度器仍应在受限容器中运行 reference，并对 CPU、内存、磁盘、网络和最长运行时间设置
配额；CLI 的安全处理不能代替基础设施隔离。

## CI 最小门槛

每次合并至少执行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m neuro_dataset_factory --help
python -m neuro_dataset_factory.ops.download_remote_payloads --help
python -m neuro_dataset_factory.ops.build_completed_remote_delivery --help
```

发布前再用一个小型、可公开 fixture 完成对应管线的 smoke run。网络 repository API 和私有
S3 的集成测试应单独标记，避免把临时网络故障混入离线单元测试。

## 兼容性约定

- 消费者必须忽略不认识的附加字段。
- 消费者应拒绝高于自身支持范围的 `schema_version`。
- ID、路径、status 和 checksum 字段属于集成契约；修改时应提升 schema version。
- report 的统计字段可以新增，不应作为唯一正确性依据。
- LLM prompt version 与数据 schema version 分开管理。
