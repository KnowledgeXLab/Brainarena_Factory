# 数据契约

本页描述团队调度器最常接触的边界。具体字段验证仍以 `contracts.py`、各 stage validator 和
对应测试为准。

## 通用规则

- JSONL 每个非空行必须是一个 JSON object。
- 文本使用 UTF-8；写入采用结尾换行。
- 核心 report 包含 `schema_version`、`ok`、`stats`、`issues`。
- 所有路径字段都应明确是相对哪个 root；BrainArena 公共 registry 不允许生成机器绝对路径。
- checksum 应携带算法；没有可靠 checksum 时至少记录远程大小，并在下载完成后计算 SHA-256。
- 消费者忽略新增字段，但拒绝未知的更高 schema version。

## 本地 canonical 流程

| 文件 | 生产者 | 关键字段 | 消费者 |
|---|---|---|---|
| `dataset_seeds.jsonl` | 人工/采集系统 | `dataset_id`、`version`、`local_path`、`license` | `profile` |
| `dataset_registry.jsonl` | `profile` | 数据版本、manifest、tree hash、科学元数据 | 论文、生成、打包阶段 |
| `verified_paper_links.jsonl` | `validate-links` | `link_id`、版本匹配、data support、证据 | 生成、打包阶段 |
| `task_candidates*.jsonl` | LLM + dedup + reference | `task_id`、query、required files、reference、rubric | reference、打包阶段 |
| `reference_results.jsonl` | `run-references` | exit code、metric、artifact、状态 | 人工审计、打包阶段 |
| canonical package | `build-packages` | solver-visible task + evaluator-only target study | `materialize` |

`dataset_seeds.local_path` 是 profile 时的本地完整数据根目录。profile 之后如果文件发生变化，
重新计算的 SHA-256 会使打包失败。

## 远程 acquisition 流程

### 远程验证记录

一条远程验证记录至少标识论文、repository、record、version 和验证状态。允许状态：

| 状态 | 含义 | 后续动作 |
|---|---|---|
| `verified_downloadable` | 可列出具体文件、下载 URL 和版本 | 可进入 handoff |
| `needs_manual_review` | 页面存在但无法自动确定 payload | 人工处理 |
| `verification_deferred_network` | 临时网络/API 故障 | 重试 |
| `rejected` | 失效、无权限、纯代码或没有可下载数据 | 不进入任务合成 |

### `acquisition_queue.jsonl` / `DOWNLOAD_QUEUE.jsonl`

每行代表一个去重后的远程数据记录，主要字段：

- `dataset_id`：稳定、安全的本地目录 ID；
- `repository`、`record_id`、`version`：远程身份三元组；
- `files`：完整文件列表；
- `file_count`、`total_bytes`：声明汇总；
- `target_relative_path`：统一落盘位置；
- `source_papers`：依赖此记录的论文集合。

每个 `files[]` 项应包含可下载 URL、相对落盘路径、字节数和可选 checksum。下载器会拒绝
绝对路径和 `..` 路径穿越。

### `REMOTE_DATA_LOCATOR.json`

这是论文级数据依赖描述，不是 payload。主要字段包括论文 ID、状态、数据记录列表和目标路径。
看到此文件不能推断数据已经下载。

### 下载器数据集状态

位置：`<state_root>/datasets/<dataset_id>.json`。关键字段：

- `complete`；
- `expected_files` / `verified_files`；
- `expected_bytes` / `verified_bytes`；
- 每文件 `ok`、`status`、实际字节数；
- repository、record、version。

完成态构建同时要求：报告存在、payload 目录存在、`complete=true`、文件数一致、总字节数一致。

## 任务成熟度契约

| 层级 | 数据 | Reference | 合法状态 |
|---|---|---|---|
| locator only | 未下载 | 未运行 | `awaiting_data_download`, `enabled=false` |
| payload verified | 已下载并校验 | 未运行 | `reference_pending`, `enabled=false` |
| reference verified | 已下载并校验 | 指标/产物通过 | 可进入 canonical build |
| canonical audited | 完整 | verified | 才能由发布方显式启用 |

不要依据目录名或任务数量推断成熟度。状态字段和 audit report 才是权威来源。

## ID 与去重

- 稳定 ID 由标准化身份字段计算，不应使用数组位置或运行时间。
- 远程数据以 `(repository, record_id, version)` 去重。
- 一篇论文可依赖多个 dataset；一个 dataset 也可被多篇论文共享。
- split 必须按 dataset/paper 连通组分配，不能逐任务随机打散。

## Schema 升级

以下修改需要提升 schema version，并提供迁移说明：

- 删除或重命名必需字段；
- 修改路径根的语义；
- 修改 status 的含义或质量门；
- 修改 ID 计算方式；
- 修改 checksum 或文件完整性定义。

只新增可选字段、统计字段或额外 provenance 通常可以保持兼容，但仍应更新本文档和测试。
