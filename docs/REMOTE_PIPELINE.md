# 远程 acquisition 管线

这条流程从论文语料中的数据链接构建可下载 locator 和 provisional 任务。它不会把 locator
伪装成 payload，也不会在 reference 尚未运行时启用任务。

示例假设：

- 论文索引为 CSV，其中包含 `data_link` 及神经科学筛选字段；
- 论文正文为解析后的 JSONL，可通过 S3-compatible endpoint 流式读取；
- 所有目录、URI、endpoint 和模型由调用方显式传入。

## A. 定位并验证远程数据

### 1. 扫描论文索引

本地 CSV：

```bash
neuro-dataset-factory scan-paper-corpus \
  --csv inputs/papers.csv \
  --out work/paper_candidates.jsonl
```

S3-compatible CSV：

```bash
neuro-dataset-factory scan-paper-corpus \
  --s3-uri 's3://<bucket>/<prefix>/data.csv' \
  --endpoint-url 'https://<s3-endpoint>' \
  --out work/paper_candidates.jsonl
```

### 2. 选择可自动验证的 repository

```bash
neuro-dataset-factory select-remote-candidates \
  --candidates work/paper_candidates.jsonl \
  --repositories figshare,zenodo,osf,dryad,dandi,openneuro \
  --include-direct-files \
  --out work/verification_candidates.jsonl
```

repository 列表应按团队当前政策配置。代码还支持其他 repository verifier；不要因为 URL
能访问就绕过文件级验证。

### 3. 文件级验证

```bash
neuro-dataset-factory verify-remote-data \
  --candidates work/verification_candidates.jsonl \
  --out work/verified_remote.jsonl \
  --jobs 8 \
  --timeout 30 \
  --max-retries 2
```

只有 `verified_downloadable` 可以进入 acquisition handoff。`verification_deferred_network`
进入重试批次；`needs_manual_review` 进入人工队列；`rejected` 不进入任务合成。

### 4. 构建 acquisition queue

```bash
neuro-dataset-factory build-acquisition-handoff \
  --verified work/verified_remote.jsonl \
  --out-root work/acquisition_handoff
```

主要输出：

- `acquisition_queue.jsonl`：按远程数据记录去重后的下载队列；
- `datasets/<dataset_id>/data_locator.json`：版本、文件 URL、大小和 checksum；
- 稳定的 `target_relative_path=data/<dataset_id>`。

## B. 构建 provisional 任务

### 5. 提取论文上下文和图片对象

```bash
neuro-dataset-factory extract-paper-contexts \
  --verified work/verified_remote.jsonl \
  --s3-uri 's3://<bucket>/<prefix>/data.jsonl' \
  --endpoint-url 'https://<s3-endpoint>' \
  --out work/paper_contexts.jsonl

neuro-dataset-factory extract-paper-figures \
  --verified work/verified_remote.jsonl \
  --s3-uri 's3://<bucket>/<prefix>/data.jsonl' \
  --endpoint-url 'https://<s3-endpoint>' \
  --out work/paper_figures.jsonl
```

这两个阶段只保留已验证论文，并按需流式读取，不应预先复制完整论文语料库。

### 6. 生成任务候选

```bash
neuro-dataset-factory generate-remote-candidates \
  --handoff-root work/acquisition_handoff \
  --paper-contexts work/paper_contexts.jsonl \
  --out work/task_candidates.jsonl \
  --cache-dir work/llm_cache \
  --env-file .env \
  --model '<approved-model>' \
  --tasks-per-paper 4 \
  --jobs 4
```

生成器会同时写 accepted、retry/失败报告等 checkpoint 文件。只把 accepted 文件交给打包阶段。

### 7. 映射论文图

```bash
neuro-dataset-factory map-task-figures \
  --candidates work/task_candidates.accepted.jsonl \
  --figures work/paper_figures.jsonl \
  --out work/figure_mappings.jsonl \
  --cache-dir work/figure_llm_cache \
  --env-file .env \
  --model '<approved-model>'

neuro-dataset-complete-figures \
  --candidates work/task_candidates.accepted.jsonl \
  --figures work/paper_figures.jsonl \
  --mappings work/figure_mappings.jsonl
```

补全命令只把没有模型结果的任务保守标记为无精确匹配，不会制造 GT 图片。

### 8. 构建并导出 provisional package

```bash
neuro-dataset-factory build-provisional-packages \
  --candidates work/task_candidates.accepted.jsonl \
  --handoff-root work/acquisition_handoff \
  --out-root work/provisional_packages

neuro-dataset-factory audit-provisional-packages work/provisional_packages

neuro-dataset-factory materialize-remote-brainarena \
  --packages-root work/provisional_packages \
  --out-root deliveries/remote_provisional

neuro-dataset-factory materialize-gt-figures \
  --mappings work/figure_mappings.jsonl \
  --brainarena-root deliveries/remote_provisional \
  --endpoint-url 'https://<s3-endpoint>'

neuro-dataset-factory audit-remote-brainarena deliveries/remote_provisional
neuro-dataset-factory audit-gt-figures deliveries/remote_provisional
```

此时任务仍为 `awaiting_data_download`，不能启用。

## C. 下载并构建完成态 payload 交付

远程 BrainArena 导出根目录包含 `DOWNLOAD_QUEUE.jsonl`。先做 dry run：

```bash
neuro-dataset-download \
  --queue deliveries/remote_provisional/DOWNLOAD_QUEUE.jsonl \
  --output-root payloads \
  --state-root state/download \
  --dry-run
```

实际下载：

```bash
neuro-dataset-download \
  --queue deliveries/remote_provisional/DOWNLOAD_QUEUE.jsonl \
  --output-root payloads \
  --state-root state/download \
  --jobs 4 \
  --per-host-jobs 2 \
  --skip-completed-reports
```

下载器会校验声明大小；有 checksum 时同时校验 checksum。失败时保留 `.part` 和数据集状态，
重新运行可续传。不要仅依据 stdout 日志声明完成，应检查
`state/download/datasets/<dataset_id>.json`。

只保留依赖数据全部完成的论文，构建新的完成态目录：

```bash
neuro-dataset-finalize-payloads \
  --source-root deliveries/remote_provisional \
  --output-root deliveries/payload_verified \
  --state-root state/download \
  --payload-root payloads
```

完成态构建会验证文件数、总字节数和下载报告，并创建指向共享 payload store 的符号链接。
输出状态仍然是 `reference_pending`、`enabled=false`。下一步必须为真实 payload 编写/审核并
执行 reference，之后走 canonical audit；本命令不会自动提升为可训练状态。

## D. 批次和 release 运维

以下命令属于 release 管理，不是主任务生成路径：

```bash
neuro-dataset-jsonl --help
neuro-dataset-build-delta --help
neuro-dataset-merge-releases --help
neuro-dataset-redact-titles --help
neuro-dataset-resolve-dois --help
```

delta 和 merge 的输出目录应为全新目录；在保留旧 delivery 的情况下生成新 release，方便审计和回滚。
