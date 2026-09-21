# 本地 canonical 管线

这条流程适用于完整数据已经落盘、能够实际执行 reference 的数据集。以下路径均为示例，
调用方应替换为自己的工作目录。

## 1. Profile 数据集

准备 `dataset_seeds.jsonl`，格式参考 `examples/dataset_seeds.jsonl`。`local_path` 必须指向
完整数据集版本，不应指向裁剪样本。

```bash
neuro-dataset-factory profile \
  --seeds inputs/dataset_seeds.jsonl \
  --out-root work/registry
```

主要输出：

- `work/registry/dataset_registry.jsonl`
- `work/registry/datasets/<dataset_id>/manifest.jsonl`
- 文件大小、角色、逐文件 SHA-256 和整树 hash

## 2. 发现并验证论文关联

可选的公开发现：

```bash
neuro-dataset-factory search-papers \
  --registry work/registry/dataset_registry.jsonl \
  --out work/paper_candidates.jsonl
```

发现结果始终是 unverified。人工补充论文证据并生成 `paper_links.jsonl` 后运行：

```bash
neuro-dataset-factory validate-links \
  --registry work/registry/dataset_registry.jsonl \
  --links inputs/paper_links.jsonl \
  --out work/verified_paper_links.jsonl
```

进入后续阶段至少需要 `dataset_version_match=true`、`data_support=yes` 和足够的结果位置证据。

## 3. 生成与去重任务候选

```bash
neuro-dataset-factory generate-candidates \
  --registry work/registry/dataset_registry.jsonl \
  --links work/verified_paper_links.jsonl \
  --out work/task_candidates.raw.jsonl \
  --cache-dir work/llm_cache \
  --env-file .env \
  --model '<approved-model>' \
  --tasks-per-link 4 \
  --jobs 4

neuro-dataset-factory deduplicate \
  --candidates work/task_candidates.raw.jsonl \
  --out work/task_candidates.accepted.jsonl \
  --duplicates work/task_candidates.duplicates.jsonl
```

LLM 输出必须保持 `reference.status=unverified`。模型给出的数字或结论不能直接成为 GT。

## 4. 审核并运行 reference

先人工审查每条 `reference.command`、对应代码、输入路径和期望产物。确认后显式授权：

```bash
neuro-dataset-factory run-references \
  --registry work/registry/dataset_registry.jsonl \
  --candidates work/task_candidates.accepted.jsonl \
  --out-candidates work/task_candidates.reference_verified.jsonl \
  --results work/reference_results.jsonl \
  --out-root work/reference_runs \
  --workspace-root . \
  --allow-execution
```

不要在无人审核的生产批处理中使用 `--accept-observed-targets`。

## 5. 构建、导出和 audit

```bash
neuro-dataset-factory build-packages \
  --registry work/registry/dataset_registry.jsonl \
  --links work/verified_paper_links.jsonl \
  --candidates work/task_candidates.reference_verified.jsonl \
  --out-root work/packages

neuro-dataset-factory materialize \
  --packages-root work/packages \
  --out-root work/brainarena

neuro-dataset-factory audit-brainarena work/brainarena

neuro-dataset-factory assign-splits \
  --registry work/brainarena/benchmark/task_registry.json \
  --out work/brainarena/benchmark/task_registry.json \
  --manifest work/brainarena/benchmark/split_manifest.json
```

正式发布不得使用 `build-packages --allow-unverified`。最终交付前保存所有 report、输入 manifest、
本包版本、reference 环境版本和 split manifest。
