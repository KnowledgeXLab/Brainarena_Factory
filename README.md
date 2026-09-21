# Neuro Dataset Factory

这是经过清理的神经科学数据与训练任务生成管线。目录只包含可复用代码、运维命令、
配置示例、数据契约文档和离线测试；不包含历史 `work/`、LLM 缓存、下载 payload、凭据或
机器绑定的实验 case。

## 先确认要接哪条管线

本项目包含两条状态不同的流程：

| 流程 | 入口 | 输出状态 | 可否直接启用训练任务 |
|---|---|---|---|
| 本地 canonical | 本地完整数据集 + 已审核论文关联 | reference verified canonical package | 通过全部 audit 后可以 |
| 远程 acquisition | 论文语料中的数据链接 | `awaiting_data_download` provisional package | 不可以；先下载 payload、执行 reference |

不要把 `REMOTE_DATA_LOCATOR.json` 当成原始数据。远程流程生成的任务默认
`enabled=false`，即使格式审计通过，也只表示数据定位和任务结构有效。

## 安装

要求 Python 3.10+。核心离线能力使用标准库；S3 语料读取使用可选的 `boto3`；远程
payload 下载还要求系统中存在 `curl`。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[s3,test]'

neuro-dataset-factory --help
neuro-dataset-download --help
```

如果不需要 S3 和 pytest，也可以只运行：

```bash
python -m pip install -e .
```

## 最小可运行示例

仓库自带一个只用于验证安装、输入解析和 SHA-256 manifest 的微型数据目录：

```text
examples/minimal_dataset/
├── LICENSE
└── recordings.csv
examples/minimal_dataset_seeds.jsonl
```

从项目根目录执行：

```bash
python -m neuro_dataset_factory profile \
  --seeds examples/minimal_dataset_seeds.jsonl \
  --out-root /tmp/neuro_dataset_factory_demo/registry
```

命令成功时会输出 `"ok": true`，并生成：

```text
/tmp/neuro_dataset_factory_demo/registry/
├── dataset_registry.jsonl
├── profile_report.json
└── datasets/minimal_neuro_demo_v1/
    ├── manifest.jsonl
    └── profile_report.json
```

其中，输入 `minimal_dataset_seeds.jsonl` 的核心格式是：

```json
{
  "dataset_id": "minimal_neuro_demo_v1",
  "name": "Minimal Neuroscience Dataset",
  "version": "v1",
  "local_path": "examples/minimal_dataset",
  "source_url": "https://example.org/minimal-neuro-dataset",
  "license": "CC0-1.0",
  "modalities": ["electrophysiology"],
  "species": ["mouse"]
}
```

接入真实数据时，复制这条 seed，替换 `dataset_id`、版本、来源、许可证和 `local_path`；
`local_path` 应指向完整数据集版本。这个微型示例不是正式科学数据，不能用于任务生成或训练。
完成 profile 后，继续按照[本地 canonical 管线](docs/LOCAL_PIPELINE.md)验证论文关联、生成候选、
运行 reference 并打包。远程论文语料从[远程 acquisition 管线](docs/REMOTE_PIPELINE.md)的
`scan-paper-corpus` 阶段开始。

## 离线验收

标准库测试入口会覆盖全部离线测试，包括下载器参数和断点状态逻辑：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

安装了 pytest 时也可以运行：

```bash
pytest -q
```

这些测试不访问网络、不调用模型，也不需要真实数据 payload。

## 文档导航

- [团队接入说明](docs/INTEGRATION.md)：安装边界、稳定接口、状态与安全约束。
- [远程数据管线](docs/REMOTE_PIPELINE.md)：从论文语料到下载、完成态交付的命令链。
- [本地 canonical 管线](docs/LOCAL_PIPELINE.md)：从数据 profile 到 reference 和导出。
- [数据契约](docs/DATA_CONTRACTS.md)：主要 JSON/JSONL 的生产者、消费者和状态语义。
- [交付清单](HANDOFF_MANIFEST.md)：本目录包含和明确不包含的内容。

## 代码布局

```text
neuro_dataset_factory/       # 可安装核心包
  cli.py                     # 主 CLI
  contracts.py               # schema version、验证状态和稳定 ID
  remote_data.py             # 远程记录解析、验证与 acquisition handoff
  remote_tasks.py            # provisional 任务生成与审计
  reference_runner.py        # 经人工授权的 reference 执行器
  ops/                       # 下载、完成态交付、delta/merge 等运维命令
configs/                     # 可公开的发现配置
examples/                    # 不含真实路径和凭据的契约示例
docs/                        # 团队接入与运行文档
tests/                       # 完全离线的测试
```

## 凭据规则

复制 `.env.example` 为 `.env` 后填写所需值。`.env` 已被 git 忽略。API key、AWS 凭据、
代理认证信息不得写入命令历史、配置 JSONL、LLM cache、日志或 provenance。

核心 LLM 命令只有在显式传入 `--env-file .env` 时才读取该文件。S3 凭据使用标准 AWS
环境变量。S3 endpoint、bucket URI、输出目录和模型名均通过命令参数传入，不需要修改源码。

## 质量门摘要

- 数据集版本、许可证、文件大小和 SHA-256 必须保留。
- LLM 生成结果只能是 `reference.status=unverified`。
- `reference.command` 必须人工审核，并显式传入 `--allow-execution` 才会执行。
- reference 进程、指标、容差和声明产物全部通过后才能升级为 `verified`。
- 公开 query 不得包含隐藏 GT 结论、目标数值或 evaluator 路径。
- train/validation/test 按 dataset/paper 关联组切分。
- 未下载远程 payload、只有 locator 的任务永远不能被标记为 canonical。
