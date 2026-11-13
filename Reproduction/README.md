# Reproduction

维护者：[wangdy123411](https://github.com/wangdy123411)

本目录整理本项目的数据准备、FSD 推理、论文实验复现、审计与可视化代码。模型从仓库根目录的 `fsd/` 导入。作者原版文件、README、许可证与引用信息保留在仓库根目录；本次新增内容全部位于 `Reproduction/`。

## 目录

| 路径 | 用途 |
| --- | --- |
| `run.py` | 统一命令入口 |
| `project_paths.py` | 项目、数据、权重与输入路径 |
| `prepare_zeroshot_manifest.py`、`sample_*.py`、`download_zeroshot_datasets.py` | 数据抽样、下载与清单生成 |
| `evaluate_zeroshot.py`、`zeroshot_common.py` | 原图推理、逐图记录及来源配对指标 |
| `zeroshot_validation/` | float64 原始 FSD 缓存、断点验证与表格导出 |
| `reproduce/` | 4.1 检测、4.2 来源归因、4.3 聚类 |
| `reproduce/section42_audit/`、`reproduce/external_validation/` | 来源、拒识规则与独立样本审计 |
| `reproduce/visualize43/` | 聚类嵌入、图表与三维 PCA 展示 |
| `reproduce/formal_runs/` | 已使用的正式实验调度脚本，保留实验日期、校验和与 GPU 约束 |
| `ablations/` | 600 图残差上采样实验及 2×、3× 上限消融 |
| `analysis/` | 数据核验、ImageNet 配对分析和论文图表 |
| `scripts/` | 初期 200 图实验、权重准备、中间量导出与结果验证 |
| `zeroshot_monitor/` | 本地进度采集、恢复检查和可选的统计发布客户端 |
| `configs/`、`tests/` | 数据来源、论文参考配置与已有回归测试 |
| `SOURCE_INDEX.csv` | 已整理文件与来源的对应关系 |

只收录项目实验源码、配置和已有测试。数据集、模型权重、结果缓存、虚拟环境、远程登录配置、网站部署模板及 PPT 临时构建文件均不包含在此目录中。

## 环境

使用 Python 3.12 或更高版本，在仓库根目录安装模型与扩展依赖：

```bash
python -m pip install -e .
python -m pip install -r Reproduction/requirements.txt
python Reproduction/run.py --help
```

CUDA 推理需要适配本机驱动的 PyTorch。GPU、权重与数据需要自行准备。`analysis/paper_figures_41_42/render.py` 使用 Windows 字体；`reproduce/formal_runs/` 使用 Linux 文件锁并保留历史 Blackwell 实验限制。

## 路径配置

| 环境变量 | 默认位置 |
| --- | --- |
| `FSD_PROJECT_ROOT` | `Reproduction/`，存放实验数据与输出 |
| `FSD_DATA_ROOT` | `$FSD_PROJECT_ROOT/Data`，整理后的评估数据 |
| `FSD_ARCHIVE_ROOT` | `$FSD_DATA_ROOT`，原始下载包和抽样目录 |
| `FSD_WEIGHTS_DIR` | `$FSD_PROJECT_ROOT/Weights/fsd_v1.2.0` |
| `FSD_INPUTS_ROOT` | `$FSD_PROJECT_ROOT/inputs`，复现清单与分片输入 |
| `FSD_PAPER_REFERENCE` | `Reproduction/configs/paper_reference.json` |

旧实验可将 `FSD_PROJECT_ROOT` 指向现有 `Project2`，并分别设置数据与缓存目录。模型代码始终读取当前仓库根目录的 `fsd/`。命令行提供路径参数时，以该命令的参数为准。

以下命令均在仓库根目录运行；`DATA_ROOT`、`MANIFEST.csv`、`SHARD_DIR` 等是需要替换的路径。嵌套脚本使用统一入口运行，以便加载扩展模块。

## 数据与推理

```bash
python Reproduction/run.py manifest --config Reproduction/configs/eval_paper.template.json --data-root DATA_ROOT --output MANIFEST.csv --paper-counts --seed 2026
python Reproduction/run.py evaluate --manifest MANIFEST.csv --data-root DATA_ROOT --output-dir Reproduction/Results/dry_run --dry-run
python Reproduction/run.py evaluate --manifest MANIFEST.csv --data-root DATA_ROOT --weights-dir WEIGHTS_DIR --output-dir Reproduction/Results/smoke --device cpu --threads 4 --limit-per-source 2
python Reproduction/run.py validate --manifest MANIFEST.csv --data-root DATA_ROOT --weights-dir WEIGHTS_DIR --output-dir Reproduction/Results/evaluation --device cuda:0
```

`validate` 保存 `predictions.csv`、运行元数据和 `feature_cache/raw_fsd.sqlite`。数据清单使用 `image_id,relative_path,label,source,split,sha256`，其中真实图标签为 0、生成图为 1。`sample-zip`、`sample-zip-fast` 和 `sample-ossia` 的参数通过对应命令的 `--help` 查看。

`download` 下载完整归档；其解压数量限制不限制下载量。需要少量样本时使用支持范围请求的抽样入口。

## 4.1–4.3 复现

```bash
python Reproduction/run.py reproduce 4.1 --manifest MANIFEST.csv --shard SHARD_DIR --device cpu
python Reproduction/run.py reproduce 4.2 --manifest MANIFEST.csv --shard SHARD_DIR --device cpu
python Reproduction/run.py reproduce 4.3 --manifest MANIFEST.csv --shard SHARD_DIR --device cpu
```

多分片输入可重复传入 `--shard`。每个分片包含 `run_metadata.json`、`predictions.csv` 和 `feature_cache/raw_fsd.sqlite`（或根目录的同名数据库）。输入核对图像 ID、来源、标签、图像与向量 SHA256 以及提取配置。正式模式拒绝缺行和重复 ID。首次验证可加 `--smoke`。

4.1 使用发布的检测投影、GMM 和阈值，按真实来源与生成器逐对计算 AUC。4.2 默认使用发布的 14 路来源模型；历史自拟合实验保留为独立的 `4.2-refit`。4.3 在投影前的 960 维 FSD 上运行 K-means；三维图仅用于可视化。

本项目可用数据与论文原始测试划分不同；缺失来源、替代来源与开放集范围由输出单独记录，不将部分覆盖结果当作完整论文复现。论文参考数值与本地测量分别输出。

其他脚本可按相对路径运行，例如：

```bash
python Reproduction/run.py script ablations/residual_cap2_600/compute.py --help
python Reproduction/run.py attribution --help
python Reproduction/run.py cluster-embed --help
python Reproduction/run.py pca --help
```

历史审计及消融脚本依赖已有结果目录、原始清单和校验和。代码清理会改变源码哈希，因此旧结果继续作为只读实验记录；新代码使用新的输出目录，不能直接续写旧代码创建的运行。

## 检查

```bash
python Reproduction/run.py check
python Reproduction/run.py metrics
```

测试覆盖清单与抽样、指标、特征缓存、缓存评分、表格导出和进度恢复。它们不替代完整数据上的 GPU 实验。

方法与数据来源保持在各配置的来源字段中；模型许可与引用要求见根目录 `LICENSE` 和 `CITATION.cff`。
