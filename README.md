# EmbodiedDINO: Dense Visual Features for Online 3D Instance Segmentation

本仓库整理 **EmbodiedDINO 的 AAAI-27 投稿相关研究代码**，研究从连续 posed RGB-D 观测中进行在线 3D 实例分割。仓库名为 `EmbodiedDINO`，主分支为 `main`。代码来自有实验记录的本地研究快照。

This is an EmbodiedDINO research-code snapshot built on AutoSeg3D/EmbodiedSAM. It does not imply conference acceptance or a fully packaged, one-command reproduction release.

## 同步覆盖范围

首次提交 `d4566ed` 仅包含主研究快照及少量证据，**不是整个研究工作区的全部代码**。本次补充同步覆盖以下本地来源，并对每个来源文件与发布文件进行 SHA-256 比对：

| 来源 | 校验条目数 | 发布位置 |
| --- | ---: | --- |
| 主模型、配置、缓存与训练评测脚本 | 88 | `oneformer3d/`、`configs/`、`tools/`、`scripts/` |
| 工作区顶层实验运行、监控和分析脚本 | 25 | `tools/` |
| 论文图表脚本、补充实验配置、历史代码与补丁 | 174 | `research_archive/` |
| AutoSeg3D/ESAM 数据预处理代码、说明与文本元数据 | 105 | `data/` |

共 **392 个来源条目**，具体原路径、发布路径和哈希见 [覆盖清单](evidence/source_coverage.json)，范围说明见 [同步审计](docs/SOURCE_COVERAGE.md)。条目数包含文档、元数据和保留的重复版本，不等于独立可执行程序数。

这证明已覆盖清单中列出的本地源码；尚未证明训练服务器上没有额外修改。核对时服务器 SSH 连接被重置。原始数据、权重、二进制特征缓存、机器专用 VPN/SSH 管理脚本和第三方完整依赖仓库不在此同步范围。

## 从哪里开始

| 路径 | 用途 |
| --- | --- |
| [`oneformer3d/`](oneformer3d/) | DINOv2/MSRI/OCWE 模型实现 |
| [`configs/codex_v5/`](configs/codex_v5/) | matched host、proposal injection、历史消融配置 |
| [`configs/codex_owf/`](configs/codex_owf/) | OCWE/object-feedback adapter 与 joint 实验配置 |
| [`frozen_ocwe/`](frozen_ocwe/) | 2026-07-20 冻结 OCWE 核心源码、配置、脚本；不是独立完整工程 |
| [`evidence/embodieddino/multiseed.md`](evidence/embodieddino/multiseed.md) | 三个 paired seeds 的独立进程评测记录 |
| [`research_archive/`](research_archive/) | 按原路径保存的历史实验配置、消融、图表脚本与补丁；不自动替换主模型 |
| [`data/`](data/) | 预处理源代码、README、划分与标签文本元数据；不包含原始 RGB-D 数据 |
| [`上游 README`](docs/README_UPSTREAM_AUTOSEG3D.md) | AutoSeg3D 原介绍、结果、致谢与引用 |

## 方法说明

原有在线分割器使用 3D 几何与 2D mask support。EmbodiedDINO 增加冻结 DINOv2 的稠密特征，使视觉描述符在 proposal 解码前参与 3D 表征学习。

1. **Geometry-consistent feature lifting**：通过相机内参、位姿和数据变换信息，把逐帧 2D 特征对齐到当前观测的 3D 点。训练增强后的几何对应必须正确；缓存不能替代对齐。
2. **Multi-Scale Residual Integration (MSRI)**：将提升后的描述符构成 sparse feature pyramid，残差注入 backbone decoder 的多个尺度，再进行 superpoint pooling 和 query decoding。主结果来自 proposal stream 注入，而非仅向语义分类旁路添加特征。
3. **Object-Conditioned World Encoding (OCWE)**：在 mask fragments 内池化 DINO 描述符形成 proto-object tokens，通过门控残差写回 stride-2 encoder field。冻结 host 的 adapter 实验与后续 joint-training 是不同协议，不能混报。

OCWE 冻结实验不是持久物体记忆，也不是主表全部增益的来源。保留的 dual-path、owner-residual、LGQC 等研究选项，不代表主方法默认启用。

## 代码地图

| 模块 | 入口 |
| --- | --- |
| 稀疏多尺度特征 | [`dino_sparse_fpn.py`](oneformer3d/dino_sparse_fpn.py) |
| backbone 注入 / object feedback | [`mink_unet.py`](oneformer3d/mink_unet.py) |
| 在线模型与特征流整合 | [`mixformer3d.py`](oneformer3d/mixformer3d.py) |
| 几何投影 | [`projection_utils.py`](oneformer3d/projection_utils.py) |
| 数据、缓存加载 | [`loading.py`](oneformer3d/loading.py) |
| 缓存准备与投影检查 | [`scripts/codex_tools/`](scripts/codex_tools/) |
| 训练 / 评测 | [`train.py`](tools/train.py)、[`test.py`](tools/test.py) |
| OCWE 联合训练 / 配对控制 | [`run_owf_joint_stage1.sh`](tools/run_owf_joint_stage1.sh)、[`run_owf_matched_triplet.sh`](tools/run_owf_matched_triplet.sh) |
| 表征诊断 / 分析 | [`run_representation_diagnostics.sh`](tools/run_representation_diagnostics.sh)、[`analyze_representation_dumps.py`](tools/analyze_representation_dumps.py) |
| 多种子汇总 / 迁移重评 | [`summarize_owf_seed_sweep.py`](tools/summarize_owf_seed_sweep.py)、[`run_transfer_eval_repeats.sh`](tools/run_transfer_eval_repeats.sh) |

## 已归档结果

以下来自 2026-07-13 ScanNet200 class-agnostic、312 场景的 fresh-process evaluation。数值是三个 paired seeds 的均值与样本标准差；不是本次同步重新训练得到的结果。

| Method | AP | AP50 | AP25 |
| --- | --- | --- | --- |
| Matched host | 44.48 ± 0.35 | 66.20 ± 0.42 | 80.67 ± 0.09 |
| EmbodiedDINO | 50.21 ± 0.68 | 71.51 ± 0.46 | 83.01 ± 0.17 |
| Paired gain | +5.73 ± 0.34 | +5.30 ± 0.21 | +2.34 ± 0.08 |

逐 seed AP 增益为 +5.34、+5.92、+5.93。详见[评测记录](evidence/embodieddino/multiseed.md)，其中明确区分 loop validation 和 fresh-process evaluation。三个种子的描述统计不是显著性检验；不同观测协议的跨论文结果不构成模块因果对照。

## 环境、数据与权重

参考上游[安装说明](docs/installation.md)和[数据准备](docs/dataset_preparation.md)。其历史环境为 Python 3.8、PyTorch 1.13.1、CUDA 11.6、MMEngine 0.10.3、MMDetection3D 1.4.0、MMCV 2.0.0、MMDetection 3.2.0、MinkowskiEngine 0.5.4。本次未在本地 Mac 上验证 GPU 训练环境。

运行前需要准备：

- 获得使用许可的数据集、预处理点云、annotation pickle。
- 与 RGB-D 帧对应的 DINOv2 缓存、相机位姿及内参。
- 所选实验对应的初始化或评测 checkpoint。
- subset 实验指定的固定划分文件。

Git 同步不包含数据集、缓存和权重，也没有未经核验的权重下载链接。配置保留历史 `/datadisk...` 路径，必须按本机实际位置修改数据、缓存、checkpoint 和输出路径。

## 训练与评测入口

从仓库根目录运行，确保导入本次同步版本的 `oneformer3d`：

```bash
git clone git@github.com:Xin200203/EmbodiedDINO.git
cd EmbodiedDINO
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

修改配置中的数据与缓存路径，并准备正确权重后，可使用下面的命令模板。`/path/to/...` 必须替换，初始化权重与评测权重用途不同。

```bash
python tools/train.py \
  configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only.py \
  --work-dir work_dirs/embodieddino_msri \
  --cfg-options load_from=/path/to/initialization.pth

python tools/test.py \
  configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only.py \
  /path/to/evaluation_checkpoint.pth \
  --work-dir work_dirs/embodieddino_eval
```

Matched host 使用 `configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab0_autoseg_clean_sameinit.py`。复现须同时匹配数据划分、初始化、seed、训练预算、checkpoint selection 和 evaluator。

OCWE 完整验证集入口为 `configs/codex_owf/AutoSeg3D_scannet200_stage1_v5ab1_owf_fullval.py`，继承 subset50、sub10 和 LGQC 配置。不要单独复制一个配置。`OWF_*` 环境变量控制模式、特征组成、路径等设置；冻结 DINO-only 设置请对照冻结脚本，不能从通用默认值推断。

## 来源与复现边界

更详细的缓存格式、OCWE 开关、运行顺序与故障定位见 [复现指南](docs/EMBODIEDDINO_REPRODUCIBILITY.md)。

主实现同步自本地 `object_world_encoder_20260719` 研究目录，含后续实验选项；它与冻结 OCWE 两个核心文件不完全相同。冻结副本单独保存，不能混用后声称复现了冻结结果。

冻结 `mink_unet.py` SHA-256：
`64d993c38977815b62b974678b3fdb428f91d3b81b0615e9b8765bc3b15aa1b6`

冻结 `mixformer3d.py` SHA-256：
`058e3fa4d188d4dca0ab9a2e8b583f4081e33e9247c16df14fc34bc0efa16b83`

同步文件校验值见 [`SHA256SUMS`](SHA256SUMS)。验证覆盖可执行 Python 源文件语法、Shell 语法、主研究配置继承文件、文件完整性及入口文档链接；历史归档中的 DGI 摘录不是完整 Python 模块，已在审计中单独说明。未执行 CUDA 训练或重评。完整复现仍需上述数据、权重和环境。

## 上游归属

本工作基于 AutoSeg3D、EmbodiedSAM/ESAM 等开源组件。保留 [LICENSE](LICENSE) 和[上游介绍及 citation](docs/README_UPSTREAM_AUTOSEG3D.md)。上游作者、会议接收信息和结果不代表 EmbodiedDINO 的接收状态。
