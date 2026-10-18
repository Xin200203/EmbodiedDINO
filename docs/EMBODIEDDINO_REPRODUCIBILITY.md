# EmbodiedDINO reproduction guide

## 1. Choose the experiment before choosing the command

The previously missing transfer and cls198 object-only configs have been recovered from both servers. Select the matching complete workspace using [the remote recovery guide](REMOTE_SOURCE_SYNC_20261018.md); older and newer model implementations are preserved separately.

The main MSRI configuration is `configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab1_offline_dino_injection_only.py`. It enables `dino_enable`, `dino_require`, and `backbone.use_dino`, uses 1024-dimensional features, and disables `dual_output`, `decoder.dual_path`, and `owner_residual`. This is proposal-stream injection, not the historical semantic-only branch.

The matched host configuration is `configs/codex_v5/AutoSeg3D_scannet200_stage1_v5ab0_autoseg_clean_sameinit.py`. Confirm initialization and the actual evaluator against your chosen experiment record. A generic config alone does not identify a paper-table checkpoint.

`configs/codex_owf/` adds object feedback. The frozen adapter study and the joint configuration are different experiments. The full-validation config inherits subset50, which inherits sub10, which in turn inherits the LGQC adapter configuration. Keep the complete config tree.

## 2. Prepare the sparse-convolution environment

Use the upstream installation guide for the CUDA/MMDetection3D/MinkowskiEngine stack. Do not interpret a successful Python syntax check as an import or CUDA compatibility test. Import `oneformer3d` from this repository and check the resolved `__file__` before launching a run if other checkouts are on `PYTHONPATH`.

The inherited package list is not an exhaustive environment lock. Actual versions of `torch_scatter`, `networkx`, `easydict` and the CUDA/MMEngine stack have now been read from both machines and recorded in [the observed environment table](REMOTE_SOURCE_SYNC_20261018.md). Use that evidence when preparing the matching environment.

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -c 'import oneformer3d; print(oneformer3d.__file__)'
```

## 3. Prepare data and the dense feature stream

The main config expects `data/scannet200-mv_fast/` and the associated ScanNet200 annotation files. Dataset downloads and redistribution are governed by the respective dataset licenses. Preprocessing source and text metadata are now included under `data/`: AutoSeg3D files take precedence and ESAM supplies missing helpers. Raw datasets, binary statistics/caches and dependencies such as segmentator/pointops are not bundled. The inherited preparation instructions distinguish these external inputs.

The cached feature pipeline is:

1. `scripts/codex_tools/build_mv_dino_manifest.py`: enumerate RGB frames.
2. `scripts/codex_tools/extract_mv_dino_cache.py`: extract frozen DINOv2 descriptors.
3. `scripts/codex_tools/compact_dino_cache.py`: optional cache compaction.
4. `scripts/codex_tools/smoke_mv_dino_projection.py`: inspect projection coverage.

Read each script's `--help` in the prepared environment for its required input arguments. The extraction script uses RGB conversion, deterministic 420x560 resizing, ImageNet normalization, and a patch size of 14. ViT-L features have a 30x40 patch grid and 1024 channels. Feature caching is per RGB frame; lifting into augmented 3D coordinates remains part of model execution.

The MSRI config defines `DINO_CACHE_ROOT` as a Python constant. Change it before config evaluation: changing a similarly named shell variable does not automatically override that constant. Inspect resolved pipelines, because the constant is inserted into nested `LoadDINOFeatureFromCache` transforms. Change historical `load_from` paths as well.

Projection guards in the main config use a 0.90 minimum hit ratio. If these checks fail, inspect frame correspondence, poses, intrinsics, resizing and augmentation transforms. Do not silence the guard to make a run proceed.

## 4. OCWE frozen adapter settings

The frozen triplet script explicitly sets:

| Variable | Frozen triplet setting | Meaning |
| --- | --- | --- |
| `OWF_PLACEMENT` | `pre_memory` | write before sparse memory processing |
| `OWF_INCLUDE_DINO` | `1` | include DINO token descriptors |
| `OWF_INCLUDE_S2` | `0` by default in triplet | exclude stride-2 feature input from token composition |
| `OWF_INCLUDE_XYZ` | `0` by default in triplet | exclude XYZ input from token composition |
| `OWF_ORACLE_PURITY` | `0` | no oracle purity filter |
| `OWF_PERSISTENT_ORACLE` | `0` | no persistent oracle option |
| `OWF_OUTPUT_SCALE` | `1.0` default | residual scale |
| `OWF_MODE` | `stats-only`, `real`, `shuffled` | no-write, aligned and shuffled controls |

The generic sub10 config defaults to including S2 and XYZ, so it is not equivalent to the frozen DINO-only setting without these overrides. The adapter config sets `object_feedback_freeze_base=True`; the joint-training configuration must be interpreted separately.

The archived runner requires `OWF_REPO_ROOT`, `OWF_PYTHON`, `OWF_CHECKPOINT`, `OWF_GRID_ROOT`, and `OWF_DINO_CACHE`. Set them to actual paths. Each run should use a new output directory: the archived script initializes `summary.tsv` and can overwrite a previous summary if an old grid root is reused.

`frozen_ocwe/` contains only selected frozen files. It is not a standalone checkout. Reconstruct the matching environment and verify dependencies before overlaying those files in a separate working copy. Preserve the main snapshot and never claim that a later mixed source tree has the frozen hashes.

## 5. Record enough evidence to compare runs

For every reported result, retain the exact source revision, resolved configuration, data split, feature-cache identity, initialization/checkpoint hash, seed, selected epoch, evaluator protocol and final metrics. The archived main result uses 312 evaluation scenes and three paired seeds. Fresh-process evaluation and in-training loop validation have distinct results; the README reports the fresh-process numbers.

The snapshot does not include training checkpoints or feature caches. No checkpoint download availability is claimed. The three-seed summary is descriptive evidence, not a newly rerun benchmark or significance test.

## 6. Common failure modes

| Symptom | Check first |
| --- | --- |
| `oneformer3d` import fails | CUDA sparse-convolution/MMDetection3D dependencies and `PYTHONPATH` |
| config inheritance fails | complete `configs/` tree and installed `mmdet3d` config package |
| annotation file not found | working directory, dataset root and fixed subset pickle |
| DINO cache miss | exact RGB frame names, manifest and cache root |
| low projection hit ratio | camera/point coordinate convention and augmentation synchronization |
| checkpoint shape mismatch | host architecture, DINO width, enabled modules and checkpoint provenance |
| results differ from README | seed, checkpoint selection, independent evaluator and data/cache identity |

## 7. What was verified for this sync

The release preparation checks Python parsing without importing GPU packages, Shell syntax, main local config inheritance paths, frozen-source SHA-256 values and entry-document links. An explicitly labelled historical DGI excerpt in `research_archive/` is not a standalone parseable Python module. Historical configs can reference their original remote working trees; they are evidence, not automatically runnable replacements. See `docs/SOURCE_COVERAGE.md` for the exact audit boundary. These checks establish archive consistency, not end-to-end training correctness.
