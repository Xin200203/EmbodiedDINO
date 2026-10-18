# Remote source recovery and experiment selection

## Scope and provenance

On 2026-10-18, sources were read from the training server and workstation. Collection and verification did not start training or evaluation, modify remote model files, or alter checkpoints. The 184 direct route was intermittent; collection succeeded through the existing 73/76 SSH route.

| Snapshot under `remote_sources/` | Original source directory | Files | Config load checks |
| --- | --- | ---: | ---: |
| `host76_object_world` | `/datadisk3/xxy/AutoSeg3D_v5_object_world_20260719` | 107 | 32 passed |
| `host184_object_world` | `/home/nebula/xxy/AutoSeg3D_v5_object_world_20260719` | 101 | 33 passed |
| `host76_stage1` | `/datadisk1/xxy/AutoSeg3D_v5_stage1_clean_20260604` | 98 | 29 passed |
| `host184_transfer` | `/home/nebula/xxy/AutoSeg3D_v5_transfer_20260706` | 87 | 28 passed |
| `host76_granlearn` | `/datadisk1/xxy/AutoSeg3D_v5_granlearn_20260627` | 124 | Static inheritance checks only |
| `host76_phaseD` | `/datadisk1/xxy/AutoSeg3D_v5_phaseD_joint_20260704` | 127 | Static inheritance checks only |

The 644 files include model source, configurations, experiment scripts, requirements and licenses. Each snapshot has a separate `SOURCE_MANIFEST.json` listing original root, relative file names, sizes and SHA-256 values. Source directories were selected explicitly: `oneformer3d`, `configs`, `tools`, and `scripts`. Data, checkpoints, caches, logs, third-party full clones and symlink targets were excluded. See [collection summary](../evidence/remote_source_summary.json).

## Recovered components

The exact three configurations identified in the previous audit are now preserved both in their original snapshots and at the previously missing root paths:

- `configs/codex_v5/AutoSeg3D_v5ab0_eval_scenenn.py`
- `configs/codex_v5/AutoSeg3D_v5ab1_eval_scenenn.py`
- `configs/codex_owf/AutoSeg3D_scannet200_stage1_v5cls1_owf_objectonly_joint.py`

The cls198 base and CA object-only full-validation config were also added. Exact source/destination hashes for these five additions are in [the promotion manifest](../evidence/promoted_remote_configs.json). No pre-existing root model file was replaced.

The stage-1 snapshot additionally contains paired seed 7/123 configs, no-augmentation/zero-initialization/single-scale ablations, class-sensitive host/ours/ESAM-E configs, CLIP/DINOv3 feature-cache tools, pipeline latency measurement, open-vocabulary evaluation and qualitative-output helpers. The transfer snapshot includes SceneNN and 3RScan host/ours configurations, stage-2 transfer variants and valid-rate diagnostics. Historical granularity and Phase D sources were collected directly, so their code no longer depends on the damaged historical export. That damaged file was not overwritten; old log/history recovery is not claimed.

## Select a complete workspace

The root model is byte-identical to the collected 76 object-world model. Other snapshots differ in implementation, not only configuration:

- `host76_stage1`: different `instance_merge.py`, `mink_unet.py`, `mixformer3d.py`.
- `host184_object_world`: different `mink_unet.py`, `mixformer3d.py`.
- `host184_transfer`: different `instance_merge.py`, `loading.py`, `mink_unet.py`, `mixformer3d.py`.

Run experiments within the matching snapshot and export that directory on `PYTHONPATH`. The root-level recovered SceneNN files repair the missing file references, but do not make the newer root model equivalent to the original transfer model.

Example, after preparing the original data/cache/checkpoint dependencies and adjusting paths:

```bash
cd remote_sources/host184_transfer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python tools/test.py \
  configs/codex_v5/AutoSeg3D_v5ab1_eval_scenenn.py \
  /path/to/matching/checkpoint.pth \
  --work-dir /path/to/new/evaluation_directory
```

Relative `data/` and `ckpts/` entries must resolve from this working directory. Use your existing prepared data directories; they were intentionally not bundled. Historical absolute `/datadisk...` and `/home/nebula...` references still require adjustment on a new machine. Launchers that use `TRANSFER_REPO_ROOT` or `OWF_JOINT_REPO_ROOT` should point to the corresponding complete snapshot, not an arbitrary checkout containing a similarly named configuration.

## Observed environments

Package metadata was queried from the installed ESAM environments. No credentials or package index URLs are included. Full distribution name/version reports are saved alongside config-load results:

- [76 object-world environment](../evidence/remote_environment_host76_object_world.json)
- [76 stage-1 environment](../evidence/remote_environment_host76_stage1.json)
- [184 object-world environment](../evidence/remote_environment_host184_object_world.json)
- [184 transfer environment](../evidence/remote_environment_host184_transfer.json)

| Component | Server 76 | Workstation 184 |
| --- | --- | --- |
| Python | 3.8.18 | 3.8.20 |
| torch | 1.13.1+cu117 | 1.13.1+cu116 |
| torchvision | 0.14.1+cu117 | 0.14.1+cu116 |
| mmengine | 0.10.7 | 0.10.7 |
| mmcv | 2.1.0 | 2.1.0 |
| mmdet / mmdet3d | 3.2.0 / 1.4.0 | 3.2.0 / 1.4.0 |
| MinkowskiEngine | 0.5.4 | 0.5.4 |
| torch-scatter | 2.1.1+pt113cu117 | 2.1.2 |
| networkx / easydict | 3.1 / 1.13 | 3.1 / 1.13 |
| numpy | 1.24.1 | 1.24.1 |

These are observed environments, not newly solved portable lockfiles. In particular, sparse/CUDA extension builds must match the intended PyTorch/CUDA environment. The inherited requirements file is retained for provenance and is not a complete environment specification.

## Verification and limits

All 644 collected source files were compared with hashes computed from their remote bytes during collection. Python parsing, shell syntax and relative config inheritance were checked for all six snapshots. In the four primary remote workspaces, `import oneformer3d` succeeded from the expected source root and MMEngine loaded 122 configs without errors. Every config checked there also matched the collected file hash.

This validates source preservation, module imports and config loading. It does not certify a new training run, numerical reproduction, fresh-machine installation or availability of external data/weights. Frozen historical results remain associated with their original checkpoint and experiment records.

To verify a downloaded checkout without remote access:

```bash
shasum -a 256 -c SHA256SUMS
```

The root checksum manifest excludes itself. Per-snapshot manifests offer additional provenance. The helpers `tools/collect_remote_sources.py` and `tools/probe_remote_source_environment.py` use bounded SSH operations. The collector refuses to overwrite a differing existing snapshot; the environment probe writes a new validation report.
