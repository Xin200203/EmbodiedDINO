# Source coverage audit

## Correction to the first sync

Commit `d4566ed` published 105 files. It covered the copied main model snapshot, but omitted workspace-level experiment runners, figure/analysis code, historical ablation/configuration files and upstream preprocessing programs. A clean Git working tree established only that the prepared release had been pushed; it did not establish that all project code had been collected.

## Reproducible local inventory

`tools/audit_source_coverage.py` enumerates these source roots beneath the local `Online_3D` workspace:

- `.codex_worktrees/object_world_encoder_20260719/{oneformer3d,configs,tools,scripts}`: primary snapshot, copied into corresponding release directories.
- That snapshot's `patches/`: preserved under `research_archive/snapshot_patches/`.
- Workspace `tools/`: experiment launchers, monitors, result analysis and figure helpers.
- Workspace `docs/` and `docs_audit/`: code and configuration files, preserved under `research_archive/` with original paths.
- `external_audit/current_v5_184_snapshot/`: older model snapshot preserved separately, never overlaid on the primary model.
- Git-tracked preprocessing source/docs/text metadata in `AutoSeg3D/data` and `ESAM/data`: AutoSeg3D takes precedence at overlapping paths; ESAM fills missing files. Alternative versions of overlapping upstream files remain in their upstream repositories.

Source extensions are explicitly enumerated in the script. Every selected source receives a destination, SHA-256 and comparison result. The initial expanded inventory contains 392 source entries: 88 main snapshot, 25 workspace tools, 174 historical files and 105 preprocessing/support files. Preserved duplicates remain separate evidence entries.

From the release checkout:

```bash
python tools/audit_source_coverage.py --workspace /path/to/Online_3D
git fetch origin
python tools/audit_source_coverage.py --workspace /path/to/Online_3D --ref origin/main
```

The second command compares each original source against bytes stored in the fetched Git revision. It does not merely compare local filenames. `--sync` copies missing files but refuses to treat differing existing files as a successful match; it never overwrites them. `--report` writes a machine-readable inventory.

## Intentional exclusions and unresolved scope

- `tools/vpn_keepalive.sh` and `docs_audit/ssh_184_robust.sh` are local network-management helpers, excluded explicitly.
- Raw RGB-D datasets, checkpoints, feature caches, generated PDFs/images/logs, editor settings, bytecode and binary metadata are not source-code payloads and are not bundled.
- Complete third-party checkouts and their Git histories are not copied. Upstream dependencies remain required.
- The paper's LaTeX text and presentation assets are not part of this code inventory; figure-generating programs are archived.
- The training-host SSH probe was reset by the jump host, so remote-only changes could not be inventoried. Local coverage does not establish that this is the latest remote source for every reported experiment.

## Validation and archive semantics

Primary source files and supplemental executable scripts are syntax checked without importing CUDA packages. All main configuration inheritance paths are checked locally, excluding package-qualified `mmdet3d::` bases. Frozen OCWE hashes are checked independently.

`research_archive/docs/experiment_records/20260621_184_dgi_overseg_init/mixformer3d_dgi_excerpt.py` begins with an explicit excerpt comment and contains a fragment from inside a class. Its indentation makes it invalid as a standalone Python module; this is preserved byte-for-byte as historical evidence, not presented as executable code.

Historical archived configs, launchers and figure scripts can require their original remote directory layout, later source versions, result dumps or image assets. Their presence establishes preservation, not universal executability. No training jobs or evaluation jobs are launched by this audit.
