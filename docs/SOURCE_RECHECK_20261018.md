# EmbodiedDINO source recheck, 2026-10-18

**Historical record, resolved in the subsequent remote sync:** campus connectivity was restored and the three missing configs plus their source workspaces have been recovered. See [remote recovery](REMOTE_SOURCE_SYNC_20261018.md) for current status, hashes and runtime config-load checks. The archive corruption below remains a historical fact; its associated source code was recovered directly from the server.

## Result

The published code/evidence inventory is synchronized with its declared local sources. It is **not yet a complete runnable reproduction of every experiment**. This audit was performed against remote code revision `ca64402ff07428892c758ec220ceea86c28948c8`; the subsequent commit records these findings without changing model or experiment code.

## Confirmed missing experiment configurations

| Missing path | Referencing entrypoint | Consequence |
| --- | --- | --- |
| `configs/codex_v5/AutoSeg3D_v5ab0_eval_scenenn.py` | `tools/run_transfer_eval_repeats.sh:20` | Default SceneNN host evaluation cannot load its config. |
| `configs/codex_v5/AutoSeg3D_v5ab1_eval_scenenn.py` | `tools/run_transfer_eval_repeats.sh:23` | Default SceneNN EmbodiedDINO evaluation cannot load its config. |
| `configs/codex_owf/AutoSeg3D_scannet200_stage1_v5cls1_owf_objectonly_joint.py` | `tools/supervise_owf_cls198_objectonly_eval.sh:9` and `tools/queue_owf_cls198_objectonly_off_184.sh:7` | Default cls198 object-only launch/evaluation lacks its config. These runners accept `OWF_JOINT_CONFIG`, but no equivalent replacement was verified. |

The exact files were not found anywhere in the local `Online_3D` workspace. A targeted connector search of the related `Xin200203/3D_Reconstruction` and `Xin200203/autoseg3D` repositories returned no matches for the searched identifiers; indexed default-branch search does not establish absence from every branch or server.

A resolved SceneNN valid-rate config is preserved under `research_archive/docs/paper_writing/evidence/aaai27_sync/T3/`. It is evidence for that diagnostic run, not proof of byte-equivalence to either missing transfer configuration. The T4 configs inspected have `num_instance_classes=1` and cannot be silently substituted for the missing cls198 configuration.

## Damaged historical export

Local file:

`docs/experiment_records/20260626_sam_fastsam_granularity_audit/artifacts/76_granlearn_phaseC_20260628/final_export_20260704.tgz`

- Size: 98,304 bytes.
- SHA-256: `03f6e398395b9aca7e9128ed1e02f06cf62a0803ffb4961ac6e233ab39d28e89`.
- Normal archive enumeration fails because the compressed stream ends prematurely.
- A bounded streaming read reached 28 headers and stopped inside `./logs/phaseC_fulltrain_ft4ep_lam010_dino.train.log`.
- Four complete Python configs were readable before the truncation. All four match files already published by SHA-256: `phaseC_fulltrain_ft4ep_lam010_no_dino.py`, `phaseC_fulltrain_ft4ep_lam010_dino.py`, `phaseC_fulltrain_ft4ep_control.py`, and `phaseC_fulltrain_ft4ep_lam005_dino_weighted.py`.

This establishes coverage only for the readable members. The missing tail cannot be reconstructed or certified from this damaged archive. The local file was neither overwritten nor represented as a valid release archive.

## Positive checks

- GitHub `main` was observed at `ca64402ff07428892c758ec220ceea86c28948c8`, matching the prepared checkout.
- All 392 entries in the declared local source inventory match the fetched `origin/main` file contents.
- A broader hash-based scan of the workspace's research source extensions found no additional unmatched loose code beyond the two already excluded VPN/SSH helpers. Upstream/vendor checkouts, agent state, bytecode and release copies are excluded from this project-source scan.
- All 44 statically identified relative or repository-qualified module references under `oneformer3d/`, `tools/`, `scripts/` and `data/` resolve to local modules. This is module-file resolution, not a CUDA import or symbol/runtime test.
- README and dataset-preparation entry links resolve locally. Main configuration inheritance was verified in the prior audit.
- False positives from a simple path scan were inspected: the `tools/vpn_keepalive.sh` occurrence is the explicit exclusion table in the audit script; `tools/dataset_converters/update_infos_to_v2.py` occurs in an inherited documentation example, not an executed local import.

## External dependencies and remaining evidence limits

Core files explicitly import `torch_scatter`, `networkx` and `easydict`, but the inherited `requirements.txt` and installation guide do not explicitly enumerate those three packages. They may be present in an original environment or pulled in transitively; neither is a substitute for a verified environment specification. GPU packages, datasets and weights were not installed or executed during this source audit.

Fresh SSH probes of the host-76 jump route and host 184 both failed with connection resets. No remote source listing or hash comparison was possible. The local VPN keepalive log ends on 2026-08-13 and no current tmux session was found; old successful probes were not treated as current connectivity evidence. No experiment was launched or interrupted.

## Recovery order

1. Recover the three exact missing configurations, with their experiment/source provenance, from the training workspaces or a valid backup.
2. Compare remote model/config/script hashes with the published inventory, preserving alternate versions rather than overwriting the active snapshot.
3. Obtain an intact historical export if its missing tail is needed.
4. Record the actual GPU environment and run data/cache/config preflight before claiming end-to-end reproducibility.
