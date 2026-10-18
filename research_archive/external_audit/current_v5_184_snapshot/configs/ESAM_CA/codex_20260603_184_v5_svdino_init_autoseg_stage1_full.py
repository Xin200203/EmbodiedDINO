_base_ = "./ESAM_online_scannet200_CA_dual_path_dino.py"

# Full training: v5 online dual-path architecture, initialized from single-frame DINO/v5 checkpoint,
# evaluated/trained with AutoSeg3D-stage1-style protocol cadence/thresholds.
model = dict(
    test_cfg=dict(
        topk_insts=100,
        inst_score_thr=0.3,
    )
)

custom_hooks = [
    dict(type="EmptyCacheHook", after_iter=True),
    # User gate: warn only if projection valid rate falls below 90%.
    dict(type="DINOAlignmentSanityHook", log_interval_epochs=1, valid_rate_error_threshold=0.90),
]

load_from = "work_dirs/ESAM_sv_scannet200_CA_dino/best_all_ap_50%_epoch_115.pth"
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=128, val_interval=16)
param_scheduler = dict(type="PolyLR", begin=0, end=128, power=0.9)
default_hooks = dict(
    checkpoint=dict(
        interval=16,
        max_keep_ckpts=3,
        save_best=["all_ap_50%"],
        rule="greater"))
randomness = dict(seed=2050318485, deterministic=False)
