# 31 — 三随机种子完赛与独立重评

Date: 2026-07-13

## 1. 启动决策

当前最高价值、最小规模的实验是对 seed 7/123 的四个 validation-selected checkpoints 做 fresh-process 全 312 场景重评。该实验直接决定主表能否从单 seed 数字升级为三 seed mean/std，不改变模型、训练或评测代码。

暂不启动新的长周期训练：class-sensitive 多 seed、额外 feature encoder 和 stage-2 修复都无法在同等资源下提供更高的投稿收益。

## 2. 已完成的 loop-validation 结果

| Seed | Host checkpoint | Host AP/AP50/AP25 | Ours checkpoint | Ours AP/AP50/AP25 | AP gain |
|---:|---|---|---|---|---:|
| 2050318485 | E96 | 44.09/65.73/80.71 | E96 | 49.68/71.34/83.11 | +5.59 |
| 7 | E128 | 44.59/66.51/80.57 | E128 | 50.57/71.65/82.81 | +5.98 |
| 123 | E128 | 44.76/66.37/80.74 | E112 | 50.66/71.92/83.07 | +5.90 |

使用 sample standard deviation：

| Method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Matched host | 44.48 +/- 0.35 | 66.20 +/- 0.42 | 80.67 +/- 0.09 |
| EmbodiedDINO | 50.30 +/- 0.54 | 71.64 +/- 0.29 | 83.00 +/- 0.16 |
| Paired gain | +5.82 +/- 0.21 | +5.43 +/- 0.26 | +2.32 +/- 0.08 |

这些数字只用于记录 loop validation。论文主表仍等待下述 fresh-process 结果，不混用两种口径。

## 3. Fresh-process evaluation

启动脚本：`scripts/run_multiseed_independent_eval_20260713.sh`

| Run | GPU | Output directory | Status |
|---|---:|---|---|
| Host seed 7 E128 | 1 | `eval_v5ab0_seed7_e128_independent_20260713` | 44.59/66.51/80.57 |
| Ours seed 7 E128 | 2 | `eval_v5ab1_seed7_e128_independent_20260713` | 50.51/71.62/82.82 |
| Host seed 123 E128 | 3 | `eval_v5ab0_seed123_e128_independent_20260713` | 44.76/66.37/80.74 |
| Ours seed 123 E112 | 5 | `eval_v5ab1_seed123_e112_independent_20260713` | 50.69/71.90/83.15 |

Stop condition: 四条日志均出现 `Epoch(test) [312/312]` 的 `all_ap/all_ap_50%/all_ap_25%`，随后与原 seed 2050318485 的 E96 independent results 合并计算三 seed mean/std。

启动记录：第一次 launcher 从远端 home 目录执行，四条进程均在加载 checkpoint 前因相对数据路径找不到而退出，没有产生或消费评测结果。launcher 随后加入 `cd "$REPO"` 并原位重启；该工程失败不纳入实验统计。

四条重评均于 2026-07-13 完成，未出现评测错误。合并原 seed 的 E96 fresh-process 结果后：

| Method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Matched host | 44.48 +/- 0.35 | 66.20 +/- 0.42 | 80.67 +/- 0.09 |
| EmbodiedDINO | 50.21 +/- 0.68 | 71.51 +/- 0.46 | 83.01 +/- 0.17 |
| Paired gain | +5.73 +/- 0.34 | +5.30 +/- 0.21 | +2.34 +/- 0.08 |

逐 seed AP gain 为 +5.34、+5.92、+5.93。stop condition 已满足。

## 4. 论文同步规则

- 四条重评已完成：主表 controlled block、Abstract、Introduction、Conclusion 和 Limitations 统一切换为三 seed fresh-process mean/std。
- 不做显著性声明；仅报告三个 paired seeds 的逐 seed 增益与 descriptive mean/std。
