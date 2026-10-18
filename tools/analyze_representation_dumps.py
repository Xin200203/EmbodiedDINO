#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import warnings
from collections import defaultdict

import numpy as np
import torch


VARIANT_LABELS = {
    "host": "Matched host",
    "embodieddino": "EmbodiedDINO",
    "ocwe_no_write": "No feedback",
    "ocwe_aligned": "Aligned feedback",
    "ocwe_shuffled": "Shuffled feedback",
}

VARIANT_COLORS = {
    "host": "#6B7280",
    "embodieddino": "#2563A6",
    "ocwe_no_write": "#A3A3A3",
    "ocwe_aligned": "#2E7D5B",
    "ocwe_shuffled": "#D97706",
}


def variant_label(name):
    return VARIANT_LABELS.get(name, name.replace("_", " "))


def variant_color(name):
    return VARIANT_COLORS.get(name, "#4B5563")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze post-filter proposals before temporal merging and online "
            "3D representations."))
    parser.add_argument(
        "--dump", action="append", required=True, metavar="NAME=DIR",
        help="Variant name and directory containing representation .pt files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--purity-threshold", type=float, default=0.8)
    parser.add_argument("--min-instance-points", type=int, default=20)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--require-min-frames-per-variant", type=int, default=0)
    parser.add_argument("--require-min-scenes-per-variant", type=int, default=0)
    parser.add_argument("--require-gt-every-frame", action="store_true")
    parser.add_argument("--require-retrieval-queries", action="store_true")
    parser.add_argument(
        "--preferred-visualization-scene",
        help=("Prioritize this scene for qualitative plots when it satisfies "
              "each plot's validity requirements; otherwise fall back to the "
              "deterministic lexicographic rule."))
    return parser.parse_args()


def parse_dumps(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=DIR, got {value}")
        name, path = value.split("=", 1)
        if not name or not os.path.isdir(path):
            raise ValueError(f"invalid dump specification: {value}")
        result[name] = path
    return result


def load_frames(path):
    frames = []
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".pt"):
            continue
        full_path = os.path.join(path, filename)
        try:
            payload = torch.load(full_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(full_path, map_location="cpu")
        payload["_path"] = full_path
        frames.append(payload)
    if not frames:
        raise ValueError(f"no .pt representation dumps found in {path}")
    return frames


def dominant_labels(group_ids, point_labels, num_groups, purity_threshold):
    labels = torch.full((num_groups,), -1, dtype=torch.long)
    purities = torch.zeros(num_groups, dtype=torch.float32)
    group_ids = torch.as_tensor(group_ids, dtype=torch.long).reshape(-1)
    point_labels = torch.as_tensor(point_labels, dtype=torch.long).reshape(-1)
    valid = (
        (group_ids >= 0) & (group_ids < num_groups) & (point_labels >= 0))
    if not valid.any():
        return labels, purities
    valid_groups = group_ids[valid]
    unique_labels, dense_labels = torch.unique(
        point_labels[valid], sorted=True, return_inverse=True)
    num_labels = unique_labels.numel()
    pair_ids = valid_groups * num_labels + dense_labels
    counts = torch.bincount(
        pair_ids, minlength=num_groups * num_labels).reshape(
            num_groups, num_labels)
    dominant_counts, dominant_indices = counts.max(dim=1)
    totals = counts.sum(dim=1)
    nonempty = totals > 0
    purities[nonempty] = (
        dominant_counts[nonempty].float() / totals[nonempty].float())
    accepted = nonempty & (purities >= purity_threshold)
    labels[accepted] = unique_labels[dominant_indices[accepted]]
    return labels, purities


def proposal_statistics(
        frames, topk_values, thresholds, min_points, warn_missing=True):
    best_by_k = {k: [] for k in topk_values}
    missing_gt_frames = 0
    for frame in frames:
        masks = torch.as_tensor(frame["proposal_masks"], dtype=torch.bool)
        scores = torch.as_tensor(frame["proposal_scores"], dtype=torch.float32)
        if frame.get("gt_instance_ids") is None:
            missing_gt_frames += 1
            continue
        gt_ids = torch.as_tensor(frame["gt_instance_ids"], dtype=torch.long)
        instances = []
        for instance_id in torch.unique(gt_ids[gt_ids >= 0]).tolist():
            mask = gt_ids == instance_id
            if int(mask.sum()) >= min_points:
                instances.append(mask)
        if not instances:
            continue
        gt_masks = torch.stack(instances, dim=0)
        order = torch.argsort(scores, descending=True)
        masks = masks[order]
        for k in topk_values:
            selected = masks[:k]
            if selected.numel() == 0:
                best_by_k[k].extend([0.0] * len(instances))
                continue
            intersection = selected.float() @ gt_masks.float().T
            union = (selected.sum(1, keepdim=True) +
                     gt_masks.sum(1).unsqueeze(0) - intersection)
            iou = intersection / union.clamp(min=1.0)
            best_by_k[k].extend(iou.max(dim=0).values.tolist())

    if missing_gt_frames and warn_missing:
        warnings.warn(
            f"proposal metrics skipped {missing_gt_frames} frame(s) without "
            "gt_instance_ids")
    rows = []
    for k in topk_values:
        values = np.asarray(best_by_k[k], dtype=np.float64)
        row = {
            "topk": k,
            "visible_instance_frames": int(values.size),
            "missing_gt_frames": missing_gt_frames,
            "mean_best_iou": float(values.mean()) if values.size else math.nan}
        for threshold in thresholds:
            key = f"recall_iou_{threshold:.2f}"
            row[key] = float((values >= threshold).mean()) if values.size else math.nan
        rows.append(row)
    return rows


def superpoint_records(frames, purity_threshold):
    records = []
    missing_gt_frames = 0
    for frame in frames:
        if frame.get("gt_instance_ids") is None:
            missing_gt_frames += 1
            continue
        features = torch.as_tensor(
            frame["superpoint_features"], dtype=torch.float32)
        xyz_payload = frame.get("superpoint_xyz")
        xyz = (
            torch.as_tensor(xyz_payload, dtype=torch.float32)
            if xyz_payload is not None else None)
        if xyz is not None and xyz.shape[0] != features.shape[0]:
            xyz = None
        sp_ids = torch.as_tensor(frame["superpoint_ids"], dtype=torch.long)
        gt_ids = torch.as_tensor(frame["gt_instance_ids"], dtype=torch.long)
        labels, purities = dominant_labels(
            sp_ids, gt_ids, features.shape[0], purity_threshold)
        valid = labels >= 0
        features = torch.nn.functional.normalize(features[valid], dim=1)
        valid_xyz = xyz[valid] if xyz is not None else None
        for index, (feature, label, purity) in enumerate(zip(
                features, labels[valid], purities[valid])):
            records.append(dict(
                scene=str(frame["scene_name"]),
                frame=int(frame["frame_index"]),
                instance=int(label),
                purity=float(purity),
                feature=feature.numpy(),
                xyz=(valid_xyz[index].numpy()
                     if valid_xyz is not None else None)))
    if missing_gt_frames:
        warnings.warn(
            f"retrieval metrics skipped {missing_gt_frames} frame(s) without "
            "gt_instance_ids")
    return records


def average_precision(binary_relevance):
    positives = int(binary_relevance.sum())
    if positives == 0:
        return math.nan
    precision = np.cumsum(binary_relevance) / np.arange(
        1, binary_relevance.size + 1)
    return float(precision[binary_relevance].sum() / positives)


def instance_frame_prototypes(records):
    grouped = defaultdict(list)
    for record in records:
        key = (record["scene"], record["frame"], record["instance"])
        grouped[key].append(record["feature"])
    prototypes = []
    for (scene, frame, instance), features in grouped.items():
        feature = np.mean(np.stack(features), axis=0)
        norm = np.linalg.norm(feature)
        if norm <= 1e-12:
            continue
        prototypes.append(dict(
            scene=scene,
            frame=frame,
            instance=instance,
            feature=feature / norm))
    return prototypes


def retrieval_statistics(records):
    prototypes = instance_frame_prototypes(records)
    by_scene = defaultdict(list)
    for record in prototypes:
        by_scene[record["scene"]].append(record)
    top1, aps, same_scores, different_scores = [], [], [], []
    for scene_records in by_scene.values():
        features = np.stack([record["feature"] for record in scene_records])
        frames = np.asarray([record["frame"] for record in scene_records])
        labels = np.asarray([record["instance"] for record in scene_records])
        similarities = features @ features.T
        for index in range(len(scene_records)):
            candidates = frames != frames[index]
            positives = candidates & (labels == labels[index])
            negatives = candidates & (labels != labels[index])
            if not positives.any() or not negatives.any():
                continue
            candidate_scores = similarities[index, candidates]
            candidate_relevance = positives[candidates]
            order = np.argsort(-candidate_scores)
            ranked = candidate_relevance[order]
            top1.append(float(ranked[0]))
            aps.append(average_precision(ranked))
            same_scores.extend(similarities[index, positives].tolist())
            different_scores.extend(similarities[index, negatives].tolist())
    same_mean = float(np.mean(same_scores)) if same_scores else math.nan
    different_mean = (
        float(np.mean(different_scores)) if different_scores else math.nan)
    return dict(
        superpoints=len(records),
        instance_frame_prototypes=len(prototypes),
        retrieval_queries=len(top1),
        retrieval_r1=float(np.mean(top1)) if top1 else math.nan,
        retrieval_map=float(np.mean(aps)) if aps else math.nan,
        same_instance_cosine=same_mean,
        different_instance_cosine=different_mean,
        cosine_margin=same_mean - different_mean)


def centroid_compactness(features, labels):
    features = torch.nn.functional.normalize(features.float(), dim=1)
    values = []
    for label in torch.unique(labels[labels >= 0]).tolist():
        selected = features[labels == label]
        if selected.shape[0] < 2:
            continue
        centroid = torch.nn.functional.normalize(
            selected.mean(dim=0, keepdim=True), dim=1)
        values.extend((selected @ centroid.T).reshape(-1).tolist())
    return float(np.mean(values)) if values else math.nan


def instance_centroid_margin(features, labels):
    features = torch.nn.functional.normalize(features.float(), dim=1)
    unique_labels = torch.unique(labels[labels >= 0])
    if unique_labels.numel() < 2:
        return math.nan
    centroids = []
    for label in unique_labels.tolist():
        centroid = features[labels == label].mean(dim=0, keepdim=True)
        centroids.append(torch.nn.functional.normalize(centroid, dim=1)[0])
    centroids = torch.stack(centroids, dim=0)
    similarities = features @ centroids.T
    label_to_index = {
        int(label): index for index, label in enumerate(unique_labels.tolist())}
    margins = []
    for index, label in enumerate(labels.tolist()):
        if label < 0:
            continue
        own_index = label_to_index[int(label)]
        own_similarity = similarities[index, own_index]
        competing = torch.cat(
            [similarities[index, :own_index], similarities[index, own_index + 1:]])
        margins.append(float(own_similarity - competing.max()))
    return float(np.mean(margins)) if margins else math.nan


def object_feedback_statistics(frames, purity_threshold):
    rows = []
    for frame in frames:
        feedback = frame.get("object_feedback")
        if not isinstance(feedback, dict):
            continue
        inverse = torch.as_tensor(feedback["point_to_voxel"], dtype=torch.long)
        instance_ids = feedback.get("point_instance_ids")
        if instance_ids is None:
            continue
        before = torch.as_tensor(
            feedback["voxel_features_before"], dtype=torch.float32)
        update = torch.as_tensor(feedback["voxel_update"], dtype=torch.float32)
        after_payload = feedback.get("voxel_features_after")
        after = (
            torch.as_tensor(after_payload, dtype=torch.float32)
            if after_payload is not None else before + update)
        labels, _ = dominant_labels(
            inverse, instance_ids, before.shape[0], purity_threshold)
        valid = labels >= 0
        if not valid.any():
            continue
        before_compact = centroid_compactness(before[valid], labels[valid])
        after_compact = centroid_compactness(after[valid], labels[valid])
        margin_before = instance_centroid_margin(before[valid], labels[valid])
        margin_after = instance_centroid_margin(after[valid], labels[valid])
        input_norm = before[valid].norm(dim=1).mean().clamp(min=1e-8)
        update_ratio = float(
            update[valid].norm(dim=1).mean().div(input_norm))
        rows.append(dict(
            scene=str(frame["scene_name"]),
            frame=int(frame["frame_index"]),
            mode=str(feedback.get("mode", "unknown")),
            compactness_before=before_compact,
            compactness_after=after_compact,
            compactness_delta=after_compact - before_compact,
            centroid_margin_before=margin_before,
            centroid_margin_after=margin_after,
            centroid_margin_delta=margin_after - margin_before,
            update_input_norm_ratio=update_ratio))
    if not rows:
        return dict(frames=0), []
    keys = [
        "compactness_before", "compactness_after", "compactness_delta",
        "centroid_margin_before", "centroid_margin_after",
        "centroid_margin_delta", "update_input_norm_ratio"]
    summary = {"frames": len(rows)}
    for key in keys:
        summary[key] = float(np.nanmean([row[key] for row in rows]))
    return summary, rows


def aggregate_feedback_by_scene(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["scene"])].append(row)
    metric_keys = [
        "compactness_before", "compactness_after", "compactness_delta",
        "centroid_margin_before", "centroid_margin_after",
        "centroid_margin_delta", "update_input_norm_ratio"]
    scene_rows = []
    for (variant, scene), frame_rows in grouped.items():
        result = dict(
            variant=variant, scene=scene, frames=len(frame_rows))
        for key in metric_keys:
            values = np.asarray(
                [row[key] for row in frame_rows], dtype=np.float64)
            result[key] = (
                float(np.nanmean(values))
                if np.isfinite(values).any() else math.nan)
        scene_rows.append(result)
    return scene_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap_rows(
        rows, pairs, metric_keys, bootstrap_samples, group_key=None):
    indexed = {}
    groups = set()
    for row in rows:
        group = row[group_key] if group_key is not None else "all"
        groups.add(group)
        indexed[(row["variant"], row["scene"], group)] = row
    rng = np.random.default_rng(2050318485)
    results = []
    for baseline, comparison in pairs:
        for group in sorted(groups):
            scenes = sorted({
                scene for variant, scene, row_group in indexed
                if variant == baseline and row_group == group and
                (comparison, scene, group) in indexed})
            for metric in metric_keys:
                deltas = []
                for scene in scenes:
                    baseline_value = float(indexed[(baseline, scene, group)][metric])
                    comparison_value = float(
                        indexed[(comparison, scene, group)][metric])
                    if (math.isfinite(baseline_value) and
                            math.isfinite(comparison_value)):
                        deltas.append(comparison_value - baseline_value)
                values = np.asarray(deltas, dtype=np.float64)
                if values.size == 0:
                    continue
                if bootstrap_samples > 0:
                    sample_indices = rng.integers(
                        0, values.size, size=(bootstrap_samples, values.size))
                    bootstrap_means = values[sample_indices].mean(axis=1)
                    ci_low, ci_high = np.quantile(
                        bootstrap_means, [0.025, 0.975]).tolist()
                else:
                    ci_low = ci_high = math.nan
                result = dict(
                    baseline=baseline,
                    comparison=comparison,
                    metric=metric,
                    scenes=int(values.size),
                    mean_delta=float(values.mean()),
                    median_delta=float(np.median(values)),
                    positive_scene_fraction=float((values > 0).mean()),
                    ci95_low=float(ci_low),
                    ci95_high=float(ci_high))
                if group_key is not None:
                    result[group_key] = group
                results.append(result)
    return results


def render_plots(output_dir, proposal_rows, representation_rows, thresholds):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for variant, rows in proposal_rows.items():
        row = max(rows, key=lambda item: item["topk"])
        y = [row[f"recall_iou_{threshold:.2f}"] for threshold in thresholds]
        ax.plot(
            thresholds, y, marker="o", linewidth=1.8,
            color=variant_color(variant), label=variant_label(variant))
    ax.set(
        xlabel="IoU threshold",
        ylabel="Proposal recall before temporal merge",
        ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "proposal_recall.png"), dpi=220)
    plt.close(fig)


def render_retrieval_plot(output_dir, representation_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    variants = list(representation_rows)
    if not variants:
        return
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    for axis, key, title in zip(
            axes, ("retrieval_r1", "retrieval_map", "cosine_margin"),
            ("Cross-frame R@1", "Cross-frame mAP", "Cosine margin")):
        axis.bar(
            x, [representation_rows[name][key] for name in variants],
            color=[variant_color(name) for name in variants])
        axis.set_xticks(
            x, [variant_label(name) for name in variants], rotation=25,
            ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "representation_retrieval.png"), dpi=220)
    plt.close(fig)


def render_feedback_summary(output_dir, feedback_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    variants = [
        name for name, row in feedback_rows.items() if row.get("frames", 0) > 0]
    if not variants:
        return
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    specifications = (
        ("compactness_delta", "Within-instance compactness change"),
        ("centroid_margin_delta", "Instance-centroid margin change"),
        ("update_input_norm_ratio", "Residual / input norm"))
    for axis, (key, title) in zip(axes, specifications):
        axis.bar(
            x, [feedback_rows[name].get(key, math.nan) for name in variants],
            color=[variant_color(name) for name in variants])
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xticks(
            x, [variant_label(name) for name in variants], rotation=25,
            ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "ocwe_representation_change.png"), dpi=220)
    plt.close(fig)


def render_instance_embedding_maps(
        output_dir, records_by_variant, preferred_scene=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    prototypes_by_variant = {
        name: instance_frame_prototypes(records)
        for name, records in records_by_variant.items()}
    scene_sets = [
        {record["scene"] for record in records}
        for records in prototypes_by_variant.values() if records]
    if not scene_sets:
        return None
    common_scenes = set.intersection(*scene_sets)
    selected_scene = None
    selected_instances = []
    scene_order = sorted(common_scenes)
    if preferred_scene in common_scenes:
        scene_order.remove(preferred_scene)
        scene_order.insert(0, preferred_scene)
    for scene in scene_order:
        label_sets = []
        for records in prototypes_by_variant.values():
            label_counts = defaultdict(int)
            for record in records:
                if record["scene"] == scene:
                    label_counts[record["instance"]] += 1
            labels = {
                label for label, count in label_counts.items() if count >= 2}
            label_sets.append(labels)
        common_labels = set.intersection(*label_sets)
        if len(common_labels) >= 3:
            selected_scene = scene
            selected_instances = sorted(common_labels)[:8]
            break
    if selected_scene is None:
        return None

    variants = list(prototypes_by_variant)
    selected_by_variant = {}
    all_features = []
    for variant in variants:
        selected = [
            record for record in prototypes_by_variant[variant]
            if record["scene"] == selected_scene and
            record["instance"] in selected_instances]
        selected_by_variant[variant] = selected
        all_features.extend(record["feature"] for record in selected)
    all_features = np.stack(all_features)
    global_mean = all_features.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(
        all_features - global_mean, full_matrices=False)
    basis = right[:2].T
    projected_by_variant = {
        variant: (np.stack([record["feature"] for record in selected]) -
                  global_mean) @ basis
        for variant, selected in selected_by_variant.items()}
    all_projected = np.concatenate(list(projected_by_variant.values()), axis=0)
    x_padding = max(1e-6, 0.05 * np.ptp(all_projected[:, 0]))
    y_padding = max(1e-6, 0.05 * np.ptp(all_projected[:, 1]))
    x_limits = (
        float(all_projected[:, 0].min() - x_padding),
        float(all_projected[:, 0].max() + x_padding))
    y_limits = (
        float(all_projected[:, 1].min() - y_padding),
        float(all_projected[:, 1].max() + y_padding))
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        1, len(variants), figsize=(3.1 * len(variants), 3.1), squeeze=False)
    for axis, variant in zip(axes[0], variants):
        selected = selected_by_variant[variant]
        projected = projected_by_variant[variant]
        for color_index, instance in enumerate(selected_instances):
            indices = [
                index for index, record in enumerate(selected)
                if record["instance"] == instance]
            if not indices:
                continue
            axis.scatter(
                projected[indices, 0], projected[indices, 1], s=15,
                color=colors(color_index % 10), alpha=0.75,
                label=f"inst. {instance}")
        axis.set_title(variant_label(variant))
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"GT-conditioned instance-frame prototypes: {selected_scene}")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "instance_embedding_pca.png"), dpi=220)
    plt.close(fig)
    return dict(
        scene=selected_scene,
        instances=selected_instances,
        rule=("preferred scene when valid, otherwise lexicographically first "
              "scene with at least three instances observed in at least two "
              "captured frames for every variant"),
        projection="one shared PCA basis fitted across all displayed variants")


def render_anchor_similarity_maps(
        output_dir, records_by_variant, preferred_scene=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    variants = [
        name for name in (
            "host", "embodieddino", "ocwe_no_write", "ocwe_aligned",
            "ocwe_shuffled")
        if records_by_variant.get(name)]
    if len(variants) < 2:
        return None

    indexed = {}
    for variant in variants:
        by_scene_frame = defaultdict(list)
        for record in records_by_variant[variant]:
            if record.get("xyz") is not None:
                by_scene_frame[(record["scene"], record["frame"])].append(
                    record)
        indexed[variant] = by_scene_frame
    common_keys = set.intersection(*[
        set(indexed[variant]) for variant in variants])
    scenes = sorted({scene for scene, _ in common_keys})
    if preferred_scene in scenes:
        scenes.remove(preferred_scene)
        scenes.insert(0, preferred_scene)

    selected = None
    for scene in scenes:
        frames = sorted({
            frame for current_scene, frame in common_keys
            if current_scene == scene})
        for anchor_index, anchor_frame in enumerate(frames[:-1]):
            for target_frame in frames[anchor_index + 1:]:
                common_instances = None
                target_instance_union = set()
                for variant in variants:
                    anchor_instances = {
                        record["instance"]
                        for record in indexed[variant][(scene, anchor_frame)]}
                    target_instances = {
                        record["instance"]
                        for record in indexed[variant][(scene, target_frame)]}
                    visible = anchor_instances & target_instances
                    common_instances = (
                        visible if common_instances is None
                        else common_instances & visible)
                    target_instance_union |= target_instances
                if common_instances and len(target_instance_union) >= 3:
                    selected = (
                        scene, anchor_frame, target_frame,
                        sorted(common_instances)[0])
                    break
            if selected is not None:
                break
        if selected is not None:
            break
    if selected is None:
        return None

    scene, anchor_frame, target_frame, instance = selected
    plot_rows = []
    all_similarities = []
    for variant in variants:
        anchor_records = [
            record for record in indexed[variant][(scene, anchor_frame)]
            if record["instance"] == instance]
        target_records = indexed[variant][(scene, target_frame)]
        anchor = np.mean(
            np.stack([record["feature"] for record in anchor_records]), axis=0)
        anchor /= max(np.linalg.norm(anchor), 1e-12)
        target_features = np.stack([
            record["feature"] for record in target_records])
        similarities = target_features @ anchor
        xyz = np.stack([record["xyz"] for record in target_records])
        labels = np.asarray([
            record["instance"] for record in target_records])
        plot_rows.append((variant, xyz, labels, similarities))
        all_similarities.append(similarities)

    all_similarities = np.concatenate(all_similarities)
    vmin, vmax = np.quantile(all_similarities, [0.02, 0.98]).tolist()
    if vmax - vmin < 1e-6:
        vmin, vmax = float(vmin - 0.5), float(vmax + 0.5)
    fig, axes = plt.subplots(
        1, len(plot_rows), figsize=(3.1 * len(plot_rows), 3.15),
        squeeze=False)
    for axis, (variant, xyz, labels, similarities) in zip(
            axes[0], plot_rows):
        image = axis.scatter(
            xyz[:, 0], xyz[:, 1], c=similarities, s=7, cmap="viridis",
            vmin=vmin, vmax=vmax, rasterized=True)
        target = labels == instance
        if target.any():
            axis.scatter(
                xyz[target, 0], xyz[target, 1], s=18, facecolors="none",
                edgecolors="black", linewidths=0.55)
        axis.set_title(variant_label(variant))
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.72,
        label="Cosine similarity")
    fig.suptitle(
        f"{scene}: frame {anchor_frame} instance {instance} -> "
        f"frame {target_frame}")
    fig.savefig(
        os.path.join(output_dir, "anchor_similarity_map.png"), dpi=220,
        bbox_inches="tight")
    plt.close(fig)
    return dict(
        scene=scene, anchor_frame=anchor_frame, target_frame=target_frame,
        instance=instance,
        rule=("preferred scene when valid, otherwise lexicographically first "
              "common scene/frame pair with one shared instance and at least "
              "three target-frame instances"),
        color_scale="shared 2nd--98th percentile cosine range",
        ground_truth_use=("instance identity selects and outlines the anchor; "
                          "it does not enter model inference"))


def render_feedback_maps(output_dir, frames_by_variant, preferred_scene=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    indexed = {}
    for variant, frames in frames_by_variant.items():
        indexed[variant] = {}
        for frame in frames:
            feedback = frame.get("object_feedback")
            if (isinstance(feedback, dict) and
                    feedback.get("point_xyz") is not None):
                key = (str(frame["scene_name"]), int(frame["frame_index"]))
                indexed[variant][key] = frame
    comparable_variants = [
        name for name in ("ocwe_no_write", "ocwe_aligned", "ocwe_shuffled")
        if indexed.get(name)]
    if len(comparable_variants) < 2:
        return None
    common_keys = set.intersection(*[
        set(indexed[name]) for name in comparable_variants])
    if not common_keys:
        return None
    ordered_keys = sorted(
        common_keys,
        key=lambda key: (
            0 if preferred_scene is not None and key[0] == preferred_scene
            else 1,
            key[0], key[1]))
    selected_key = ordered_keys[0]
    selected = []
    for variant in comparable_variants:
        frame = indexed[variant][selected_key]
        feedback = frame["object_feedback"]
        xyz = torch.as_tensor(feedback["point_xyz"]).float().numpy()
        inverse = torch.as_tensor(
            feedback["point_to_voxel"], dtype=torch.long)
        update = torch.as_tensor(
            feedback["voxel_update"], dtype=torch.float32)
        energy = update.norm(dim=1)[inverse].numpy()
        selected.append((variant, xyz, energy))
    positive_parts = [
        energy[energy > 0] for _, _, energy in selected if (energy > 0).any()]
    positive = (
        np.concatenate(positive_parts) if positive_parts else np.asarray([]))
    vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
    fig, axes = plt.subplots(
        1, len(selected), figsize=(3.4 * len(selected), 3.2), squeeze=False)
    for axis, (variant, xyz, energy) in zip(axes[0], selected):
        image = axis.scatter(
            xyz[:, 0], xyz[:, 1], c=energy, s=1.5, cmap="magma",
            vmin=0.0, vmax=vmax, rasterized=True)
        axis.set_title(variant_label(variant))
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.subplots_adjust(top=0.82, right=0.88, wspace=0.18)
    color_axis = fig.add_axes([0.90, 0.20, 0.015, 0.55])
    fig.colorbar(
        image, cax=color_axis, label="Object-feedback residual norm")
    fig.suptitle(f"{selected_key[0]} frame {selected_key[1]}")
    fig.savefig(
        os.path.join(output_dir, "ocwe_update_energy.png"), dpi=220,
        bbox_inches="tight")
    plt.close(fig)
    return dict(
        scene=selected_key[0], frame=selected_key[1],
        rule=("preferred scene's first shared frame when available, otherwise "
              "the lexicographically first frame shared by all variants"),
        color_scale="shared 0--99th percentile of positive residual norms")


def render_paired_diagnostic_summary(
        output_dir, proposal_delta_rows, representation_delta_rows,
        feedback_delta_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def matching(rows, baseline, comparison, metrics, **filters):
        selected = []
        for metric, label in metrics:
            candidates = [
                row for row in rows
                if row["baseline"] == baseline and
                row["comparison"] == comparison and
                row["metric"] == metric and
                all(row.get(key) == value for key, value in filters.items())]
            if candidates:
                selected.append((label, candidates[0]))
        return selected

    panels = [
        (matching(
            proposal_delta_rows, "host", "embodieddino",
            [("mean_best_iou", "Mean best IoU"),
             ("recall_iou_0.25", "Recall@0.25"),
             ("recall_iou_0.50", "Recall@0.50"),
             ("recall_iou_0.75", "Recall@0.75")], topk=20),
         "Pre-merge proposals\nEmbodiedDINO - host", "#2563A6"),
        (matching(
            representation_delta_rows, "host", "embodieddino",
            [("retrieval_r1", "Cross-frame R@1"),
             ("retrieval_map", "Cross-frame mAP"),
             ("cosine_margin", "Cosine margin")]),
         "Instance retrieval\nEmbodiedDINO - host", "#2563A6"),
        (matching(
            feedback_delta_rows, "ocwe_no_write", "ocwe_aligned",
            [("compactness_after", "Compactness"),
             ("centroid_margin_after", "Centroid margin")]) +
         matching(
            feedback_delta_rows, "ocwe_shuffled", "ocwe_aligned",
            [("compactness_after", "Compactness vs shuffled"),
             ("centroid_margin_after", "Margin vs shuffled")]),
         "Object-level feedback\naligned - control", "#2E7D5B")]

    active_panels = [panel for panel in panels if panel[0]]
    if not active_panels:
        return
    fig, axes = plt.subplots(
        1, len(active_panels),
        figsize=(3.5 * len(active_panels), 3.25), squeeze=False)
    for axis, (rows, title, color) in zip(axes[0], active_panels):
        y = np.arange(len(rows))
        means = np.asarray([row["mean_delta"] for _, row in rows])
        lower = means - np.asarray([row["ci95_low"] for _, row in rows])
        upper = np.asarray([row["ci95_high"] for _, row in rows]) - means
        axis.errorbar(
            means, y, xerr=np.vstack([lower, upper]), fmt="o",
            color=color, ecolor=color, capsize=3, linewidth=1.3)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(y, [label for label, _ in rows])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.supxlabel("Mean paired-scene delta with 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "representation_diagnostic_summary.png"),
        dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    dump_dirs = parse_dumps(args.dump)
    os.makedirs(args.output_dir, exist_ok=True)
    proposal_by_variant = {}
    representation_by_variant = {}
    feedback_by_variant = {}
    feedback_frame_rows = []
    records_by_variant = {}
    frames_by_variant = {}
    coverage_by_variant = {}
    validation_errors = []
    all_proposal_rows = []
    proposal_scene_rows = []
    representation_scene_rows = []
    for variant, path in dump_dirs.items():
        frames = load_frames(path)
        frames_by_variant[variant] = frames
        frames_with_gt = sum(
            frame.get("gt_instance_ids") is not None for frame in frames)
        scene_count = len({str(frame["scene_name"]) for frame in frames})
        coverage_by_variant[variant] = dict(
            frames=len(frames),
            scenes=scene_count,
            frames_with_gt=frames_with_gt)
        if len(frames) < args.require_min_frames_per_variant:
            validation_errors.append(
                f"{variant}: found {len(frames)} frames, expected at least "
                f"{args.require_min_frames_per_variant}")
        if scene_count < args.require_min_scenes_per_variant:
            validation_errors.append(
                f"{variant}: found {scene_count} scenes, expected at least "
                f"{args.require_min_scenes_per_variant}")
        if args.require_gt_every_frame and frames_with_gt != len(frames):
            validation_errors.append(
                f"{variant}: GT instance ids present in {frames_with_gt}/"
                f"{len(frames)} frames")
        proposal_rows = proposal_statistics(
            frames, args.topk, args.iou_thresholds,
            args.min_instance_points)
        for row in proposal_rows:
            row["variant"] = variant
        proposal_by_variant[variant] = proposal_rows
        all_proposal_rows.extend(proposal_rows)
        frames_by_scene = defaultdict(list)
        for frame in frames:
            frames_by_scene[str(frame["scene_name"])].append(frame)
        for scene, scene_frames in frames_by_scene.items():
            scene_rows = proposal_statistics(
                scene_frames, args.topk, args.iou_thresholds,
                args.min_instance_points, warn_missing=False)
            for row in scene_rows:
                row["variant"] = variant
                row["scene"] = scene
            proposal_scene_rows.extend(scene_rows)
        records = superpoint_records(frames, args.purity_threshold)
        records_by_variant[variant] = records
        representation_by_variant[variant] = retrieval_statistics(records)
        if (args.require_retrieval_queries and
                representation_by_variant[variant]["retrieval_queries"] < 1):
            validation_errors.append(
                f"{variant}: no valid cross-frame retrieval queries")
        records_by_scene = defaultdict(list)
        for record in records:
            records_by_scene[record["scene"]].append(record)
        for scene, scene_records in records_by_scene.items():
            row = retrieval_statistics(scene_records)
            row["variant"] = variant
            row["scene"] = scene
            representation_scene_rows.append(row)
        feedback_summary, variant_feedback_rows = object_feedback_statistics(
            frames, args.purity_threshold)
        feedback_by_variant[variant] = feedback_summary
        for row in variant_feedback_rows:
            row["variant"] = variant
        feedback_frame_rows.extend(variant_feedback_rows)

    write_csv(os.path.join(args.output_dir, "proposal_recall.csv"),
              all_proposal_rows)
    write_csv(os.path.join(args.output_dir, "proposal_recall_by_scene.csv"),
              proposal_scene_rows)
    write_csv(
        os.path.join(args.output_dir, "representation_retrieval_by_scene.csv"),
        representation_scene_rows)
    feedback_scene_rows = aggregate_feedback_by_scene(feedback_frame_rows)
    write_csv(
        os.path.join(args.output_dir, "object_feedback_by_frame.csv"),
        feedback_frame_rows)
    write_csv(
        os.path.join(args.output_dir, "object_feedback_by_scene.csv"),
        feedback_scene_rows)
    pairs = [
        ("host", "embodieddino"),
        ("ocwe_no_write", "ocwe_aligned"),
        ("ocwe_shuffled", "ocwe_aligned")]
    proposal_metric_keys = ["mean_best_iou"] + [
        f"recall_iou_{threshold:.2f}" for threshold in args.iou_thresholds]
    proposal_delta_rows = paired_bootstrap_rows(
        proposal_scene_rows, pairs, proposal_metric_keys,
        args.bootstrap_samples, group_key="topk")
    representation_delta_rows = paired_bootstrap_rows(
        representation_scene_rows, pairs,
        ["retrieval_r1", "retrieval_map", "cosine_margin"],
        args.bootstrap_samples)
    feedback_delta_rows = paired_bootstrap_rows(
        feedback_scene_rows, pairs,
        ["compactness_after", "centroid_margin_after",
         "compactness_delta", "centroid_margin_delta"],
        args.bootstrap_samples)
    write_csv(
        os.path.join(args.output_dir, "proposal_paired_scene_deltas.csv"),
        proposal_delta_rows)
    write_csv(
        os.path.join(
            args.output_dir, "representation_paired_scene_deltas.csv"),
        representation_delta_rows)
    write_csv(
        os.path.join(
            args.output_dir, "object_feedback_paired_scene_deltas.csv"),
        feedback_delta_rows)
    summary = dict(
        coverage=coverage_by_variant,
        proposal=proposal_by_variant,
        representation=representation_by_variant,
        object_feedback=feedback_by_variant,
        paired_scene_deltas=dict(
            proposal=proposal_delta_rows,
            representation=representation_delta_rows,
            object_feedback=feedback_delta_rows),
        protocol=dict(
            purity_threshold=args.purity_threshold,
            min_instance_points=args.min_instance_points,
            topk=args.topk,
            iou_thresholds=args.iou_thresholds,
            bootstrap_samples=args.bootstrap_samples,
            proposal_stage=(
                "post-threshold-and-NMS frame proposals before temporal "
                "association and merging")),
        validation_errors=validation_errors)
    summary_path = os.path.join(
        args.output_dir, "representation_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    render_plots(
        args.output_dir, proposal_by_variant, representation_by_variant,
        args.iou_thresholds)
    render_retrieval_plot(args.output_dir, representation_by_variant)
    embedding_selection = render_instance_embedding_maps(
        args.output_dir, records_by_variant,
        args.preferred_visualization_scene)
    anchor_selection = render_anchor_similarity_maps(
        args.output_dir, records_by_variant,
        args.preferred_visualization_scene)
    render_feedback_summary(args.output_dir, feedback_by_variant)
    feedback_selection = render_feedback_maps(
        args.output_dir, frames_by_variant,
        args.preferred_visualization_scene)
    render_paired_diagnostic_summary(
        args.output_dir, proposal_delta_rows, representation_delta_rows,
        feedback_delta_rows)
    with open(os.path.join(
            args.output_dir, "visualization_selection.json"), "w") as handle:
        json.dump(dict(
            anchor_similarity=anchor_selection,
            instance_embedding=embedding_selection,
            ocwe_update_energy=feedback_selection),
            handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_errors:
        raise RuntimeError(
            "representation diagnostic validation failed: " +
            "; ".join(validation_errors))


if __name__ == "__main__":
    main()
