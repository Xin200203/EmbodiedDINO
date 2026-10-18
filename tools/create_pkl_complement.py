#!/usr/bin/env python3

import argparse
import copy
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the scene complement of a subset dataset PKL.")
    parser.add_argument("full_pkl")
    parser.add_argument("subset_pkl")
    parser.add_argument("output_pkl")
    return parser.parse_args()


def load_pkl(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def data_list(payload):
    return payload["data_list"] if isinstance(payload, dict) else payload


def scene_id(item):
    point_cloud = item.get("point_cloud", {})
    value = point_cloud.get("lidar_idx")
    if value is None:
        raise KeyError("missing point_cloud.lidar_idx")
    return value


def main():
    args = parse_args()
    full = load_pkl(args.full_pkl)
    subset = load_pkl(args.subset_pkl)
    full_items = data_list(full)
    subset_items = data_list(subset)

    full_ids = [scene_id(item) for item in full_items]
    subset_ids = {scene_id(item) for item in subset_items}
    if len(full_ids) != len(set(full_ids)):
        raise ValueError("full PKL contains duplicate scene IDs")
    if len(subset_ids) != len(subset_items):
        raise ValueError("subset PKL contains duplicate scene IDs")
    missing = subset_ids.difference(full_ids)
    if missing:
        raise ValueError(f"subset contains {len(missing)} IDs absent from full PKL")

    complement_items = [
        item for item in full_items if scene_id(item) not in subset_ids
    ]
    complement_ids = {scene_id(item) for item in complement_items}
    if complement_ids.intersection(subset_ids):
        raise AssertionError("subset and complement overlap")
    if len(complement_items) + len(subset_items) != len(full_items):
        raise AssertionError("subset and complement do not partition full PKL")

    if isinstance(full, dict):
        output = copy.copy(full)
        output["data_list"] = complement_items
    else:
        output = complement_items

    output_dir = os.path.dirname(os.path.abspath(args.output_pkl))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_pkl, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"full={len(full_items)} subset={len(subset_items)} "
        f"complement={len(complement_items)} output={args.output_pkl}")


if __name__ == "__main__":
    main()
