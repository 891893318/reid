import argparse
import json
import os

import numpy as np
import torch

from diagnose_multiproto import (
    build_model,
    build_prototypes,
    collect_paths,
    extract_features,
)


def hard_negative_diagnostics(query_features, query_labels, prototypes, prototype_labels, margin):
    similarities = query_features @ prototypes.T
    positive_mask = torch.from_numpy(
        query_labels[:, None] == prototype_labels.numpy()[None, :]
    )
    valid_queries = positive_mask.any(dim=1)
    similarities = similarities[valid_queries]
    positive_mask = positive_mask[valid_queries]

    positive_similarity = similarities.masked_fill(~positive_mask, -1.0).max(dim=1).values
    hard_negative_similarity = similarities.masked_fill(positive_mask, -1.0).max(dim=1).values
    separation = positive_similarity - hard_negative_similarity
    violations = torch.relu(margin - separation)
    incorrect = separation <= 0

    quantiles = torch.quantile(separation, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "valid_queries": int(valid_queries.sum()),
        "top1": float((separation > 0).float().mean()),
        "positive_similarity_mean": float(positive_similarity.mean()),
        "hard_negative_similarity_mean": float(hard_negative_similarity.mean()),
        "separation_mean": float(separation.mean()),
        "separation_p10": float(quantiles[0]),
        "separation_p50": float(quantiles[1]),
        "separation_p90": float(quantiles[2]),
        "active_ratio": float((violations > 0).float().mean()),
        "hinge_loss_mean": float(violations.mean()),
        "incorrect_ratio": float(incorrect.float().mean()),
        "incorrect_active_ratio": float(
            ((violations > 0) & incorrect).float().sum() / incorrect.float().sum().clamp_min(1)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--search-mode", choices=["all", "indoor"], required=True)
    parser.add_argument("--data-root", default="/root/data/SYSU-MM01")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rgb_cameras = [1, 2, 4, 5] if args.search_mode == "all" else [1, 2]
    rgb_paths, rgb_labels, rgb_cameras = collect_paths(args.data_root, rgb_cameras)
    ir_paths, ir_labels, ir_cameras = collect_paths(args.data_root, [3, 6])

    model = build_model(args.checkpoint, device)
    rgb_features = extract_features(
        model, rgb_paths, "rgb", device, args.batch_size, args.workers
    )
    ir_features = extract_features(
        model, ir_paths, "ir", device, args.batch_size, args.workers
    )
    rgb_prototypes, rgb_prototype_labels = build_prototypes(
        rgb_features, rgb_labels, rgb_cameras, "single"
    )
    ir_prototypes, ir_prototype_labels = build_prototypes(
        ir_features, ir_labels, ir_cameras, "single"
    )

    result = {
        "search_mode": args.search_mode,
        "checkpoint": args.checkpoint,
        "margin": args.margin,
        "rgb_to_ir": hard_negative_diagnostics(
            rgb_features, rgb_labels, ir_prototypes, ir_prototype_labels, args.margin
        ),
        "ir_to_rgb": hard_negative_diagnostics(
            ir_features, ir_labels, rgb_prototypes, rgb_prototype_labels, args.margin
        ),
    }
    result["mean_active_ratio"] = float(np.mean([
        result["rgb_to_ir"]["active_ratio"],
        result["ir_to_rgb"]["active_ratio"],
    ]))
    result["mean_separation"] = float(np.mean([
        result["rgb_to_ir"]["separation_mean"],
        result["ir_to_rgb"]["separation_mean"],
    ]))

    text = json.dumps(result, indent=2, ensure_ascii=True)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as file:
            file.write(text + "\n")


if __name__ == "__main__":
    main()
