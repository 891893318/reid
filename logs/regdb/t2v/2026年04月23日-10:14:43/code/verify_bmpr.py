import warnings

import numpy as np
import torch
import torch.nn.functional as F

import datasets
import models
from main import get_parser
from utils import set_seed
from wsl import CMA


warnings.filterwarnings("ignore")


def infer_num_classes(args):
    if args.dataset == "sysu":
        return 395
    if args.dataset == "regdb":
        return 206
    if args.dataset == "llcm":
        return 713
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def invert_relabel_dict(relabel_dict, num_classes):
    inverse = {}
    for gt_label, pseudo_label in relabel_dict.items():
        inverse[int(pseudo_label)] = int(gt_label)
    for idx in range(num_classes):
        inverse.setdefault(idx, idx)
    return inverse


def build_class_stats(scores, labels, num_classes, device):
    scores = scores.to(device)
    labels = labels.to(device)
    proto = torch.zeros(num_classes, scores.size(1), device=device)
    counts = torch.bincount(labels, minlength=num_classes).float().to(device)
    proto.index_add_(0, labels, scores)
    proto = proto / counts.unsqueeze(1).clamp_min(1.0)
    active = counts > 0
    return proto, counts, active


def top12_margin(matrix):
    top_vals, _ = torch.topk(matrix, k=min(2, matrix.size(1)), dim=1)
    if top_vals.size(1) == 1:
        return top_vals[:, 0]
    return top_vals[:, 0] - top_vals[:, 1]


def minmax_norm(vec, mask):
    out = torch.zeros_like(vec)
    if mask.sum() == 0:
        return out
    valid = vec[mask]
    min_v = valid.min()
    max_v = valid.max()
    if torch.isclose(max_v, min_v):
        out[mask] = 1.0
    else:
        out[mask] = (valid - min_v) / (max_v - min_v)
    return out


def pair_correct(idx, assign, src_inverse, dst_inverse):
    return src_inverse[int(idx)] == dst_inverse[int(assign)]


def bucket_stats(mask, assign, reliability, src_inverse, dst_inverse):
    indices = torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()
    total = len(indices)
    if total == 0:
        return {
            "precision": 0.0,
            "count": 0,
            "correct": 0,
            "mean_rel": 0.0,
        }
    assign_cpu = assign.cpu()
    rel_cpu = reliability.cpu()
    correct = 0
    rel_values = []
    for idx in indices:
        rel_values.append(float(rel_cpu[idx].item()))
        if pair_correct(idx, int(assign_cpu[idx].item()), src_inverse, dst_inverse):
            correct += 1
    return {
        "precision": correct / total,
        "count": total,
        "correct": correct,
        "mean_rel": float(np.mean(rel_values)),
    }


def top1_stats(assign, active_mask, src_inverse, dst_inverse):
    indices = torch.nonzero(active_mask, as_tuple=False).flatten().cpu().tolist()
    total = len(indices)
    correct = 0
    assign_cpu = assign.cpu()
    for idx in indices:
        if pair_correct(idx, int(assign_cpu[idx].item()), src_inverse, dst_inverse):
            correct += 1
    return correct / max(total, 1), correct, total


def print_block(title, rows):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, value in rows:
        print(f"{key}: {value}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.model_path == "default":
        raise ValueError("Please provide a checkpoint with --model-path for BMPR verification.")

    args.num_classes = infer_num_classes(args)
    set_seed(args.seed)

    print_block(
        "BMPR Verification Setup",
        [
            ("dataset", args.dataset),
            ("arch", args.arch),
            ("trial", args.trial),
            ("test_mode", args.test_mode),
            ("checkpoint", args.model_path),
            ("bmpr_high_th", args.bmpr_high_th),
            ("bmpr_mid_th", args.bmpr_mid_th),
            ("bmpr_margin_weight", args.bmpr_margin_weight),
            ("bmpr_proto_weight", args.bmpr_proto_weight),
        ],
    )

    dataset = datasets.create(args)
    model = models.create(args)
    model.resume_model(args.model_path)
    cma = CMA(args)
    cma.extract(args, model, dataset)

    device = model.device
    vis_proto, _, vis_active = build_class_stats(cma.vis_scores, cma.rgb_ids, args.num_classes, device)
    ir_proto, _, ir_active = build_class_stats(cma.ir_scores, cma.ir_ids, args.num_classes, device)

    row_margin = top12_margin(vis_proto)
    col_margin = top12_margin(ir_proto)
    row_assign = torch.argmax(vis_proto, dim=1)
    col_assign = torch.argmax(ir_proto, dim=1)

    vis_memory = F.normalize(cma.vis_memory.to(device), dim=1)
    ir_memory = F.normalize(cma.ir_memory.to(device), dim=1)
    proto_matrix = ((torch.matmul(vis_memory, ir_memory.t()) + 1.0) * 0.5).clamp(0.0, 1.0)
    proto_gap = top12_margin(proto_matrix)

    active_rows = vis_active
    active_cols = ir_active
    row_margin_norm = minmax_norm(row_margin, active_rows)
    col_margin_norm = minmax_norm(col_margin, active_cols)
    proto_gap_norm = minmax_norm(proto_gap, active_rows)
    proto_col_gap_norm = minmax_norm(top12_margin(proto_matrix.t()), active_cols)

    row_reliability = (
        args.bmpr_margin_weight * row_margin_norm +
        args.bmpr_proto_weight * proto_gap_norm
    )
    col_reliability = (
        args.bmpr_margin_weight * col_margin_norm +
        args.bmpr_proto_weight * proto_col_gap_norm
    )

    common_row_mask = active_rows & (row_reliability >= args.bmpr_high_th)
    common_col_mask = active_cols & (col_reliability >= args.bmpr_high_th)
    specific_row_mask = active_rows & (row_reliability >= args.bmpr_mid_th) & (row_reliability < args.bmpr_high_th)
    specific_col_mask = active_cols & (col_reliability >= args.bmpr_mid_th) & (col_reliability < args.bmpr_high_th)
    remain_row_mask = active_rows & (row_reliability < args.bmpr_mid_th)
    remain_col_mask = active_cols & (col_reliability < args.bmpr_mid_th)

    rgb_inverse = invert_relabel_dict(dataset.train_rgb.relabel_dict, args.num_classes)
    ir_inverse = invert_relabel_dict(dataset.train_ir.relabel_dict, args.num_classes)

    row_top1_acc, row_correct, row_total = top1_stats(row_assign, active_rows, rgb_inverse, ir_inverse)
    col_top1_acc, col_correct, col_total = top1_stats(col_assign, active_cols, ir_inverse, rgb_inverse)

    common_row = bucket_stats(common_row_mask, row_assign, row_reliability, rgb_inverse, ir_inverse)
    specific_row = bucket_stats(specific_row_mask, row_assign, row_reliability, rgb_inverse, ir_inverse)
    remain_row = bucket_stats(remain_row_mask, row_assign, row_reliability, rgb_inverse, ir_inverse)
    common_col = bucket_stats(common_col_mask, col_assign, col_reliability, ir_inverse, rgb_inverse)
    specific_col = bucket_stats(specific_col_mask, col_assign, col_reliability, ir_inverse, rgb_inverse)
    remain_col = bucket_stats(remain_col_mask, col_assign, col_reliability, ir_inverse, rgb_inverse)

    row_rel_correct = row_reliability[
        torch.tensor(
            [pair_correct(idx, int(row_assign[idx].item()), rgb_inverse, ir_inverse) for idx in range(args.num_classes)],
            device=device,
            dtype=torch.bool,
        )
        & active_rows
    ]
    row_rel_wrong = row_reliability[
        ~torch.tensor(
            [pair_correct(idx, int(row_assign[idx].item()), rgb_inverse, ir_inverse) for idx in range(args.num_classes)],
            device=device,
            dtype=torch.bool,
        )
        & active_rows
    ]
    col_rel_correct = col_reliability[
        torch.tensor(
            [pair_correct(idx, int(col_assign[idx].item()), ir_inverse, rgb_inverse) for idx in range(args.num_classes)],
            device=device,
            dtype=torch.bool,
        )
        & active_cols
    ]
    col_rel_wrong = col_reliability[
        ~torch.tensor(
            [pair_correct(idx, int(col_assign[idx].item()), ir_inverse, rgb_inverse) for idx in range(args.num_classes)],
            device=device,
            dtype=torch.bool,
        )
        & active_cols
    ]

    print_block(
        "Legacy Top1 Quality",
        [
            ("row_top1", f"{row_top1_acc:.4f} ({row_correct}/{row_total})"),
            ("col_top1", f"{col_top1_acc:.4f} ({col_correct}/{col_total})"),
        ],
    )

    print_block(
        "BMPR Route Precision",
        [
            ("common_row", f"{common_row['precision']:.4f} ({common_row['correct']}/{common_row['count']})"),
            ("specific_row", f"{specific_row['precision']:.4f} ({specific_row['correct']}/{specific_row['count']})"),
            ("remain_row", f"{remain_row['precision']:.4f} ({remain_row['correct']}/{remain_row['count']})"),
            ("common_col", f"{common_col['precision']:.4f} ({common_col['correct']}/{common_col['count']})"),
            ("specific_col", f"{specific_col['precision']:.4f} ({specific_col['correct']}/{specific_col['count']})"),
            ("remain_col", f"{remain_col['precision']:.4f} ({remain_col['correct']}/{remain_col['count']})"),
        ],
    )

    print_block(
        "BMPR Route Coverage",
        [
            ("common_row_count", common_row["count"]),
            ("specific_row_count", specific_row["count"]),
            ("remain_row_count", remain_row["count"]),
            ("common_col_count", common_col["count"]),
            ("specific_col_count", specific_col["count"]),
            ("remain_col_count", remain_col["count"]),
        ],
    )

    print_block(
        "Reliability Distribution",
        [
            ("common_row_mean_rel", f"{common_row['mean_rel']:.4f}"),
            ("specific_row_mean_rel", f"{specific_row['mean_rel']:.4f}"),
            ("remain_row_mean_rel", f"{remain_row['mean_rel']:.4f}"),
            ("row_correct_mean_rel", f"{row_rel_correct.mean().item():.4f}" if row_rel_correct.numel() else "0.0000"),
            ("row_wrong_mean_rel", f"{row_rel_wrong.mean().item():.4f}" if row_rel_wrong.numel() else "0.0000"),
            ("col_correct_mean_rel", f"{col_rel_correct.mean().item():.4f}" if col_rel_correct.numel() else "0.0000"),
            ("col_wrong_mean_rel", f"{col_rel_wrong.mean().item():.4f}" if col_rel_wrong.numel() else "0.0000"),
        ],
    )

    print_block(
        "Component Diagnostics",
        [
            ("row_margin_mean", f"{row_margin[active_rows].mean().item():.4f}" if active_rows.any() else "0.0000"),
            ("col_margin_mean", f"{col_margin[active_cols].mean().item():.4f}" if active_cols.any() else "0.0000"),
            ("proto_gap_mean", f"{proto_gap[active_rows].mean().item():.4f}" if active_rows.any() else "0.0000"),
        ],
    )

    print("\nInterpretation:")
    print("- High-confidence `common` should be both precise and non-trivially sized.")
    print("- `specific` should have lower precision than `common`, but clearly higher than `remain`.")
    print("- If all samples collapse into `common`, BMPR thresholds or reliability design are too weak.")
    print("- If `row_correct_mean_rel` is not clearly above `row_wrong_mean_rel`, reliability is not separating noise.")


if __name__ == "__main__":
    main()
