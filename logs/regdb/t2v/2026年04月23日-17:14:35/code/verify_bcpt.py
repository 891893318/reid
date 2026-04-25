import argparse
import warnings

import numpy as np
import torch

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


def map_pair_accuracy(pair_dict, src_inverse, dst_inverse):
    total = len(pair_dict)
    if total == 0:
        return 0.0, 0, 0
    correct = 0
    for src, dst in pair_dict.items():
        if src_inverse[int(src)] == dst_inverse[int(dst)]:
            correct += 1
    return correct / total, correct, total


def mask_precision(mask, assign, src_inverse, dst_inverse):
    indices = torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()
    total = len(indices)
    if total == 0:
        return 0.0, 0, 0
    correct = 0
    assign_cpu = assign.cpu()
    for idx in indices:
        if src_inverse[idx] == dst_inverse[int(assign_cpu[idx].item())]:
            correct += 1
    return correct / total, correct, total


def topk_hit_rate(matrix, src_inverse, dst_inverse, topk):
    if matrix.numel() == 0:
        return 0.0, 0, 0
    k = min(topk, matrix.size(1))
    values, indices = torch.topk(matrix, k=k, dim=1)
    total = 0
    hit = 0
    for row_idx in range(matrix.size(0)):
        target_gt = src_inverse[row_idx]
        dst_candidates = [dst_inverse[int(col.item())] for col in indices[row_idx]]
        if target_gt in dst_candidates:
            hit += 1
        total += 1
    return hit / max(total, 1), hit, total


def confidence_split_stats(confidence, assign, src_inverse, dst_inverse):
    confidence = confidence.cpu().numpy()
    assign = assign.cpu().numpy()
    correct_conf = []
    wrong_conf = []
    for idx, conf in enumerate(confidence):
        if src_inverse[idx] == dst_inverse[int(assign[idx])]:
            correct_conf.append(float(conf))
        else:
            wrong_conf.append(float(conf))
    return {
        "correct_mean_conf": float(np.mean(correct_conf)) if correct_conf else 0.0,
        "wrong_mean_conf": float(np.mean(wrong_conf)) if wrong_conf else 0.0,
        "correct_count": len(correct_conf),
        "wrong_count": len(wrong_conf),
    }


def aligned_transport_stats(transport, rgb_inverse, ir_inverse, num_classes):
    rgb_order = sorted(range(num_classes), key=lambda x: rgb_inverse[x])
    ir_order = sorted(range(num_classes), key=lambda x: ir_inverse[x])
    aligned = transport[rgb_order][:, ir_order]
    diag = torch.diagonal(aligned)
    offdiag_sum = aligned.sum() - diag.sum()
    denom = max(aligned.numel() - diag.numel(), 1)
    return {
        "diag_mean": float(diag.mean().item()),
        "diag_std": float(diag.std().item()) if diag.numel() > 1 else 0.0,
        "offdiag_mean": float((offdiag_sum / denom).item()),
        "diag_offdiag_gap": float(diag.mean().item() - (offdiag_sum / denom).item()),
    }


def print_block(title, rows):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, value in rows:
        print(f"{key}: {value}")


def main():
    parser = get_parser()
    parser.add_argument("--verify-topk", default=3, type=int, help="Top-k used for offline BCPT hit-rate analysis")
    args = parser.parse_args()

    if args.model_path == "default":
        raise ValueError("Please provide a checkpoint with --model-path for BCPT verification.")

    args.num_classes = infer_num_classes(args)
    set_seed(args.seed)

    print_block(
        "BCPT Verification Setup",
        [
            ("dataset", args.dataset),
            ("arch", args.arch),
            ("trial", args.trial),
            ("test_mode", args.test_mode),
            ("checkpoint", args.model_path),
            ("bcpt_topk", args.bcpt_topk),
            ("verify_topk", args.verify_topk),
        ],
    )

    dataset = datasets.create(args)
    model = models.create(args)
    model.resume_model(args.model_path)
    cma = CMA(args)
    cma.extract(args, model, dataset)
    bcpt_r2i, bcpt_i2r = cma.get_label()

    legacy_r2i, _ = cma._get_label(cma.vis_scores.numpy(), "rgb")
    legacy_i2r, _ = cma._get_label(cma.ir_scores.numpy(), "ir")

    rgb_inverse = invert_relabel_dict(dataset.train_rgb.relabel_dict, args.num_classes)
    ir_inverse = invert_relabel_dict(dataset.train_ir.relabel_dict, args.num_classes)

    legacy_row_acc, legacy_row_correct, legacy_row_total = map_pair_accuracy(legacy_r2i, rgb_inverse, ir_inverse)
    legacy_col_acc, legacy_col_correct, legacy_col_total = map_pair_accuracy(legacy_i2r, ir_inverse, rgb_inverse)
    bcpt_row_acc, bcpt_row_correct, bcpt_row_total = map_pair_accuracy(bcpt_r2i, rgb_inverse, ir_inverse)
    bcpt_col_acc, bcpt_col_correct, bcpt_col_total = map_pair_accuracy(bcpt_i2r, ir_inverse, rgb_inverse)

    transport_row = cma.transport_row.cpu()
    transport_col = cma.transport_col.cpu()
    topk_row_acc, topk_row_hit, topk_row_total = topk_hit_rate(transport_row, rgb_inverse, ir_inverse, args.verify_topk)
    topk_col_acc, topk_col_hit, topk_col_total = topk_hit_rate(transport_col, ir_inverse, rgb_inverse, args.verify_topk)

    common_row_prec, common_row_correct, common_row_total = mask_precision(
        cma.common_row_mask, cma.row_assign, rgb_inverse, ir_inverse
    )
    specific_row_prec, specific_row_correct, specific_row_total = mask_precision(
        cma.specific_row_mask, cma.row_assign, rgb_inverse, ir_inverse
    )
    remain_row_prec, remain_row_correct, remain_row_total = mask_precision(
        cma.remain_row_mask, cma.row_assign, rgb_inverse, ir_inverse
    )

    common_col_prec, common_col_correct, common_col_total = mask_precision(
        cma.common_col_mask, cma.col_assign, ir_inverse, rgb_inverse
    )
    specific_col_prec, specific_col_correct, specific_col_total = mask_precision(
        cma.specific_col_mask, cma.col_assign, ir_inverse, rgb_inverse
    )
    remain_col_prec, remain_col_correct, remain_col_total = mask_precision(
        cma.remain_col_mask, cma.col_assign, ir_inverse, rgb_inverse
    )

    transport_stats = aligned_transport_stats(transport_row, rgb_inverse, ir_inverse, args.num_classes)
    row_conf_stats = confidence_split_stats(cma.row_conf, cma.row_assign, rgb_inverse, ir_inverse)
    col_conf_stats = confidence_split_stats(cma.col_conf, cma.col_assign, ir_inverse, rgb_inverse)

    print_block(
        "Pair Accuracy",
        [
            ("legacy_row_top1", f"{legacy_row_acc:.4f} ({legacy_row_correct}/{legacy_row_total})"),
            ("legacy_col_top1", f"{legacy_col_acc:.4f} ({legacy_col_correct}/{legacy_col_total})"),
            ("bcpt_row_top1", f"{bcpt_row_acc:.4f} ({bcpt_row_correct}/{bcpt_row_total})"),
            ("bcpt_col_top1", f"{bcpt_col_acc:.4f} ({bcpt_col_correct}/{bcpt_col_total})"),
            (f"bcpt_row_top{args.verify_topk}", f"{topk_row_acc:.4f} ({topk_row_hit}/{topk_row_total})"),
            (f"bcpt_col_top{args.verify_topk}", f"{topk_col_acc:.4f} ({topk_col_hit}/{topk_col_total})"),
        ],
    )

    print_block(
        "Transport Quality",
        [
            ("diag_mean", f"{transport_stats['diag_mean']:.6f}"),
            ("diag_std", f"{transport_stats['diag_std']:.6f}"),
            ("offdiag_mean", f"{transport_stats['offdiag_mean']:.6f}"),
            ("diag_offdiag_gap", f"{transport_stats['diag_offdiag_gap']:.6f}"),
        ],
    )

    print_block(
        "Route Precision",
        [
            ("common_row_precision", f"{common_row_prec:.4f} ({common_row_correct}/{common_row_total})"),
            ("specific_row_precision", f"{specific_row_prec:.4f} ({specific_row_correct}/{specific_row_total})"),
            ("remain_row_precision", f"{remain_row_prec:.4f} ({remain_row_correct}/{remain_row_total})"),
            ("common_col_precision", f"{common_col_prec:.4f} ({common_col_correct}/{common_col_total})"),
            ("specific_col_precision", f"{specific_col_prec:.4f} ({specific_col_correct}/{specific_col_total})"),
            ("remain_col_precision", f"{remain_col_prec:.4f} ({remain_col_correct}/{remain_col_total})"),
        ],
    )

    print_block(
        "Confidence Split",
        [
            ("row_correct_mean_conf", f"{row_conf_stats['correct_mean_conf']:.4f}"),
            ("row_wrong_mean_conf", f"{row_conf_stats['wrong_mean_conf']:.4f}"),
            ("col_correct_mean_conf", f"{col_conf_stats['correct_mean_conf']:.4f}"),
            ("col_wrong_mean_conf", f"{col_conf_stats['wrong_mean_conf']:.4f}"),
        ],
    )

    print("\nInterpretation:")
    print("- If `bcpt_row_top1` and `bcpt_col_top1` are not better than legacy, BCPT matching is not yet helping.")
    print("- If `diag_offdiag_gap` is small or negative, the transport matrix is not concentrating on GT-aligned pairs.")
    print("- If `common_*_precision` is low, phase2 common supervision is likely injecting noise.")
    print("- If `correct_mean_conf` is not noticeably higher than `wrong_mean_conf`, confidence weighting is weak.")


if __name__ == "__main__":
    main()
