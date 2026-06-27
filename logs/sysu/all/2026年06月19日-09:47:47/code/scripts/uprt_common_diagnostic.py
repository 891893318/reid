#!/usr/bin/env python3
"""Offline feasibility diagnostic for UPRT common-relation uncertainty."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets  # noqa: E402
import models  # noqa: E402
from models.relation import build_uprt_posterior  # noqa: E402
from utils import set_seed  # noqa: E402
from wsl import CMA  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("UPRT common-relation uncertainty diagnostic")
    parser.add_argument(
        "--checkpoint",
        default="logs/sysu/all/innovation1/models/stage2/model_99.pth",
    )
    parser.add_argument(
        "--output",
        default="docx/innovation1_all_uprt_common_diagnostic.json",
    )
    parser.add_argument("--data-path", default="/root/data/")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--test-batch", default=128, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--expert-temperature", default=3.0, type=float)
    parser.add_argument("--shared-temperature", default=0.07, type=float)
    return parser.parse_args()


def build_runtime_args(cli):
    device = f"cuda:{cli.device}" if torch.cuda.is_available() else "cpu"
    return SimpleNamespace(
        dataset="sysu",
        arch="resnet",
        mode="test",
        data_path=cli.data_path,
        save_path="/tmp/uprt_common_diagnostic",
        lr=0.0003,
        weight_decay=0.0005,
        milestones=[30, 70],
        relabel=1,
        tri_weight=0.25,
        img_h=288,
        img_w=144,
        seed=cli.seed,
        num_workers=cli.num_workers,
        batch_pidnum=8,
        pid_numsample=4,
        test_batch=cli.test_batch,
        sigma=0.8,
        temperature=cli.expert_temperature,
        enable_rgmfd=1,
        rgmfd_reduction=16,
        rgmfd_gate_scale=0.5,
        resume=0,
        debug="wsl",
        search_mode="all",
        gall_mode="single",
        test_mode="t2v",
        num_classes=395,
        device=device,
        cre_confidence=0,
        cre_sample_rate=1.0,
        cre_count_weight=0.0,
        cre_prob_weight=0.0,
        cre_margin_weight=0.0,
        cre_entropy_weight=0.0,
        cre_proto_weight=0.0,
        cre_min_margin_start=0.0,
        cre_min_margin_end=0.0,
        cre_margin_decay_epoch=1,
    )


def _model_features(model, images, modal):
    if modal == "rgb":
        _, shared_bn = model.model(x1=images)
    else:
        _, shared_bn = model.model(x2=images)
    return shared_bn


@torch.no_grad()
def extract_modality(model, loader, modal):
    labels_all = []
    gts_all = []
    shared_all = []
    expert_logits_all = []
    legacy_logits_all = []

    for images, infos in loader:
        images = images.to(model.device)
        labels = infos[:, 1].long()
        gts = infos[:, -1].long()

        shared_bn = _model_features(model, images, modal)
        if modal == "rgb":
            expert_logits, _ = model.classifier2(shared_bn)
            legacy_logits = expert_logits
        else:
            expert_logits, _ = model.classifier1(shared_bn)
            # Reproduce the existing CRE path, which passes IR as positional x1.
            _, legacy_bn = model.model(images)
            legacy_logits, _ = model.classifier1(legacy_bn)

        labels_all.append(labels.cpu())
        gts_all.append(gts.cpu())
        shared_all.append(shared_bn.cpu())
        expert_logits_all.append(expert_logits.cpu())
        legacy_logits_all.append(legacy_logits.cpu())

    return {
        "labels": torch.cat(labels_all),
        "gts": torch.cat(gts_all),
        "shared": torch.cat(shared_all),
        "expert_logits": torch.cat(expert_logits_all),
        "legacy_logits": torch.cat(legacy_logits_all),
    }


def build_relations(r2i_pair_dict, i2r_pair_dict):
    common, specific, remain = {}, {}, {}
    for rgb, ir in r2i_pair_dict.items():
        if ir in i2r_pair_dict and i2r_pair_dict[ir] == rgb:
            common[int(rgb)] = int(ir)
        elif rgb not in i2r_pair_dict.values() and ir not in i2r_pair_dict:
            specific[int(rgb)] = int(ir)
        else:
            remain[int(rgb)] = int(ir)
    for ir, rgb in i2r_pair_dict.items():
        if common.get(rgb) == ir:
            continue
        if rgb not in r2i_pair_dict.values() and ir not in r2i_pair_dict:
            specific[int(rgb)] = int(ir)
        else:
            remain[int(rgb)] = int(ir)
    return common, specific, remain


def class_average(values, labels, num_classes):
    result = values.new_zeros((num_classes, values.shape[1]))
    counts = values.new_zeros(num_classes)
    result.index_add_(0, labels, values)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=values.dtype))
    return result / counts.clamp_min(1).unsqueeze(1)


def label_to_gt(labels, gts):
    return {int(label): int(gt) for label, gt in zip(labels.tolist(), gts.tolist())}


def normalized_entropy(probabilities):
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / math.log(probabilities.shape[1])


def top1_margin(probabilities):
    top2 = probabilities.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def js_divergence(first, second):
    midpoint = 0.5 * (first + second)
    first_kl = (first * (first.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=1)
    second_kl = (second * (second.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=1)
    return 0.5 * (first_kl + second_kl) / math.log(2.0)


def selected_margin_uncertainty(matrix, rows, cols):
    selected = matrix[rows, cols]
    masked = matrix.clone()
    masked[rows, cols] = float("-inf")
    strongest_alternative = masked[rows].max(dim=1).values
    return strongest_alternative - selected


def rank_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def score_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def top_uncertain_metrics(labels, scores, fractions=(0.05, 0.10, 0.20)):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    wrong_total = max(1, int(labels.sum()))
    result = {}
    for fraction in fractions:
        count = max(1, int(math.ceil(len(labels) * fraction)))
        selected = labels[order[:count]]
        result[f"{int(fraction * 100)}pct"] = {
            "count": count,
            "wrong_count": int(selected.sum()),
            "wrong_ratio": float(selected.mean()),
            "wrong_recall": float(selected.sum() / wrong_total),
        }
    return result


def rejection_metrics(labels, scores, fractions=(0.05, 0.10, 0.20)):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    wrong_total = max(1, int(labels.sum()))
    correct_total = max(1, int((labels == 0).sum()))
    result = {}
    for fraction in fractions:
        count = max(1, int(math.ceil(len(labels) * fraction)))
        rejected = labels[order[:count]]
        retained = labels[order[count:]]
        result[f"{int(fraction * 100)}pct"] = {
            "rejected_count": count,
            "wrong_rejected": int(rejected.sum()),
            "wrong_rejection_recall": float(rejected.sum() / wrong_total),
            "correct_retention": float((correct_total - (rejected == 0).sum()) / correct_total),
            "retained_count": int(len(retained)),
            "retained_accuracy": float(1.0 - retained.mean()) if len(retained) else 1.0,
        }
    return result


def correction_topk_metrics(matrix, wrong_indices, rgb_ids, rgb_gt, ir_gt, topks=(1, 3, 5, 10)):
    ranked_candidates = matrix.argsort(dim=1, descending=True)
    result = {}
    for topk in topks:
        correctable = 0
        for index in wrong_indices:
            rgb_id = int(rgb_ids[index])
            target_gt = rgb_gt[rgb_id]
            candidates = ranked_candidates[rgb_id, :topk].tolist()
            correctable += int(any(ir_gt[int(candidate)] == target_gt for candidate in candidates))
        result[f"top{topk}"] = {
            "correctable_count": int(correctable),
            "correctable_ratio": float(correctable / max(1, len(wrong_indices))),
        }
    return result


def to_float_dict(values, index):
    return {name: float(value[index]) for name, value in values.items()}


def evaluate_transport_config(config, cma, common, rgb_ids, ir_ids, wrong, rgb_gt, ir_gt):
    args = SimpleNamespace(
        uprt_shared_temperature=0.07,
        uprt_min_mass=0.05,
        uprt_shared_weight=1.0,
        uprt_expert_weight=config["expert_weight"],
        uprt_prior_weight=config["prior_weight"],
        uprt_topk=10,
        uprt_epsilon=config["epsilon"],
        uprt_tau=0.5,
        uprt_iters=30,
    )
    posterior = build_uprt_posterior(args, cma, common)
    selected_probability = posterior["v2t"][rgb_ids, ir_ids].cpu().numpy()
    pair_reliability = (
        0.5
        * (
            posterior["vis_reliability"][rgb_ids]
            + posterior["ir_reliability"][ir_ids]
        )
    ).cpu().numpy()
    pair_mass = (
        0.5 * (posterior["vis_mass"][rgb_ids] + posterior["ir_mass"][ir_ids])
    ).cpu().numpy()

    ir_by_gt = {gt: label for label, gt in ir_gt.items()}
    true_ir_ids = torch.as_tensor(
        [ir_by_gt[rgb_gt[int(rgb_id)]] for rgb_id in rgb_ids.tolist()],
        dtype=torch.long,
        device=posterior["v2t"].device,
    )
    true_probability = posterior["v2t"][rgb_ids, true_ir_ids].cpu().numpy()
    uncertainty = 1.0 - pair_reliability
    metrics = score_metrics(wrong, uncertainty)
    return {
        **config,
        "wrong_uncertainty_roc_auc": metrics["roc_auc"],
        "posterior_entropy": float(posterior["stats"]["posterior_entropy"]),
        "correct_selected_probability": float(selected_probability[wrong == 0].mean()),
        "wrong_selected_probability": float(selected_probability[wrong == 1].mean()),
        "wrong_true_probability": float(true_probability[wrong == 1].mean()),
        "correct_pair_reliability": float(pair_reliability[wrong == 0].mean()),
        "wrong_pair_reliability": float(pair_reliability[wrong == 1].mean()),
        "correct_pair_mass": float(pair_mass[wrong == 0].mean()),
        "wrong_pair_mass": float(pair_mass[wrong == 1].mean()),
    }


def main():
    cli = parse_args()
    args = build_runtime_args(cli)
    set_seed(cli.seed)

    print(f"Loading SYSU and checkpoint: {cli.checkpoint}")
    dataset = datasets.create(args)
    model = models.create(args)
    model.resume_model(cli.checkpoint)
    model.set_eval()

    rgb_loader, ir_loader = dataset.get_normal_loader()
    print("Extracting RGB evidence")
    rgb = extract_modality(model, rgb_loader, "rgb")
    print("Extracting IR evidence")
    ir = extract_modality(model, ir_loader, "ir")

    cma = CMA(args)
    cma.save(
        rgb["legacy_logits"],
        ir["legacy_logits"],
        rgb["labels"],
        ir["labels"],
        torch.arange(len(rgb["labels"])),
        torch.arange(len(ir["labels"])),
        "scores",
        rgb_gt=rgb["gts"],
        ir_gt=ir["gts"],
    )
    r2i, i2r = cma.get_label()
    common, specific, remain = build_relations(r2i, i2r)

    expert_temp = cli.expert_temperature
    q_v2t = class_average(
        F.softmax(expert_temp * rgb["expert_logits"], dim=1),
        rgb["labels"],
        args.num_classes,
    )
    q_t2v = class_average(
        F.softmax(expert_temp * ir["expert_logits"], dim=1),
        ir["labels"],
        args.num_classes,
    )
    p_v = F.normalize(class_average(rgb["shared"], rgb["labels"], args.num_classes), dim=1)
    p_t = F.normalize(class_average(ir["shared"], ir["labels"], args.num_classes), dim=1)
    shared_similarity = torch.matmul(p_v, p_t.t())
    s_v2t = F.softmax(shared_similarity / cli.shared_temperature, dim=1)
    s_t2v = F.softmax(shared_similarity.t() / cli.shared_temperature, dim=1)

    rgb_ids = torch.tensor(list(common.keys()), dtype=torch.long)
    ir_ids = torch.tensor(list(common.values()), dtype=torch.long)
    rgb_gt = label_to_gt(rgb["labels"], rgb["gts"])
    ir_gt = label_to_gt(ir["labels"], ir["gts"])
    wrong = np.asarray(
        [int(rgb_gt[int(rgb_id)] != ir_gt[int(ir_id)]) for rgb_id, ir_id in zip(rgb_ids, ir_ids)],
        dtype=np.int64,
    )

    signals = {
        "expert_entropy": (
            0.5 * (normalized_entropy(q_v2t)[rgb_ids] + normalized_entropy(q_t2v)[ir_ids])
        ).numpy(),
        "expert_margin_uncertainty": (
            1.0 - 0.5 * (top1_margin(q_v2t)[rgb_ids] + top1_margin(q_t2v)[ir_ids])
        ).numpy(),
        "shared_entropy": (
            0.5 * (normalized_entropy(s_v2t)[rgb_ids] + normalized_entropy(s_t2v)[ir_ids])
        ).numpy(),
        "distribution_conflict": (
            0.5 * (js_divergence(q_v2t, s_v2t)[rgb_ids] + js_divergence(q_t2v, s_t2v)[ir_ids])
        ).numpy(),
        "pair_expert_uncertainty": (
            1.0 - 0.5 * (q_v2t[rgb_ids, ir_ids] + q_t2v[ir_ids, rgb_ids])
        ).numpy(),
        "pair_shared_uncertainty": (
            1.0 - (shared_similarity[rgb_ids, ir_ids] + 1.0) / 2.0
        ).numpy(),
        "selected_expert_margin_uncertainty": (
            0.5
            * (
                selected_margin_uncertainty(q_v2t, rgb_ids, ir_ids)
                + selected_margin_uncertainty(q_t2v, ir_ids, rgb_ids)
            )
        ).numpy(),
        "selected_shared_margin_uncertainty": (
            0.5
            * (
                selected_margin_uncertainty(shared_similarity, rgb_ids, ir_ids)
                + selected_margin_uncertainty(shared_similarity.t(), ir_ids, rgb_ids)
            )
        ).numpy(),
    }

    ranked = {name: rank_normalize(values) for name, values in signals.items()}
    composites = {
        "identity_uncertainty": np.mean(
            [ranked["expert_entropy"], ranked["expert_margin_uncertainty"], ranked["distribution_conflict"]],
            axis=0,
        ),
        "edge_uncertainty": np.mean(
            [
                ranked["pair_expert_uncertainty"],
                ranked["pair_shared_uncertainty"],
                ranked["selected_expert_margin_uncertainty"],
                ranked["selected_shared_margin_uncertainty"],
            ],
            axis=0,
        ),
    }
    composites["full_uncertainty"] = np.mean(
        [composites["identity_uncertainty"], composites["edge_uncertainty"]],
        axis=0,
    )
    all_scores = {**signals, **composites}

    score_summary = {
        name: {
            **score_metrics(wrong, values),
            "correct_mean": float(np.asarray(values)[wrong == 0].mean()),
            "wrong_mean": float(np.asarray(values)[wrong == 1].mean()),
        }
        for name, values in all_scores.items()
    }

    top_summary = {
        name: top_uncertain_metrics(wrong, composites[name])
        for name in ("identity_uncertainty", "edge_uncertainty", "full_uncertainty")
    }
    rejection_summary = {
        name: rejection_metrics(wrong, composites[name])
        for name in ("identity_uncertainty", "edge_uncertainty", "full_uncertainty")
    }

    wrong_indices = np.flatnonzero(wrong == 1)
    fused_affinity = 0.5 * (q_v2t + q_t2v.t()) + s_v2t
    correction_potential = {
        "shared": correction_topk_metrics(
            shared_similarity, wrong_indices, rgb_ids, rgb_gt, ir_gt
        ),
        "fused": correction_topk_metrics(
            fused_affinity, wrong_indices, rgb_ids, rgb_gt, ir_gt
        ),
    }
    transport_cma = SimpleNamespace(
        uprt_vis_memory=p_v.to(model.device),
        uprt_ir_memory=p_t.to(model.device),
        uprt_v2t=q_v2t.to(model.device),
        uprt_t2v=q_t2v.to(model.device),
    )
    transport_configs = [
        {"name": "current", "epsilon": 0.05, "expert_weight": 0.5, "prior_weight": 0.1},
        {"name": "soft_eps_010", "epsilon": 0.10, "expert_weight": 0.5, "prior_weight": 0.1},
        {"name": "soft_eps_020", "epsilon": 0.20, "expert_weight": 0.5, "prior_weight": 0.1},
        {"name": "weak_prior_010", "epsilon": 0.10, "expert_weight": 0.1, "prior_weight": 0.05},
        {"name": "weak_prior_020", "epsilon": 0.20, "expert_weight": 0.1, "prior_weight": 0.05},
        {"name": "shared_only_020", "epsilon": 0.20, "expert_weight": 0.0, "prior_weight": 0.0},
    ]
    transport_sweep = [
        evaluate_transport_config(
            config,
            transport_cma,
            common,
            rgb_ids.to(model.device),
            ir_ids.to(model.device),
            wrong,
            rgb_gt,
            ir_gt,
        )
        for config in transport_configs
    ]

    pair_records = []
    for index, (rgb_id, ir_id) in enumerate(zip(rgb_ids.tolist(), ir_ids.tolist())):
        pair_records.append(
            {
                "rgb_id": rgb_id,
                "ir_id": ir_id,
                "rgb_gt": rgb_gt[rgb_id],
                "ir_gt": ir_gt[ir_id],
                "wrong": int(wrong[index]),
                "signals": to_float_dict(signals, index),
                "composites": to_float_dict(composites, index),
            }
        )
    pair_records.sort(key=lambda item: item["composites"]["full_uncertainty"], reverse=True)

    result = {
        "checkpoint": cli.checkpoint,
        "dataset": "sysu/all",
        "seed": cli.seed,
        "cre_extraction": "existing path; IR is passed as positional x1",
        "uprt_evidence_extraction": "RGB x1 and IR x2 with RG-MFD shared features",
        "relation_counts": {
            "r2i": len(r2i),
            "i2r": len(i2r),
            "common": len(common),
            "specific": len(specific),
            "remain": len(remain),
        },
        "common_quality": {
            "correct": int((wrong == 0).sum()),
            "wrong": int(wrong.sum()),
            "accuracy": float(1.0 - wrong.mean()),
            "wrong_ratio": float(wrong.mean()),
        },
        "score_summary": score_summary,
        "top_uncertain_summary": top_summary,
        "rejection_summary": rejection_summary,
        "wrong_common_correction_potential": correction_potential,
        "transport_sweep": transport_sweep,
        "pairs_by_full_uncertainty": pair_records,
    }

    output_path = ROOT / cli.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")

    print(json.dumps({
        "common_quality": result["common_quality"],
        "composite_metrics": {name: score_summary[name] for name in composites},
        "top_uncertain_summary": top_summary["full_uncertainty"],
        "rejection_summary": rejection_summary["full_uncertainty"],
        "wrong_common_correction_potential": correction_potential,
        "transport_sweep": transport_sweep,
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
