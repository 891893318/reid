import math

import torch
import torch.nn.functional as F


def _normalized_entropy(probabilities):
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / math.log(probabilities.shape[1])


def _top1_margin(probabilities):
    top2 = probabilities.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _js_divergence(first, second):
    midpoint = 0.5 * (first + second)
    first_kl = (
        first * (first.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
    ).sum(dim=1)
    second_kl = (
        second * (second.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
    ).sum(dim=1)
    return 0.5 * (first_kl + second_kl) / math.log(2.0)


def _rank_normalize(values):
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(
        values.numel(), device=values.device, dtype=values.dtype
    )
    return ranks / max(1, values.numel() - 1)


def _selected_margin_uncertainty(matrix, rows, cols):
    selected = matrix[rows, cols]
    masked = matrix.clone()
    masked[rows, cols] = float("-inf")
    alternative = masked[rows].max(dim=1).values
    return alternative - selected


def _identity_uncertainty(expert, shared):
    return torch.stack(
        (
            _rank_normalize(_normalized_entropy(expert)),
            _rank_normalize(1.0 - _top1_margin(expert)),
            _rank_normalize(_js_divergence(expert, shared)),
        )
    ).mean(dim=0)


def _candidate_mask(affinity, topk):
    topk = min(max(1, int(topk)), affinity.shape[1])
    mask = torch.zeros_like(affinity, dtype=torch.bool)
    row_indices = affinity.topk(topk, dim=1).indices
    mask.scatter_(1, row_indices, True)
    col_indices = affinity.topk(topk, dim=0).indices
    mask.scatter_(0, col_indices, True)
    return mask


def _masked_softmax(logits, mask, temperature):
    temperature = max(float(temperature), 1e-6)
    masked_logits = logits / temperature
    masked_logits = masked_logits.masked_fill(~mask, float("-inf"))
    probabilities = F.softmax(masked_logits, dim=1)
    return probabilities.masked_fill(~mask, 0.0)


def _unbalanced_transport(affinity, mask, source_mass, target_mass, epsilon, tau, iters):
    epsilon = max(float(epsilon), 1e-6)
    tau = max(float(tau), 1e-6)
    rho = tau / (tau + epsilon)
    kernel = torch.exp((affinity - affinity.max()) / epsilon) * mask.float()
    left = torch.ones_like(source_mass)
    right = torch.ones_like(target_mass)
    for _ in range(max(1, int(iters))):
        left = (source_mass / torch.matmul(kernel, right).clamp_min(1e-12)).pow(rho)
        right = (target_mass / torch.matmul(kernel.t(), left).clamp_min(1e-12)).pow(rho)
    return left.unsqueeze(1) * kernel * right.unsqueeze(0)


@torch.no_grad()
def build_uprt_posterior(
    args, cma, common_dict, specific_dict=None, remain_dict=None, epoch=None
):
    vis_memory = F.normalize(cma.uprt_vis_memory.detach(), dim=1)
    ir_memory = F.normalize(cma.uprt_ir_memory.detach(), dim=1)
    q_v2t = cma.uprt_v2t.detach().clamp_min(1e-12)
    q_t2v = cma.uprt_t2v.detach().clamp_min(1e-12)

    similarity = torch.matmul(vis_memory, ir_memory.t())
    shared_temp = max(getattr(args, "uprt_shared_temperature", 0.07), 1e-6)
    s_v2t = F.softmax(similarity / shared_temp, dim=1)
    s_t2v = F.softmax(similarity.t() / shared_temp, dim=1)

    vis_uncertainty = _identity_uncertainty(q_v2t, s_v2t)
    ir_uncertainty = _identity_uncertainty(q_t2v, s_t2v)

    prior = torch.zeros_like(similarity)
    common_vis_ids = None
    common_ir_ids = None
    if common_dict:
        common_vis_ids = torch.as_tensor(
            list(common_dict.keys()), dtype=torch.long, device=similarity.device
        )
        common_ir_ids = torch.as_tensor(
            list(common_dict.values()), dtype=torch.long, device=similarity.device
        )
        prior[common_vis_ids, common_ir_ids] = 1.0

        pair_signals = torch.stack(
            (
                _rank_normalize(
                    1.0 - 0.5 * (
                        q_v2t[common_vis_ids, common_ir_ids]
                        + q_t2v[common_ir_ids, common_vis_ids]
                    )
                ),
                _rank_normalize(1.0 - (similarity[common_vis_ids, common_ir_ids] + 1.0) / 2.0),
                _rank_normalize(_selected_margin_uncertainty(q_v2t, common_vis_ids, common_ir_ids)),
                _rank_normalize(
                    _selected_margin_uncertainty(q_t2v, common_ir_ids, common_vis_ids)
                ),
                _rank_normalize(
                    _selected_margin_uncertainty(similarity, common_vis_ids, common_ir_ids)
                ),
                _rank_normalize(
                    _selected_margin_uncertainty(similarity.t(), common_ir_ids, common_vis_ids)
                ),
            )
        ).mean(dim=0)
        pair_uncertainty = 0.5 * (
            vis_uncertainty[common_vis_ids] + ir_uncertainty[common_ir_ids]
        )
        pair_uncertainty = 0.5 * (pair_uncertainty + pair_signals)
        vis_uncertainty[common_vis_ids] = torch.maximum(
            vis_uncertainty[common_vis_ids], pair_uncertainty
        )
        ir_uncertainty[common_ir_ids] = torch.maximum(
            ir_uncertainty[common_ir_ids], pair_uncertainty
        )

    for pair_dict, weight in (
        (specific_dict, getattr(args, "uprt_specific_prior", 0.5)),
        (remain_dict, getattr(args, "uprt_remain_prior", 0.2)),
    ):
        if pair_dict and weight > 0:
            vis_ids = torch.as_tensor(
                list(pair_dict.keys()), dtype=torch.long, device=similarity.device
            )
            ir_ids = torch.as_tensor(
                list(pair_dict.values()), dtype=torch.long, device=similarity.device
            )
            prior[vis_ids, ir_ids] = torch.maximum(
                prior[vis_ids, ir_ids],
                prior.new_full((vis_ids.numel(),), float(weight)),
            )

    min_mass = min(max(getattr(args, "uprt_min_mass", 0.05), 0.0), 1.0)
    vis_reliability = min_mass + (1.0 - min_mass) * (1.0 - vis_uncertainty)
    ir_reliability = min_mass + (1.0 - min_mass) * (1.0 - ir_uncertainty)
    source_mass = vis_reliability / similarity.shape[0]
    target_mass = ir_reliability / similarity.shape[1]

    expert_affinity = 0.5 * (q_v2t + q_t2v.t())
    affinity = (
        getattr(args, "uprt_shared_weight", 1.0) * similarity
        + getattr(args, "uprt_expert_weight", 0.1) * expert_affinity
        + getattr(args, "uprt_prior_weight", 0.05) * prior
    )
    mask = _candidate_mask(affinity, getattr(args, "uprt_topk", 10))
    mask |= prior.bool()
    recovery_topk = max(
        getattr(args, "uprt_topk", 10),
        getattr(args, "uprt_recovery_topk", getattr(args, "uprt_topk", 10)),
    )
    recovery_mask = _candidate_mask(affinity, recovery_topk)
    recovery_mask |= mask

    transport = _unbalanced_transport(
        affinity,
        mask,
        source_mass,
        target_mass,
        getattr(args, "uprt_epsilon", 0.1),
        getattr(args, "uprt_tau", 0.5),
        getattr(args, "uprt_iters", 30),
    )
    row_mass = transport.sum(dim=1)
    col_mass = transport.sum(dim=0)
    v2t = transport / row_mass.clamp_min(1e-12).unsqueeze(1)
    t2v = transport.t() / col_mass.clamp_min(1e-12).unsqueeze(1)

    raw_entropy = _normalized_entropy(v2t).mean()
    common_coverage = len(common_dict or {}) / max(1, similarity.shape[0])
    recovery_weight = 0.0
    if epoch is not None:
        recovery_start = getattr(args, "uprt_recovery_start_epoch", 40)
        if epoch >= recovery_start:
            target_coverage = getattr(args, "uprt_recovery_target_coverage", 0.95)
            min_coverage = getattr(args, "uprt_recovery_min_coverage", 0.8)
            coverage_span = max(1e-6, target_coverage - min_coverage)
            coverage_gap = max(0.0, min(1.0, (target_coverage - common_coverage) / coverage_span))
            min_entropy = getattr(args, "uprt_recovery_min_entropy", 0.08)
            entropy_gap = 0.0
            if min_entropy > 0:
                entropy_gap = max(
                    0.0,
                    min(1.0, (min_entropy - raw_entropy.item()) / max(min_entropy, 1e-6)),
                )
            warmup = max(1, getattr(args, "uprt_recovery_warmup_epochs", 20))
            epoch_scale = min(1.0, (epoch - recovery_start + 1) / warmup)
            recovery_weight = (
                getattr(args, "uprt_recovery_weight", 0.08)
                * epoch_scale
                * max(coverage_gap, entropy_gap)
            )

    if recovery_weight > 0:
        recovery_temp = getattr(args, "uprt_recovery_temperature", 0.2)
        recovery_v2t = _masked_softmax(affinity, recovery_mask, recovery_temp)
        recovery_t2v = _masked_softmax(affinity.t(), recovery_mask.t(), recovery_temp)
        v2t = (1.0 - recovery_weight) * v2t + recovery_weight * recovery_v2t
        t2v = (1.0 - recovery_weight) * t2v + recovery_weight * recovery_t2v

    entropy = _normalized_entropy(v2t).mean()
    common_probability = (
        v2t[common_vis_ids, common_ir_ids].mean()
        if common_vis_ids is not None
        else transport.new_tensor(0.0)
    )
    stats = {
        "mean_mass": 0.5 * (row_mass.mean() + col_mass.mean()),
        "mean_reliability": 0.5 * (vis_reliability.mean() + ir_reliability.mean()),
        "posterior_entropy": entropy,
        "raw_posterior_entropy": raw_entropy,
        "common_probability": common_probability,
        "common_coverage": transport.new_tensor(common_coverage),
        "candidate_ratio": mask.float().mean(),
        "recovery_candidate_ratio": recovery_mask.float().mean(),
        "recovery_weight": transport.new_tensor(recovery_weight),
        "prior_ratio": prior.bool().float().mean(),
    }
    return {
        "v2t": v2t,
        "t2v": t2v,
        "vis_mass": row_mass,
        "ir_mass": col_mass,
        "vis_reliability": vis_reliability,
        "ir_reliability": ir_reliability,
        "stats": stats,
    }


def posterior_cross_modal_loss(
    anchors, labels, target_memory, posterior, transported_mass, temperature
):
    if anchors.numel() == 0:
        return None
    anchors = F.normalize(anchors, dim=1)
    target_memory = F.normalize(target_memory.detach(), dim=1)
    logits = torch.matmul(anchors, target_memory.t()) / max(float(temperature), 1e-6)
    targets = posterior[labels].detach()
    weights = transported_mass[labels].detach()
    weights = weights / transported_mass.max().clamp_min(1e-12)
    per_sample = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    return (weights * per_sample).sum() / weights.sum().clamp_min(1e-12)


def posterior_classifier_loss(
    logits, labels, posterior, transported_mass, temperature=1.0
):
    if logits.numel() == 0:
        return None
    targets = posterior[labels].detach()
    weights = transported_mass[labels].detach()
    weights = weights / transported_mass.max().clamp_min(1e-12)
    log_probs = F.log_softmax(logits / max(float(temperature), 1e-6), dim=1)
    per_sample = -(targets * log_probs).sum(dim=1)
    return (weights * per_sample).sum() / weights.sum().clamp_min(1e-12)
