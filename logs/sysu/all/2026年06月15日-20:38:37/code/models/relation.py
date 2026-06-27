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
def build_uprt_posterior(args, cma, common_dict):
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
    if common_dict:
        vis_ids = torch.as_tensor(
            list(common_dict.keys()), dtype=torch.long, device=similarity.device
        )
        ir_ids = torch.as_tensor(
            list(common_dict.values()), dtype=torch.long, device=similarity.device
        )
        prior[vis_ids, ir_ids] = 1.0

        pair_signals = torch.stack(
            (
                _rank_normalize(1.0 - 0.5 * (q_v2t[vis_ids, ir_ids] + q_t2v[ir_ids, vis_ids])),
                _rank_normalize(1.0 - (similarity[vis_ids, ir_ids] + 1.0) / 2.0),
                _rank_normalize(_selected_margin_uncertainty(q_v2t, vis_ids, ir_ids)),
                _rank_normalize(
                    _selected_margin_uncertainty(q_t2v, ir_ids, vis_ids)
                ),
                _rank_normalize(
                    _selected_margin_uncertainty(similarity, vis_ids, ir_ids)
                ),
                _rank_normalize(
                    _selected_margin_uncertainty(similarity.t(), ir_ids, vis_ids)
                ),
            )
        ).mean(dim=0)
        pair_uncertainty = 0.5 * (
            vis_uncertainty[vis_ids] + ir_uncertainty[ir_ids]
        )
        pair_uncertainty = 0.5 * (pair_uncertainty + pair_signals)
        vis_uncertainty[vis_ids] = torch.maximum(
            vis_uncertainty[vis_ids], pair_uncertainty
        )
        ir_uncertainty[ir_ids] = torch.maximum(
            ir_uncertainty[ir_ids], pair_uncertainty
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

    entropy = _normalized_entropy(v2t).mean()
    common_probability = (
        v2t[vis_ids, ir_ids].mean()
        if common_dict
        else transport.new_tensor(0.0)
    )
    stats = {
        "mean_mass": 0.5 * (row_mass.mean() + col_mass.mean()),
        "mean_reliability": 0.5 * (vis_reliability.mean() + ir_reliability.mean()),
        "posterior_entropy": entropy,
        "common_probability": common_probability,
        "candidate_ratio": mask.float().mean(),
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
