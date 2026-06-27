import torch
import torch.nn as nn
import torch.nn.functional as F


class P2RNet(nn.Module):
    """
    Prototype-to-Relation network.

    The module converts CRE candidate pairs into a differentiable partial
    relation distribution. A dustbin row/column lets uncertain identities stay
    unmatched instead of being forced into a wrong cross-modal pair.
    """

    def __init__(
        self,
        feature_dim=2048,
        hidden_dim=512,
        temperature=0.07,
        sinkhorn_iters=20,
        anchor_bias=4.0,
        candidate_bias=1.0,
        mask_penalty=8.0,
        dustbin_logit=0.0,
        uncertainty_scale=1.0,
    ):
        super(P2RNet, self).__init__()
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.anchor_bias = anchor_bias
        self.candidate_bias = candidate_bias
        self.mask_penalty = mask_penalty
        self.dustbin_logit = dustbin_logit
        self.uncertainty_scale = uncertainty_scale

        self.rgb_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ir_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rgb_uncertainty = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.ir_uncertainty = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.rgb_value = nn.Linear(feature_dim, hidden_dim)
        self.ir_value = nn.Linear(feature_dim, hidden_dim)
        self.rgb_refine = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.ir_refine = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def _log_sinkhorn(self, logits, row_marginals, col_marginals):
        log_transport = logits
        log_row = row_marginals.clamp_min(1e-8).log().unsqueeze(1)
        log_col = col_marginals.clamp_min(1e-8).log().unsqueeze(0)
        for _ in range(max(1, self.sinkhorn_iters)):
            log_transport = log_transport + log_row - torch.logsumexp(log_transport, dim=1, keepdim=True)
            log_transport = log_transport + log_col - torch.logsumexp(log_transport, dim=0, keepdim=True)
        return log_transport.exp()

    def forward(self, rgb_proto, ir_proto, anchor_mask=None, candidate_mask=None):
        rgb_proto = rgb_proto.float()
        ir_proto = ir_proto.float()
        num_rgb, num_ir = rgb_proto.shape[0], ir_proto.shape[0]
        device = rgb_proto.device

        rgb_embed = F.normalize(self.rgb_proj(rgb_proto), dim=1)
        ir_embed = F.normalize(self.ir_proj(ir_proto), dim=1)
        temp = max(float(self.temperature), 1e-6)
        logits = torch.matmul(rgb_embed, ir_embed.t()) / temp

        rgb_unc = self.rgb_uncertainty(rgb_proto)
        ir_unc = self.ir_uncertainty(ir_proto).t()
        edge_uncertainty = 0.5 * (rgb_unc + ir_unc)
        logits = logits - self.uncertainty_scale * edge_uncertainty

        if candidate_mask is None:
            candidate_mask = torch.ones_like(logits, dtype=torch.bool, device=device)
        else:
            candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
        if anchor_mask is None:
            anchor_mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
        else:
            anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)

        logits = logits + candidate_mask.float() * self.candidate_bias
        logits = logits - (~candidate_mask).float() * self.mask_penalty
        logits = logits + anchor_mask.float() * self.anchor_bias

        dustbin = logits.new_full((num_rgb + 1, num_ir + 1), self.dustbin_logit)
        dustbin[:num_rgb, :num_ir] = logits
        row_marginals = logits.new_ones(num_rgb + 1)
        col_marginals = logits.new_ones(num_ir + 1)
        row_marginals[-1] = float(num_ir)
        col_marginals[-1] = float(num_rgb)
        transport = self._log_sinkhorn(dustbin, row_marginals, col_marginals)
        relation = transport[:num_rgb, :num_ir]

        row_mass = relation.sum(dim=1, keepdim=True).clamp_min(1e-6)
        col_mass = relation.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
        rgb_context = torch.matmul(relation, self.ir_value(ir_proto)) / row_mass
        ir_context = torch.matmul(relation.t(), self.rgb_value(rgb_proto)) / col_mass
        rgb_refined = self.rgb_refine(torch.cat([rgb_proto, rgb_context], dim=1))
        ir_refined = self.ir_refine(torch.cat([ir_proto, ir_context], dim=1))

        with torch.no_grad():
            row_norm = relation / relation.sum(dim=1, keepdim=True).clamp_min(1e-6)
            col_norm = relation.t() / relation.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
            row_entropy = -(row_norm * row_norm.clamp_min(1e-12).log()).sum(dim=1)
            col_entropy = -(col_norm * col_norm.clamp_min(1e-12).log()).sum(dim=1)
            norm = torch.log(relation.new_tensor(max(2, num_ir)))
            entropy = 0.5 * (row_entropy.mean() + col_entropy.mean()) / norm
            diag = {
                "row_conf": row_norm.max(dim=1).values.mean(),
                "col_conf": col_norm.max(dim=1).values.mean(),
                "entropy": entropy,
                "row_dustbin": transport[:num_rgb, num_ir].mean(),
                "col_dustbin": transport[num_rgb, :num_ir].mean(),
                "anchor_prob": relation[anchor_mask].mean() if anchor_mask.any() else relation.new_tensor(0.0),
                "candidate_mass": relation[candidate_mask].mean() if candidate_mask.any() else relation.new_tensor(0.0),
            }

        return {
            "relation": relation,
            "transport": transport,
            "logits": logits,
            "uncertainty": edge_uncertainty,
            "rgb_refined": rgb_refined,
            "ir_refined": ir_refined,
            "diag": diag,
        }
