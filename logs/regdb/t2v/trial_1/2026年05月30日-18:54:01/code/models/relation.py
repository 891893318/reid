import torch
import torch.nn as nn
import torch.nn.functional as F


class UOTCrossModalRelationTransformer(nn.Module):
    """
    Uncertainty-aware optimal-transport cross-modal relation module.

    It replaces a purely hard CRE relation view with a global soft transport
    plan and relation-refined modality prototypes.
    """

    def __init__(
        self,
        feature_dim=2048,
        hidden_dim=512,
        temperature=0.07,
        sinkhorn_iters=30,
        residual_scale=0.2,
    ):
        super(UOTCrossModalRelationTransformer, self).__init__()
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.residual_scale = residual_scale

        self.rgb_q = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.ir_k = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.rgb_v = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.ir_v = nn.Linear(feature_dim, hidden_dim, bias=False)
        uncertainty_dim = max(1, hidden_dim // 2)

        self.rgb_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, uncertainty_dim),
            nn.ReLU(inplace=True),
            nn.Linear(uncertainty_dim, 1),
        )
        self.ir_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, uncertainty_dim),
            nn.ReLU(inplace=True),
            nn.Linear(uncertainty_dim, 1),
        )

        self.rgb_out = nn.Linear(hidden_dim, feature_dim, bias=False)
        self.ir_out = nn.Linear(hidden_dim, feature_dim, bias=False)
        self.rgb_norm = nn.LayerNorm(feature_dim)
        self.ir_norm = nn.LayerNorm(feature_dim)

    def _sinkhorn(self, logits):
        transport = torch.exp(logits - logits.max())
        transport = transport.clamp_min(1e-12)
        for _ in range(self.sinkhorn_iters):
            transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-12)
            transport = transport / transport.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return transport

    def forward(self, rgb_memory, ir_memory):
        rgb_memory = rgb_memory.float()
        ir_memory = ir_memory.float()

        rgb_base = F.normalize(rgb_memory, dim=1)
        ir_base = F.normalize(ir_memory, dim=1)
        rgb_q = F.normalize(self.rgb_q(rgb_base), dim=1)
        ir_k = F.normalize(self.ir_k(ir_base), dim=1)
        rgb_v = self.rgb_v(rgb_base)
        ir_v = self.ir_v(ir_base)

        rgb_unc = torch.sigmoid(self.rgb_uncertainty(rgb_q))
        ir_unc = torch.sigmoid(self.ir_uncertainty(ir_k)).t()
        reliability = 1.0 - 0.5 * (rgb_unc + ir_unc)

        logits = torch.matmul(rgb_q, ir_k.t()) / max(self.temperature, 1e-6)
        logits = logits + reliability.clamp_min(1e-6).log()
        transport = self._sinkhorn(logits)

        rgb_context = torch.matmul(transport, ir_v)
        ir_context = torch.matmul(transport.t(), rgb_v)
        rgb_refined = self.rgb_norm(rgb_memory + self.residual_scale * self.rgb_out(rgb_context))
        ir_refined = self.ir_norm(ir_memory + self.residual_scale * self.ir_out(ir_context))

        row_conf = transport.max(dim=1).values
        col_conf = transport.max(dim=0).values
        return {
            "transport": transport,
            "rgb_refined": rgb_refined,
            "ir_refined": ir_refined,
            "row_conf": row_conf,
            "col_conf": col_conf,
            "reliability": reliability,
        }
