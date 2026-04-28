import torch
import torch.nn as nn


def _get_frequency_masks(height, width, low_ratio, mode="square"):
    if mode not in {"square", "circle"}:
        raise ValueError(f"Unsupported mask mode: {mode}")

    h_freqs = torch.fft.fftshift(torch.fft.fftfreq(height))
    w_freqs = torch.fft.fftshift(torch.fft.fftfreq(width))
    grid_h, grid_w = torch.meshgrid(h_freqs, w_freqs, indexing="ij")

    bound = 0.5 * float(low_ratio)
    if mode == "square":
        low_mask = ((grid_h.abs() <= bound) & (grid_w.abs() <= bound)).float()
    else:
        low_mask = ((grid_h.pow(2) + grid_w.pow(2)) <= bound**2).float()
    high_mask = 1.0 - low_mask
    return low_mask, high_mask


class HiLoFrequencyModulator(nn.Module):
    """
    HiLo-inspired feature modulation with a fixed low/high frequency split.
    Unlike MFENet, this stays outside the backbone topology and only provides
    auxiliary phase-1 supervision plus a light residual enhancement.
    """

    def __init__(self, low_ratio=0.25, low_noise_scale=0.15, fuse_scale=0.2, mask_mode="square"):
        super().__init__()
        self.low_ratio = low_ratio
        self.low_noise_scale = low_noise_scale
        self.fuse_scale = fuse_scale
        self.mask_mode = mask_mode

        self.low_scale = nn.Parameter(torch.tensor(1.0))
        self.high_scale = nn.Parameter(torch.tensor(1.0))
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.spatial_gate[0].weight)
        nn.init.zeros_(self.spatial_gate[0].bias)

    def _build_masks(self, height, width, device, dtype):
        low_mask, high_mask = _get_frequency_masks(height, width, self.low_ratio, self.mask_mode)
        low_mask = low_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        high_mask = high_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        return low_mask, high_mask

    def forward(self, feature_map):
        if feature_map.dim() != 4:
            raise ValueError("feature_map must be a 4D tensor")

        _, _, height, width = feature_map.shape
        x = feature_map.float()
        fft_map = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        low_mask, high_mask = self._build_masks(height, width, x.device, fft_map.real.dtype)

        low_fft = fft_map * low_mask
        high_fft = fft_map * high_mask

        low_map = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real
        high_map = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real

        low_noise = 1.0 + torch.randn_like(low_map) * self.low_noise_scale
        low_aug_map = low_map * low_noise + high_map

        gate_input = x.mean(dim=1, keepdim=True)
        gate = self.spatial_gate(gate_input)
        low_gate = gate[:, :1]
        high_gate = gate[:, 1:]

        mod_low = self.low_scale * low_map * low_gate
        mod_high = self.high_scale * high_map * high_gate
        fused_map = x + self.fuse_scale * (mod_low + mod_high)

        return {
            "low_map": low_map.type_as(feature_map),
            "high_map": high_map.type_as(feature_map),
            "low_aug_map": low_aug_map.type_as(feature_map),
            "fused_map": fused_map.type_as(feature_map),
            "low_mask": low_mask.type_as(feature_map),
            "high_mask": high_mask.type_as(feature_map),
        }


class FeatureFrequencyDecomposer(HiLoFrequencyModulator):
    pass
