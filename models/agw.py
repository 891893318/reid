import torch
import copy
import torch.nn as nn
import torchvision.models as models

from .classifier import Image_Classifier, GeneralizedMeanPoolingP, weights_init_kaiming
from .frequency import FeatureFrequencyDecomposer
class AGW(nn.Module):
    """
    input: x1, x2 are inputs for visible and infrared, respectively.
    """
    def __init__(self, args):
        super(AGW, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride = (1,1)
        resnet50.layer4[0].downsample[0].stride = (1,1)

        self.rgb_layers = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.maxpool)
        self.ir_layers = copy.deepcopy(self.rgb_layers)
        self.common_layers = nn.Sequential(
            resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4
        )
        self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)
        self.enable_fimro = getattr(args, "enable_fimro", 1) == 1
        self.frequency_decomposer = FeatureFrequencyDecomposer(
            low_ratio=getattr(args, "fimro_low_ratio", 0.25),
            low_noise_scale=getattr(args, "fimro_low_noise", 0.15),
            fuse_scale=getattr(args, "fimro_fuse_scale", 0.2),
            mask_mode=getattr(args, "fimro_mask_mode", "square"),
        )

    def _pool_bn(self, feature_map):
        gap_features = self.GAP(feature_map).flatten(1)
        bn_features = self.BN(gap_features)
        return gap_features, bn_features

    def _forward_with_frequency(self, features):
        freq_outputs = self.frequency_decomposer(features)
        # Conservative FIMRO: keep the baseline main path untouched and
        # use the frequency branch only for auxiliary phase-1 supervision.
        gap_features, bn_features = self._pool_bn(features)
        low_aug_gap = self.GAP(freq_outputs["low_aug_map"]).flatten(1)
        high_gap = self.GAP(freq_outputs["high_map"]).flatten(1)
        freq_outputs["low_aug_gap"] = low_aug_gap
        freq_outputs["high_gap"] = high_gap
        return gap_features, bn_features, freq_outputs

    def forward(self,x1=None,x2=None, return_freq=False):
        # rgb img as input x1,ir as x2
        if x1 == None and x2 == None:
            raise ValueError("x1 and x2 cannot be None at the same time")
        
        if x1 != None and x2 == None:
            rgb_features = self.rgb_layers(x1)
            rgb_features = self.common_layers(rgb_features)
            if return_freq and self.enable_fimro:
                return self._forward_with_frequency(rgb_features)
            GAP_features, BN_features = self._pool_bn(rgb_features)
            return GAP_features, BN_features
        
        if x1 == None and x2 != None:
            ir_features = self.ir_layers(x2)
            ir_features = self.common_layers(ir_features)
            if return_freq and self.enable_fimro:
                return self._forward_with_frequency(ir_features)
            GAP_features, BN_features = self._pool_bn(ir_features)
            return GAP_features, BN_features
        
        if x1 != None and x2 != None:
            rgb_features = self.rgb_layers(x1)
            ir_features = self.ir_layers(x2)
            features = torch.cat((rgb_features, ir_features), dim=0)
            features = self.common_layers(features)
            if return_freq and self.enable_fimro:
                return self._forward_with_frequency(features)
            GAP_features, BN_features = self._pool_bn(features)
            return GAP_features, BN_features
