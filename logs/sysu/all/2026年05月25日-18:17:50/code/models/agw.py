import torch
import copy
import torch.nn as nn
import torchvision.models as models

from .classifier import Image_Classifier, GeneralizedMeanPoolingP, weights_init_kaiming


class RGMFD(nn.Module):
    def __init__(self, feature_dim=2048, reduction=16, gate_scale=0.5):
        super(RGMFD, self).__init__()
        hidden_dim = max(128, feature_dim // max(1, reduction))
        self.gate_scale = gate_scale
        self.shared_gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, features):
        gate = self.shared_gate(features)
        shared_features = features * (1.0 + self.gate_scale * (gate - 0.5))
        specific_features = features * (1.0 - gate)
        return shared_features, specific_features, gate


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
        self.enable_rgmfd = getattr(args, "enable_rgmfd", 0) == 1
        if self.enable_rgmfd:
            self.rgmfd = RGMFD(
                feature_dim=2048,
                reduction=getattr(args, "rgmfd_reduction", 16),
                gate_scale=getattr(args, "rgmfd_gate_scale", 0.5),
            )
            self.shared_BN = nn.BatchNorm1d(2048)
            self.specific_BN = nn.BatchNorm1d(2048)
            self.shared_BN.apply(weights_init_kaiming)
            self.specific_BN.apply(weights_init_kaiming)

    def _pool_feature_map(self, feature_map, return_rg=False):
        base_features = self.GAP(feature_map).flatten(1)
        if not self.enable_rgmfd:
            bn_features = self.BN(base_features)
            return base_features, bn_features

        shared_features, specific_features, gate = self.rgmfd(base_features)
        shared_bn = self.shared_BN(shared_features)
        if return_rg:
            specific_bn = self.specific_BN(specific_features)
            rgmfd_pack = {
                "base_features": base_features,
                "shared_features": shared_features,
                "shared_bn": shared_bn,
                "specific_features": specific_features,
                "specific_bn": specific_bn,
                "gate": gate,
            }
            return shared_features, shared_bn, rgmfd_pack
        return shared_features, shared_bn

    def forward(self,x1=None,x2=None, return_rg=False):
        # rgb img as input x1,ir as x2
        if x1 == None and x2 == None:
            raise ValueError("x1 and x2 cannot be None at the same time")
        
        if x1 != None and x2 == None:
            rgb_features = self.rgb_layers(x1)
            rgb_features = self.common_layers(rgb_features)
            return self._pool_feature_map(rgb_features, return_rg)
        
        if x1 == None and x2 != None:
            ir_features = self.ir_layers(x2)
            ir_features = self.common_layers(ir_features)
            return self._pool_feature_map(ir_features, return_rg)
        
        if x1 != None and x2 != None:
            rgb_features = self.rgb_layers(x1)
            ir_features = self.ir_layers(x2)
            features = torch.cat((rgb_features, ir_features), dim=0)
            features = self.common_layers(features)
            return self._pool_feature_map(features, return_rg)
