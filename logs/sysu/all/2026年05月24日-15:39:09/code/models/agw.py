import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .classifier import Image_Classifier, GeneralizedMeanPoolingP, weights_init_kaiming
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
        self.enable_structured_feature = getattr(args, 'enable_structured_feature', 1)
        self.structured_parts = max(1, int(getattr(args, 'structured_parts', 3)))
        self.structured_weight = getattr(args, 'structured_weight', 0.2)
        self.part_bn = nn.BatchNorm1d(2048)
        self.part_bn.apply(weights_init_kaiming)
        self.part_attn = nn.Linear(2048, 1, bias=True)
        nn.init.normal_(self.part_attn.weight, std=1e-3)
        nn.init.constant_(self.part_attn.bias, 0.0)

    def _flatten_feature(self, feature):
        return feature.flatten(1)

    def _fuse_structured_feature(self, feature_map, global_feature):
        if not self.enable_structured_feature:
            return global_feature

        part_map = F.adaptive_avg_pool2d(feature_map, (self.structured_parts, 1))
        part_feature = part_map.squeeze(-1).permute(0, 2, 1).contiguous()
        batch_size, part_count, channel_dim = part_feature.shape
        part_feature = part_feature.view(batch_size * part_count, channel_dim)
        part_feature = self.part_bn(part_feature)
        part_feature = part_feature.view(batch_size, part_count, channel_dim)

        attn_score = self.part_attn(part_feature.reshape(batch_size * part_count, channel_dim))
        attn_score = attn_score.view(batch_size, part_count)
        attn_weight = torch.softmax(attn_score, dim=1).unsqueeze(-1)
        structured_feature = (part_feature * attn_weight).sum(dim=1)
        return global_feature + self.structured_weight * structured_feature


    def forward(self,x1=None,x2=None):
        # rgb img as input x1,ir as x2
        if x1 == None and x2 == None:
            raise ValueError("x1 and x2 cannot be None at the same time")
        
        if x1 != None and x2 == None:
            rgb_features = self.rgb_layers(x1)
            rgb_features = self.common_layers(rgb_features)
            GAP_features = self._flatten_feature(self.GAP(rgb_features))
            GAP_features = self._fuse_structured_feature(rgb_features, GAP_features)
            BN_features = self.BN(GAP_features)
            return GAP_features, BN_features
        
        if x1 == None and x2 != None:
            ir_features = self.ir_layers(x2)
            ir_features = self.common_layers(ir_features)
            GAP_features = self._flatten_feature(self.GAP(ir_features))
            GAP_features = self._fuse_structured_feature(ir_features, GAP_features)
            BN_features = self.BN(GAP_features)
            return GAP_features, BN_features
        
        if x1 != None and x2 != None:
            rgb_features = self.rgb_layers(x1)
            ir_features = self.ir_layers(x2)
            features = torch.cat((rgb_features, ir_features), dim=0)
            features = self.common_layers(features)
            GAP_features = self._flatten_feature(self.GAP(features))
            GAP_features = self._fuse_structured_feature(features, GAP_features)
            BN_features = self.BN(GAP_features)
            return GAP_features, BN_features
