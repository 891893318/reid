import torch
import copy
import torch.nn as nn
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
        # lightweight texture/texture-frequency branch to enrich modality-robust features
        tex_channels = 512
        self.texture_conv = nn.Sequential(
            nn.Conv2d(2048, tex_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.texture_proj = nn.Linear(tex_channels, 2048)
        self.texture_weight = getattr(args, 'feature_texture_weight', 0.2)


    def forward(self,x1=None,x2=None):
        # rgb img as input x1,ir as x2
        if x1 == None and x2 == None:
            raise ValueError("x1 and x2 cannot be None at the same time")
        
        if x1 != None and x2 == None:
            rgb_features = self.rgb_layers(x1)
            rgb_features = self.common_layers(rgb_features)
            GAP_features = self.GAP(rgb_features).squeeze()
            # texture branch
            tex = self.texture_conv(rgb_features).view(GAP_features.shape[0], -1)
            tex_proj = self.texture_proj(tex)
            GAP_features = GAP_features + self.texture_weight * tex_proj
            BN_features = self.BN(GAP_features)
            return GAP_features, BN_features
        
        if x1 == None and x2 != None:
            ir_features = self.ir_layers(x2)
            ir_features = self.common_layers(ir_features)
            GAP_features = self.GAP(ir_features).squeeze()
            tex = self.texture_conv(ir_features).view(GAP_features.shape[0], -1)
            tex_proj = self.texture_proj(tex)
            GAP_features = GAP_features + self.texture_weight * tex_proj
            BN_features = self.BN(GAP_features)
            return GAP_features, BN_features
        
        if x1 != None and x2 != None:
            rgb_features = self.rgb_layers(x1)
            ir_features = self.ir_layers(x2)
            features = torch.cat((rgb_features, ir_features), dim=0)
            features = self.common_layers(features)
            GAP_features = self.GAP(features).squeeze()
            tex = self.texture_conv(features).view(GAP_features.shape[0], -1)
            tex_proj = self.texture_proj(tex)
            GAP_features = GAP_features + self.texture_weight * tex_proj
            BN_features = self.BN(GAP_features)
            return GAP_features, BN_features
