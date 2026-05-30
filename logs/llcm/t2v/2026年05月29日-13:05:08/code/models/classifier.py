import torch.nn as nn
import torch

def weights_init_kaiming(m):
    """
    使用 Kaiming 初始化方法对网络层进行权重初始化。
    主要用于在没有使用预训练权重时的特定模块，以避免在深层网络中梯度消失或爆炸。
    """
    if isinstance(m, nn.Linear):
        # 线性层：fan_out 模式，保持前向传播或反向传播时网络方差一致
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, val=0.0)
    elif isinstance(m, nn.Conv2d):
        # 卷积层：fan_in 模式
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias:
            nn.init.constant_(m.bias, val=0.0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
        # 归一化层：权重(Gamma)设为 1，偏置(Beta)设为 0
        if m.affine:
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0.0)

def weights_init(m):
    """
    基础的正态分布权重初始化方法。
    通常用于表征特征维度的分类器层 (Classifier/全连接层) 的初始化，提供一个很小的非零初始权重池。
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=1e-3)
        if m.bias:
            nn.init.constant_(m.bias, val=0.0)

class Normalize(nn.Module):
    """
    L2 归一化层。
    常用于度量学习中的特征归一化，使得特征向量被投影到单位超球面上。
    这能方便后续直接使用余弦相似度和欧氏距离来精准衡量特征样本间的距离。
    """
    def __init__(self,power = 2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, input):
        norm = input.pow(self.power).sum(dim=1, keepdim=True).pow(1 / self.power)
        output = input / norm
        return output

class GeneralizedMeanPooling(nn.Module):
    """
    GeM (Generalized Mean Pooling) 广义平均池化层。
    它是平均池化 (Average) 和最大池化 (Max Pooling) 的泛化整合形式：
    - 当 p=1 时，等价于普通平均池化
    - 当 p 趋近于无穷大时，等价于最大池化
    在图像检索和 ReID 任务中，利用 GeM 池化操作往往比单纯的全局平均池化性能更好，
    因为它能够更重点地关注到显著特征（即具有判别性的局部区域）。
    """
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm) # p 参数 控制着池化强度的“偏好”
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        # 为了避免除根导致的数值下溢或极值问题，进行 eps 钳夹并求指数 p
        x = x.clamp(min=self.eps).pow(self.p)
        # 执行普通的自适应平均池化，然后求其 1/p 次方来恢复数值尺度
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.p)
            + ", "
            + "output_size="
            + str(self.output_size)
            + ")"
        )

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """
    可学习特征指数的 GeM 广义平均池化层。
    与上面的静态类唯一区别是：这里的归一化池定范数参数 `self.p`
    被包裹为了一个 `nn.Parameter`，它会在模型训练时随反向传播自动更新，
    自适应地找寻到网络在当前表征流下的最优池化硬度。
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        # 将 p 设为了一个可学习优化的 Tensor 参数
        self.p = nn.Parameter(torch.ones(1) * norm)

# class Text_Classifier(nn.Module):
#     def __init__(self,args):
#         super(Text_Classifier, self, ).__init__()
#         self.num_classes = args.num_classes
#         self.BN = nn.BatchNorm1d(1024)
#         self.BN.apply(weights_init_kaiming)

#         self.classifier = nn.Linear(1024, self.num_classes, bias=False)
#         self.classifier.apply(weights_init)

#         self.l2_norm = Normalize(2)

#     def forward(self, features):
#         bn_features = self.BN(features.squeeze())
#         cls_score = self.classifier(bn_features)
#         if self.training:
#             return cls_score
#         else:
#             self.l2_norm(bn_features)


class Image_Classifier(nn.Module):
    """
    图像侧主干网络分类分支。
    将经过骨架 Backbone (如 ResNet50) 与 Spatial Pooling 层提取到的最初图像高维特征，
    映射到具体的各个目标身份 (Identity Classes) 空间中。
    一方面输出身份预测定 Logits 供 Cross Entropy ID Loss 计算梯度，
    另一方面输出归一化后的表征特征尺度，供 Triplet Loss(三元组损失) 和检索测距度。
    
    **Output: x_score (分类 Logits), x_l2 (模长为 1 的 L2 归一化特征)**
    """
    def __init__(self, args):
        super(Image_Classifier, self).__init__()
        self.num_classes = args.num_classes
        # 视觉 ResNet50 或大部分 AGW 系列表征网络输出层宽为 2048
        self.classifier = nn.Linear(2048, args.num_classes, bias = False)
        self.classifier.apply(weights_init)

        # L2 归一化层，方便进行余弦相似度计算或进一步求导距离
        self.l2_norm = Normalize(2)

    def forward(self,x_bn):
        # 算出传入前向图被预测为各大 ID 区间的未 Softmax 概率
        x_score = self.classifier(x_bn)
        # 返回分类分数和单位化特征
        return x_score, self.l2_norm(x_bn)


