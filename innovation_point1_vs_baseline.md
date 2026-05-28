# 创新点1相对 baseline 的完整总结

更新时间：2026-05-28

本文档用于总结当前“创新点1”相对 baseline 论文的创新位置、方法差异、结构变化、训练目标变化和当前日志证据。

baseline 论文：

```text
Zhang 等, 2025, Weakly Supervised Visible-Infrared Person Re-Identification
via Heterogeneous Expert Collaborative Consistency Learning
```

本项目当前创新点1：

```text
Reliability-Guided Modality-invariant Feature Decoupling, RG-MFD
可靠关系引导的模态不变特征解耦
```

## 一句话结论

baseline 的核心贡献是：在弱监督 VI-ReID 中，利用单模态身份标签训练两个异质专家，再通过 CRE 建立跨模态身份对应关系，并用 CCL 进行跨模态一致性学习。

创新点1的核心贡献是：baseline 虽然能找到可靠跨模态关系，但仍然把这些关系直接作用在混合特征上，没有显式区分“模态共享身份信息”和“模态特有干扰信息”。RG-MFD 在 backbone 输出后加入 shared/specific 解耦分支，让 CRE 的可靠 common relations 主要约束 shared identity subspace，同时用 orthogonality 和 gate regularization 抑制 shared/specific 纠缠与塌缩。

因此，创新点1不是继续改 CRE 的关系排序，而是把 baseline 的“关系级伪监督”提升为“可靠关系引导的特征子空间学习”。

## Baseline 的主要创新点

baseline 论文的整体框架为：

```text
HEL -> CRE -> CCL
```

### 1. 弱监督 VI-ReID 设定

baseline 关注的设定是：

```text
可见光模态内部有身份标签
红外模态内部有身份标签
跨模态身份对应关系未知
```

也就是说，训练时不提供 RGB identity 与 IR identity 之间的显式对应关系。该设定比 fully supervised VI-ReID 标注成本更低，又比完全无监督设定有更可靠的单模态身份监督。

### 2. HEL：Heterogeneous Expert Learning

HEL 使用两个单模态专家分类器：

```text
Wv: visible expert
Wr: infrared expert
```

训练方式：

```text
RGB branch 用 RGB 内部身份标签训练
IR branch 用 IR 内部身份标签训练
```

优化目标主要包括：

```text
L_exp_id: 单模态 identity cross-entropy
L_intr_wrt: 单模态 weighted regularization triplet loss
```

在代码结构上，baseline 的 backbone 采用 AGW/ResNet 结构：

```text
RGB input -> RGB-specific shallow layers
IR input  -> IR-specific shallow layers
二者之后共享 ResNet high-level layers
GAP/GeM pooling -> BN -> classifiers
```

HEL 的作用是先让每个模态的专家获得较强的单模态身份判别能力，为后续跨模态关系预测提供基础。

### 3. CRE：Cross-modal Relationship Establishment

CRE 是 baseline 的核心关系构造模块。

HEL 训练完成后，两个专家会进行跨模态预测：

```text
RGB sample -> IR expert -> RGB-to-IR relation prediction
IR sample  -> RGB expert -> IR-to-RGB relation prediction
```

baseline 使用 Count Priority Selection 得到两个专家决策矩阵：

```text
M^{v->r}
M^{r->v}
```

然后根据双向预测一致性把跨模态关系分成三类：

```text
Mc: consistent/common relations
Ms: single/unique relations
Mw: contradictory/weak/conflicting relations
```

在本项目代码日志里，对应的诊断名称基本是：

```text
common
specific
remain
```

它们的语义可以对应理解为：

```text
common: 双向一致、可靠性最高
specific: 单向或局部成立，可靠性中等
remain: 冲突或模糊关系，噪声较大
```

CRE 的价值是：不只使用双向完全一致关系，也尝试利用单向和模糊关系，从而提高跨模态关系覆盖率。

### 4. CCL：Collaborative Consistency Learning

CCL 包括两部分：

```text
CMCL: Cross-modal Consistency Learning
CLAE: Collaborative Learning Among Experts
```

CMCL 使用 CRE 产生的关系矩阵指导跨模态训练：

```text
L_stro_id: 对可靠关系使用强 cross-modal identity supervision
L_weak_id: 对冲突/弱关系使用 relaxed weak identity constraint
L_cros_wrt: 使用 common relations 构造跨模态 triplet 正负样本
```

CLAE 进一步构建单模态 prototype，并推动两个专家对跨模态对应身份产生更一致的预测。其本质是：

```text
通过 expert homogeneity loss 提升专家之间的跨模态预测一致性
```

baseline phase2 的总目标可以概括为：

```text
L_phase2 = L_exp_id + L_stro_id + L_homo + lambda1 * L_cros_wrt + lambda2 * L_weak_id
```

### 5. Baseline 的结构性特点

baseline 的结构可以简化为：

```text
image
  -> modality-specific shallow encoder
  -> shared ResNet encoder
  -> pooled mixed feature f
  -> BN feature
  -> Wv / Wr / Wc classifiers
  -> CRE relation supervision and CCL losses
```

重要特点：

```text
baseline 建立了跨模态关系，但没有显式拆分 feature。
baseline 的所有跨模态关系监督主要作用在同一个 mixed feature space。
baseline 的 prototype 主要服务于 expert consistency，而不是显式的 shared/specific feature decoupling。
```

## Baseline 的主要不足

当前项目日志和代码分析表明，baseline 的关键不足不是 CRE 完全找不到关系，而是：

```text
CRE 能找到越来越可靠的 common relations，
但这些可靠关系直接约束混合特征，容易把 identity signal、modality bias、camera/domain noise 混在一起。
```

从已有 CRE 诊断看，训练后期 common relations 往往有较高准确率、覆盖率和稳定性；但是 specific/remain 仍然更脏、更不稳定。换句话说：

```text
关系可靠性是有层次的；
feature space 却仍然是单一混合空间。
```

这会带来三个问题：

1. common relation 的可靠监督没有被专门用于塑造模态不变身份子空间。
2. specific/remain 中的模态特有信息和噪声可能污染 identity feature。
3. baseline 的 CRE/CCL 更像样本关系监督，没有回答“可靠关系应该监督哪一部分特征”。

创新点1就是针对这三个问题设计的。

## 创新点1：RG-MFD

RG-MFD 全称：

```text
Reliability-Guided Modality-invariant Feature Decoupling
```

中文名称：

```text
可靠关系引导的模态不变特征解耦
```

核心问题：

```text
baseline 的 CRE 能找到可靠跨模态关系，但这些关系应该约束哪一部分特征？
```

核心回答：

```text
可靠 common relations 应该主要约束 shared identity feature；
模态特有变化和不可靠残差信息应由 specific feature 吸收。
```

## 结构改变

### 1. Baseline 原始结构

baseline AGW/ResNet 输出结构可以概括为：

```text
feature_map
  -> GeM/GAP pooling
  -> base feature, 2048 dim
  -> BN
  -> classifiers / metric learning / CRE-CCL losses
```

所有跨模态监督都作用在同一个 pooled feature 或 BN feature 上。

### 2. 创新点1新增 RGMFD 模块

代码位置：

```text
models/agw.py
```

新增模块：

```python
class RGMFD(nn.Module):
    shared_gate = Linear(2048, hidden) -> ReLU -> Linear(hidden, 2048) -> Sigmoid
```

输入：

```text
base_features: pooled 2048-d feature
```

输出：

```text
shared_features
specific_features
gate
```

当前实现公式：

```text
gate = sigmoid(MLP(base_features))
shared_features = base_features * (1 + gate_scale * (gate - 0.5))
specific_features = base_features * (1 - gate)
```

其中：

```text
gate_scale = 0.5
reduction = 16
```

设计含义：

```text
shared_features 使用残差式 gate 调制，而不是粗暴裁剪；
specific_features 使用 1 - gate 吸收被 shared gate 抑制的模态特有成分；
gate 不是二值 mask，而是通道级软选择。
```

### 3. 新增 shared/specific BN

代码位置：

```text
models/agw.py
```

新增：

```python
self.shared_BN = nn.BatchNorm1d(2048)
self.specific_BN = nn.BatchNorm1d(2048)
```

训练和测试主路径变为：

```text
feature_map
  -> pooling
  -> base_features
  -> RGMFD
  -> shared_features
  -> shared_BN
  -> classifiers / retrieval feature
```

当 `return_rg=True` 时，额外返回：

```text
base_features
shared_features
shared_bn
specific_features
specific_bn
gate
```

这说明创新点1不是只加了一个 loss，而是改变了 backbone 输出后的表示结构：

```text
baseline: mixed feature -> BN
创新点1: mixed feature -> shared/specific decoupling -> shared feature as main embedding
```

### 4. 主检索特征改变

原 baseline 推理时使用：

```text
BN(base_features)
```

创新点1启用后，`AGW.forward()` 返回：

```text
shared_features, shared_bn
```

因此测试阶段提取到的 retrieval embedding 实际来自 shared identity subspace，而不是原始混合特征。

这点是结构变化中最关键的部分：

```text
RG-MFD 不只是辅助训练；它改变了最终用于检索的特征定义。
```

## 损失函数改变

创新点1在 baseline phase1/phase2 的基础上新增三个约束。

### 1. Shared-specific orthogonality loss

代码位置：

```text
task/train.py::_rgmfd_regularization_losses
```

形式：

```text
L_orth = mean( <normalize(shared), normalize(specific)>^2 )
```

代码语义：

```python
shared = F.normalize(rgmfd_pack['shared_features'], dim=1)
specific = F.normalize(rgmfd_pack['specific_features'], dim=1)
L_orth = (shared * specific).sum(dim=1).pow(2).mean()
```

作用：

```text
降低 shared 与 specific 的通道信息重叠；
迫使 shared 更专注 identity/common information；
迫使 specific 更倾向吸收 modality-specific residual。
```

默认权重：

```text
rgmfd_orth_weight = 0.05
```

### 2. Gate balance regularization

代码位置：

```text
task/train.py::_rgmfd_regularization_losses
```

形式：

```text
L_gate = mean( mean(gate) - gate_target )^2
```

默认设置：

```text
rgmfd_gate_reg_weight = 0.01
rgmfd_gate_target = 0.5
```

作用：

```text
防止 gate 全部偏向 shared 或全部偏向 specific；
避免解耦分支塌缩；
保持 shared/specific 都有有效容量。
```

### 3. Reliable-relation shared alignment loss

代码位置：

```text
task/train.py
```

触发条件：

```text
enable_rgmfd = 1
epoch >= rgmfd_start_epoch
当前 batch 中存在 common relations
```

对齐方式：

```text
RGB shared feature -> matched IR memory
IR shared feature  -> matched RGB memory
```

代码语义：

```text
selected_rgb_shared = shared_bn[RGB common samples]
selected_ir_shared  = shared_bn[IR common samples]

L_rel = 1 - cosine(shared_feature, opposite_modality_memory)
```

最终权重：

```text
L_rel = common_loss_weight * rgmfd_rel_weight * cosine_alignment_loss
```

作用：

```text
只用 CRE 中可靠度最高的 common relations 对齐 shared feature；
不把所有模糊关系都强行压进身份子空间；
把 baseline 的关系监督从 mixed feature space 转移到 shared feature space。
```

## 新总目标

baseline phase2 可概括为：

```text
L_phase2_baseline =
  L_exp_id
  + L_stro_id
  + L_homo
  + lambda1 * L_cros_wrt
  + lambda2 * L_weak_id
```

创新点1加入：

```text
L_RGMFD =
  alpha * L_rel(shared, reliable_common_memory)
  + beta * L_orth(shared, specific)
  + gamma * L_gate(gate)
```

因此新目标为：

```text
L_phase2_new = L_phase2_baseline + L_RGMFD
```

phase1 中也可以启用：

```text
L_phase1_new = L_phase1_baseline + beta * L_orth + gamma * L_gate
```

当 `rgmfd_start_epoch > 0` 时，RG-MFD 辅助约束延后生效。

## 与 baseline 的本质区别

| 维度 | baseline | 创新点1 |
|---|---|---|
| 关系来源 | HEL 专家预测 + CRE | 沿用 CRE 的可靠关系 |
| 关系使用方式 | 作为 cross-modal pseudo-label 约束 mixed feature | 用 common relations 专门约束 shared identity subspace |
| 特征结构 | 单一 mixed feature | shared feature + specific feature |
| 检索特征 | BN(base feature) | shared_BN(shared feature) |
| 模态特有信息 | 没有显式承接空间 | specific feature 吸收 residual/modality-specific information |
| 防塌缩机制 | 无专门解耦 gate 约束 | gate balance regularization |
| 解耦约束 | 无显式 shared/specific 正交 | orthogonality loss |
| 论文叙事 | 建立跨模态关系并做一致性学习 | 可靠关系引导模态不变子空间学习 |

最重要的差异是：

```text
baseline 解决“如何找到跨模态关系”；
创新点1进一步解决“可靠跨模态关系应该塑造哪一部分特征”。
```

## 为什么这不是简单堆模块

RG-MFD 与 baseline 的 CRE/CCL 是强耦合关系：

```text
没有 CRE，RG-MFD 不知道哪些跨模态关系可靠；
没有 RG-MFD，CRE 的可靠关系仍只能监督 mixed feature；
二者结合后，common relations 被用于塑造 modality-invariant shared subspace。
```

因此创新点1不是单独的 attention，也不是普通 feature gate，而是：

```text
relation reliability -> feature decoupling target
```

也就是把关系可靠性转化为特征空间结构约束。

## 当前创新点1日志结果

当前识别到的“创新点1”日志：

```text
logs/sysu/all/2026年05月25日-18:11:48(创新点1)/log/log.txt
logs/sysu/indoor/2026年05月26日-09:43:30(创新点1)/log/log.txt
logs/llcm/t2v/2026年05月27日-18:01:23(创新点1)/log/log.txt
logs/llcm/v2t/2026年05月27日-19:47:16(创新点1)/log/log.txt
```

### SYSU-MM01

| 设置 | baseline paper | 创新点1 best | 提升 |
|---|---:|---:|---:|
| all Rank-1 | 70.4 | 71.23 | +0.83 |
| all mAP | 66.6 | 68.41 | +1.81 |
| indoor Rank-1 | 76.5 | 79.20 | +2.70 |
| indoor mAP | 80.2 | 82.63 | +2.43 |

当前日志配置：

```text
SYSU all:
enable_rgmfd=1
rgmfd_start_epoch=20
rgmfd_rel_weight=0.1
rgmfd_orth_weight=0.05
rgmfd_gate_reg_weight=0.01
enable_rdl=1
model_path=/root/saved_pretrain_sysu_resnet/model_20.pth

SYSU indoor:
enable_rgmfd=1
rgmfd_start_epoch=0
rgmfd_rel_weight=0.1
rgmfd_orth_weight=0.05
rgmfd_gate_reg_weight=0.01
enable_rdl=0
model_path=default
```

注意：

```text
SYSU all 和 indoor 当前最强配置不完全一致。
若论文主表要求同一数据集同一方法参数，需要后续统一或在文中解释为 protocol-specific validation。
```

### LLCM

代码中 LLCM 的协议方向为：

```text
test_mode=t2v: IR/Thermal query -> VIS gallery
test_mode=v2t: VIS query -> IR/Thermal gallery
```

| 设置 | baseline paper | 创新点1 best | 提升 |
|---|---:|---:|---:|
| LLCM IR -> VIS Rank-1 | 47.3 | 47.76 | +0.46 |
| LLCM IR -> VIS mAP | 53.3 | 54.37 | +1.07 |
| LLCM VIS -> IR Rank-1 | 55.3 | 56.60 | +1.30 |
| LLCM VIS -> IR mAP | 58.7 | 60.41 | +1.71 |

当前日志配置：

```text
LLCM t2v:
enable_rgmfd=1
rgmfd_start_epoch=0
rgmfd_rel_weight=0.3
rgmfd_orth_weight=0.05
rgmfd_gate_reg_weight=0.01
enable_rdl=0
model_path=/root/WSL_ReID/logs/llcm/t2v/2026年05月27日-11:45:00/models/stage1/model_80.pth

LLCM v2t:
enable_rgmfd=1
rgmfd_start_epoch=0
rgmfd_rel_weight=0.3
rgmfd_orth_weight=0.05
rgmfd_gate_reg_weight=0.01
enable_rdl=0
model_path=default
```

LLCM 的方法超参数一致：

```text
rel=0.3
start=0
RDL=0
orth=0.05
gate=0.01
lr=0.0003
stage1_epoch=80
stage2_epoch=120
```

但训练流程记录略有差异：

```text
t2v 显式加载 stage1 checkpoint 后跑 phase2；
v2t 使用 default，从 phase1 到 phase2 完整跑。
```

从代码看，LLCM 的 `test_mode` 只影响测试 query/gallery 方向，训练集仍是：

```text
train_vis.txt + train_nir.txt
```

因此严格论文实验中，建议让两个方向采用一致训练流程：

```text
要么两个方向都从 default 完整训练；
要么两个方向都显式加载各自 stage1/model_80.pth 后训练 phase2。
```

## 当前创新点1的论文贡献表述

可以写成如下三点。

### 贡献1：可靠关系引导的模态不变特征解耦

原文可写：

```text
We propose a reliability-guided modality-invariant feature decoupling module,
which transforms reliable cross-modal relations discovered by CRE from
sample-level pseudo supervision into explicit constraints on a modality-invariant
shared identity subspace.
```

中文表述：

```text
我们提出可靠关系引导的模态不变特征解耦模块，将 CRE 挖掘出的可靠跨模态关系从样本级伪监督提升为对模态不变共享身份子空间的显式约束。
```

### 贡献2：shared/specific gate 解耦结构

英文表述：

```text
We design a lightweight shared-specific gating mechanism to decompose the
pooled visual representation into a shared identity feature and a modality-specific
residual feature, reducing the interference of modality-dependent variations in
cross-modal identity matching.
```

中文表述：

```text
我们设计轻量级 shared-specific 门控结构，将全局视觉表征分解为共享身份特征和模态特定残差特征，降低模态相关变化对跨模态身份匹配的干扰。
```

### 贡献3：解耦约束与可靠关系对齐联合优化

英文表述：

```text
The proposed module is optimized with reliable-relation alignment, shared-specific
orthogonality, and gate-balance regularization, enabling stable decoupling while
avoiding feature collapse.
```

中文表述：

```text
该模块通过可靠关系对齐、共享/特定特征正交约束和 gate 平衡正则联合优化，在防止特征塌缩的同时实现稳定解耦。
```

## 方法章节可用写法

### Motivation

baseline 的 CRE 可以构造跨模态身份关系，但这些关系被直接用于监督混合特征。由于混合特征同时包含身份判别信息、模态差异、相机域偏差和噪声，直接对齐可能导致模态特定信息污染身份空间。为此，我们提出 RG-MFD，使可靠 common relations 只作用于 shared identity representation，而将模态特定因素隔离到 specific branch 中。

### Method

给定 backbone 输出的 pooled feature：

```text
f in R^2048
```

RG-MFD 首先生成通道级 gate：

```text
g = sigmoid(MLP(f))
```

然后得到：

```text
f_sh = f * (1 + s * (g - 0.5))
f_sp = f * (1 - g)
```

其中 `s` 是 gate scale。`f_sh` 经过 `shared_BN` 后作为主检索特征，`f_sp` 只用于解耦约束。

对可靠 common relation `(i, j)`，使用：

```text
L_rel = 1 - cos(f_sh^i, memory_opposite^j)
```

并加入：

```text
L_orth = <normalize(f_sh), normalize(f_sp)>^2
L_gate = (mean(g) - 0.5)^2
```

最终：

```text
L = L_baseline + alpha * L_rel + beta * L_orth + gamma * L_gate
```

## 建议补充的消融实验

为了让创新点1在论文里更完整，建议至少补以下消融。

### 模块级消融

| 实验 | 目的 |
|---|---|
| baseline | 原 HEL+CRE+CCL |
| + RG-MFD | 验证整体模块有效性 |
| w/o L_rel | 验证可靠关系对 shared subspace 的作用 |
| w/o L_orth | 验证 shared/specific 解耦必要性 |
| w/o L_gate | 验证 gate 防塌缩必要性 |
| mixed feature alignment | 验证对齐 shared feature 优于对齐原混合特征 |

### 参数敏感性

| 参数 | 建议范围 |
|---|---|
| rgmfd_rel_weight | 0.1 / 0.2 / 0.3 |
| rgmfd_start_epoch | 0 / 20 |
| rgmfd_orth_weight | 0 / 0.05 |
| rgmfd_gate_reg_weight | 0 / 0.01 |

### 公平性检查

| 检查项 | 当前状态 | 建议 |
|---|---|---|
| SYSU all/indoor 参数一致性 | 当前不一致 | 补统一配置或明确 protocol-specific |
| LLCM t2v/v2t 方法参数一致性 | 一致 | 训练流程最好也一致 |
| RegDB 验证 | 尚未形成完整闭环 | 补 t2v/v2t |
| stage1 checkpoint 使用 | 部分日志显式加载，部分 default | 主表实验尽量统一 |

## 当前风险点

### 1. SYSU 参数尚未统一

当前最强：

```text
SYSU all: start=20, RDL=1
SYSU indoor: start=0, RDL=0
```

这有利于展示 best performance，但如果主表强调同一数据集同一套方法参数，需要补统一配置实验。

### 2. LLCM 训练流程记录不完全一致

LLCM 的方法超参一致，但 t2v 和 v2t 的 `model_path` 记录不同。严格来说，这不影响“方法参数一致”，但论文实验最好统一流程。

### 3. RegDB 还需要补

baseline 论文主要验证 SYSU 和 LLCM；如果当前论文目标更高，建议加入 RegDB 作为额外跨数据集验证。当前日志中 RegDB 的 RG-MFD 还没有形成有效完整结果。

### 4. 需要证明不是调参涨点

创新点1应通过结构消融证明：

```text
性能提升来自 shared/specific decoupling 和 reliable-relation shared alignment，
而不是单纯来自 rel_weight、start_epoch 或 RDL 调参。
```

## 推荐论文叙事

建议把论文主线写成：

```text
Weakly supervised VI-ReID suffers not only from missing cross-modal identity
annotations, but also from the entanglement between modality-invariant identity
information and modality-specific variations in the learned representations.
Existing HEL-CRE-CCL establishes reliable cross-modal relations, but directly
uses them to supervise mixed features. We argue that reliable relations should
specifically shape the modality-invariant identity subspace. To this end, we
propose RG-MFD, a reliability-guided modality-invariant feature decoupling
framework that decomposes visual features into shared identity and modality-
specific components, and aligns only the shared component using reliable common
relations.
```

中文版本：

```text
弱监督 VI-ReID 不仅面临跨模态身份对应缺失的问题，还面临学习特征中身份信息与模态特有变化纠缠的问题。现有 HEL-CRE-CCL 能够建立可靠跨模态关系，但这些关系仍被直接用于监督混合特征。我们认为，可靠关系应专门用于塑造模态不变身份子空间。为此，本文提出 RG-MFD，通过轻量级门控将视觉表征分解为共享身份特征和模态特定特征，并仅使用高可靠 common relations 对共享身份特征进行跨模态对齐。
```

## 最终定位

创新点1相对 baseline 的真正创新位置是：

```text
从“建立可靠跨模态关系”推进到“利用可靠关系学习模态不变解耦特征”。
```

如果只说“在 baseline 上加了一个 gate”，创新性会显得弱；正确表述应强调：

```text
关系可靠性驱动的特征子空间结构化学习。
```

这也是后续冲更高分区时最应强化的理论叙事。
