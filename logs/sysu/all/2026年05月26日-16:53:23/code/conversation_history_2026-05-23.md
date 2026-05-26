# WSL_ReID 对话与实验历史总结

更新时间：2026-05-26  
工作目录：`/root/WSL_ReID`  
当前阶段：从 CRE 小改进转向 RG-MFD 主线，目标一区/二区论文实验闭环

## 说明

本文件覆盖旧版 `conversation_history_2026-05-23.md`，用于记录本项目从最初讨论到当前实验结论的完整脉络。内容不是逐字聊天记录，而是按研究决策、代码修改、实验结果和后续计划整理的可读版总结。

相关汇总文件：

- `experiment_log_summary.md`：自动扫描 `logs/*/*/*/log/log.txt` 后生成的实验表格。
- `logs/sysu/all/2026年05月25日-18:11:48(best)/log/log.txt`：当前 SYSU all 最强日志。
- `logs/sysu/indoor/2026年05月26日-09:43:30/log/log.txt`：当前 SYSU indoor 最高 Rank-1 日志，但状态为 partial。

## 一句话结论

历史讨论已经证明：**继续只改 CRE 排序策略不适合作为主涨点，CRE 后期已经接近饱和；当前最有价值的主线是 RG-MFD，即可靠关系引导的模态不变特征解耦。**

当前 SYSU all 最强结果：

```text
Rank-1: 71.23
mAP:    68.41
配置:   RG-MFD + RDL + rgmfd_start_epoch=20 + rel=0.1 + orth=0.05 + gate=0.01
日志:   logs/sysu/all/2026年05月25日-18:11:48(best)/log/log.txt
```

相对用户提到的论文 baseline：

```text
论文 Rank-1: 70.5
复现 Rank-1: 69.2 左右
旧 best Rank-1: 69.6
当前 best Rank-1: 71.23
```

因此 SYSU all 已经超过论文 Rank-1 约 0.7 个点，mAP 高接近 2 个点。

## 研究背景

项目方向：弱监督可见光-红外行人重识别，Weakly Supervised Visible-Infrared Person Re-identification。

代码目录：

```text
/root/WSL_ReID
```

参考论文主线：

- HEL：Heterogeneous Expert Learning
- CRE：Cross-modality Relation Exploration
- CCL：Collaborative Contrastive Learning

主要数据集和协议：

- SYSU-MM01：`all` 和 `indoor`
- RegDB：`t2v` 和 `v2t`
- LLCM：`t2v` 和 `v2t`

主要指标：

- Rank-1
- Rank-10
- Rank-20
- mAP
- mINP

用户目标：

- 初期目标：二区，做稳健可解释的小改进。
- 后期目标：一区和二区，一区不能只改 CRE，需要更完整的结构性贡献。

## 总体路线演进

整个对话可以分成五个阶段：

1. 从原论文 CRE 入手，尝试 confidence-aware CRE。
2. 加 CRE 诊断，确认 CRE 后期接近饱和。
3. 尝试 TRRM，用历史稳定性过滤 specific/remain。
4. 升级为一区主线 RG-MFD，做特征层解耦。
5. 结合实验日志修正判断，确定当前最佳配置是 RG-MFD + RDL + delayed start。

## 阶段一：尝试改 CRE

最早的问题是：论文还有哪里可以改进涨点，目标二区。

当时判断：

- CRE 是原论文核心模块之一。
- CRE 比主干网络更容易做局部改动。
- 如果只做二区目标，可以优先尝试改 CRE 的关系选择策略。

于是尝试把原始 Count Priority CRE 改成 confidence-aware CRE。

原始 Count Priority CRE 的思路：

- 对跨模态候选匹配关系计数。
- 选择出现次数高的关系作为可靠关系。
- 优点是稳定。
- 缺点是没有显式利用概率、margin、entropy 等置信度信息。

confidence-aware CRE 的目标：

- 不只看 count。
- 同时考虑 expert probability。
- 同时考虑 top 候选和次优候选之间的 margin。
- 同时考虑 entropy confidence。
- 可选考虑 prototype similarity。

保留的关键开关：

```text
--cre-confidence 0
--cre-confidence 1
```

含义：

- `--cre-confidence 0`：关闭 confidence-aware CRE，回到原始 Count Priority CRE。
- `--cre-confidence 1`：启用置信度感知 CRE。

这个开关后来很重要，因为置信度 CRE 并没有稳定涨点，必须保留原始 CRE 作为回退和消融基线。

## 阶段二：confidence-aware CRE 反而掉点

用户跑完后反馈：confidence-aware CRE 反而掉点。

当时的判断：

- 原始 Count Priority 在当前代码和数据上已经很稳。
- confidence-aware 重新排序可能破坏了原本稳定的 common relations。
- CRE 的问题不一定是排序不够聪明，而可能是后期已经饱和。

因此下一步不是继续调置信度权重，而是加诊断指标，确认 CRE 到底是否饱和。

## CRE 诊断指标

新增 CRE 诊断日志，用真实标签只做分析，不参与训练决策。

日志格式：

```text
CRE diag | r2i:xx.xx%(a/b) i2r:xx.xx%(a/b) common:xx.xx%(a/b) specific:xx.xx%(a/b) remain:xx.xx%(a/b) cov_c:xx.xx% cov_all:xx.xx% stable_c:xx.xx%
```

字段含义：

- `r2i`：RGB 到 IR 单向匹配准确率。
- `i2r`：IR 到 RGB 单向匹配准确率。
- `common`：双向一致的可靠关系准确率。
- `specific`：模态特定或单向成立关系准确率。
- `remain`：剩余模糊关系准确率。
- `cov_c`：common 关系覆盖率。
- `cov_all`：所有关系覆盖率。
- `stable_c`：common 关系跨 epoch 稳定性。

早期典型日志：

```text
CRE diag | r2i:54.84%(170/310) i2r:59.66%(139/233) common:95.50%(106/111) specific:33.33%(20/60) remain:32.16%(55/171) cov_c:28.10% cov_all:85.82% stable_c:0.00%
```

解释：

- 早期 `r2i/i2r` 单向匹配噪声很大。
- 早期 `common` 数量少，但准确率很高。
- 早期 `specific/remain` 很脏，不适合强监督。
- `cov_c` 低不是坏事，说明模型只保留了少量高可信 common。

中后期典型日志：

```text
CRE diag | r2i:93.21%(357/383) i2r:91.62%(339/370) common:94.68%(338/357) specific:69.23%(9/13) remain:22.22%(4/18) cov_c:90.38% cov_all:98.23% stable_c:98.60%
```

解释：

- 后期 common 准确率很高。
- common 覆盖率也高。
- common 稳定性接近 100%。
- remain 仍然很不稳定。

关键结论：

```text
CRE 后期已经接近饱和。
继续改 CRE 排序策略，收益有限。
主要噪声来自 specific/remain。
```

## 阶段三：TRRM 尝试

因为 CRE 诊断说明 early specific/remain 很脏，所以提出 TRRM。

TRRM 全称：

```text
Temporal Reliability Relation Memory
```

中文定位：

```text
时序可靠关系记忆
```

核心思路：

- 不直接相信当前 epoch 的 specific/remain。
- 维护关系的历史稳定性。
- 只有跨 epoch 稳定出现的关系才参与后续监督。
- 早期尽量只信 common，不信 specific/remain。

典型日志：

```text
TRRM diag | specific_kept:0/59 active:344 avg_mem:0.1053 avg_streak:1.00
```

解释：

- 早期 specific 一个都没保留，是正常现象。
- 这说明 TRRM 正在压住低质量 specific。
- 它符合诊断结论：早期只信 common。

但后续实验表现：

- TRRM 有解释性。
- TRRM 可以作为探索模块或消融。
- 但 SYSU all 没成为最强结果。
- 它不适合作为当前论文主线。

阶段结论：

```text
TRRM 证明了 specific/remain 噪声确实存在。
但它不是当前最强涨点来源。
```

## 阶段四：目标升级到一区

用户后来明确说：

```text
目标是一区和二区
```

并提出：

```text
一区不能只改 CRE。建议主线升级成：
Reliability-Guided Modality-invariant Feature Decoupling
```

这成为后续核心方向。

原因：

- confidence-aware CRE 属于排序策略改进，创新性偏小。
- TRRM 属于关系使用策略，仍然围绕 CRE 打转。
- 一区需要更强的结构性贡献。
- 所以主线应该从“哪些关系可靠”升级到“可靠关系如何塑造特征空间”。

## RG-MFD 定义

RG-MFD 全称：

```text
Reliability-Guided Modality-invariant Feature Decoupling
```

中文名称：

```text
可靠关系引导的模态不变特征解耦
```

它回答的问题：

```text
CRE 能找到可靠跨模态关系，但这些可靠关系应该监督哪一部分特征？
```

原始 AGW/CRE 的特征中混合了：

- 身份判别信息。
- RGB/IR 模态差异。
- 相机域差异。
- 噪声和不稳定因素。

RG-MFD 的目标：

- 从混合特征中分离出 shared feature。
- shared feature 用来承载跨模态共享身份信息。
- specific feature 用来吸收模态特有信息和干扰。

实现位置：

```text
models/agw.py
```

核心结构：

```python
shared_features = features * (1.0 + gate_scale * (gate - 0.5))
specific_features = features * (1.0 - gate)
```

含义：

- gate 由 2048 维 pooled feature 学出来。
- shared feature 不是简单裁剪，而是轻量残差式调制。
- specific feature 通过 `1.0 - gate` 接收模态相关成分。

## RG-MFD 的三个损失

### 1. 可靠关系对齐：rgmfd_rel_loss

目标：

```text
只用 CRE 挖出来的 reliable common relations，对齐 RGB/IR 的 shared feature。
```

具体做法：

- RGB shared feature 靠近对应 IR memory。
- IR shared feature 靠近对应 RGB memory。
- 使用 cosine alignment loss。
- 不盲目对齐所有样本，只对齐可靠关系。

意义：

```text
关系可靠性不是只用于 loss 权重，而是直接指导模态不变子空间学习。
```

### 2. 解耦正交：rgmfd_orth_loss

目标：

```text
让 shared feature 和 specific feature 尽量不重叠。
```

意义：

- shared 更专注身份。
- specific 更负责模态差异。
- 减少模态噪声污染身份表征。

### 3. gate 防塌缩：rgmfd_gate_loss

目标：

```text
防止 gate 全部偏向 shared 或全部偏向 specific。
```

意义：

- 避免解耦分支退化。
- 保持 shared/specific 两部分都有有效容量。

## RDL 定义和定位

用户问过：

```text
rdl 是什么
```

RDL 可以理解为：

```text
Relation-guided Dynamic Loss weighting
```

中文：

```text
关系可靠性引导的动态损失加权
```

它做的事情：

- 根据 CRE 当前关系可靠性、覆盖率、稳定性，动态调整 common/specific/remain 的训练权重。
- common 可靠时适当增强。
- specific/remain 不稳定时适当压低。

注意：

- 最初曾怀疑 RDL 会压住 SYSU all 的 Rank-1。
- 后来读取 best 日志后修正判断。
- 当前 SYSU all 最强配置里 `enable_rdl=1`。
- 关键不是单纯开关 RDL，而是 `rgmfd_start_epoch=20` 延后启动 RG-MFD 辅助损失。

## 学习率衰减讨论

用户问：

```text
学习率衰减可以设置为三个阶段，在 95 的时候再衰减有效果吗
```

当时判断：

- 当前默认 milestone 是 `[30, 70]`。
- 70 之后学习率已经到 `3e-6`。
- 如果 95 再衰减，会到 `3e-7`。
- 对已经稳定的模型，可能只是进一步冻住，不一定能推 Rank-1。

后续实验也支持：

- SYSU all 最强出现在 epoch 109。
- 但核心涨点来自 RG-MFD/RDL/delayed start，而不是额外 milestone。

建议：

```text
先不要把 95 衰减作为主变量。
优先做 rgmfd_start_epoch、RDL、rel_weight 的消融。
```

## indoor 和 all 差异讨论

用户问过：

```text
为什么 indoor 效果很好，涨点两个，all 反而不行？
```

当时基于日志判断：

- indoor gallery 只含 cam1/cam2，场景相对简单。
- all gallery 包含 cam1/cam2/cam4/cam5，跨相机和室外干扰更强。
- all-search 更依赖 top-1 排序尖锐性。
- RG-MFD 早期或过强关系约束，可能提升 mAP 但不一定提升 Rank-1。

后续日志进一步修正：

- 初版 `RG-MFD + RDL + rel=0.2 + start=0`：
  - SYSU all：Rank-1 69.29，mAP 67.13。
  - SYSU indoor：Rank-1 78.45，mAP 82.19。
- 后来 `RG-MFD + RDL + rel=0.1 + start=20 + pretrain`：
  - SYSU all：Rank-1 71.23，mAP 68.41。

关键经验：

```text
all 不是 RG-MFD 无效，而是需要更温和、更晚启动的可靠关系解耦。
```

## 最新实验汇总

完整表格见：

```text
experiment_log_summary.md
```

扫描日志数量：

```text
49
```

### SYSU all

当前最强：

```text
Run:        2026年05月25日-18:11:48(best)
方法:       RG-MFD + RDL
Best epoch: 109
Rank-1:     71.23
Rank-10:    96.26
Rank-20:    98.84
mAP:        68.41
mINP:       55.58
日志:       logs/sysu/all/2026年05月25日-18:11:48(best)/log/log.txt
```

关键参数：

```text
--dataset sysu
--search-mode all
--stage1-epoch 20
--milestone 30 70
--lr 0.0003
--cre-sample-rate 1.0
--enable-trrm 0
--enable-remain-gate 0
--enable-rgmfd 1
--rgmfd-start-epoch 20
--rgmfd-rel-weight 0.1
--rgmfd-orth-weight 0.05
--rgmfd-gate-reg-weight 0.01
--enable-rdl 1
--model-path /root/saved_pretrain_sysu_resnet/model_20.pth
```

重要对照：

```text
Base/CRE old best:
logs/sysu/all/2026年05月22日-17:31:06（best）/log/log.txt
Rank-1: 69.60
mAP:    66.41

RG-MFD without RDL:
logs/sysu/all/2026年05月25日-18:17:50/log/log.txt
Rank-1: 70.82
mAP:    68.29

RG-MFD + RDL, start=0, rel=0.2:
logs/sysu/all/2026年05月25日-09:25:43(对应indoor涨2)/log/log.txt
Rank-1: 69.29
mAP:    67.13
```

结论：

```text
RG-MFD 本身有效。
RDL 在 delayed start 配置下进一步提升 SYSU all。
rgmfd_start_epoch=20 是关键消融点。
rel_weight 从 0.2 降到 0.1 后更稳。
```

### SYSU indoor

当前最高日志：

```text
Run:        2026年05月26日-09:43:30
方法:       RG-MFD
Best epoch: 99
Rank-1:     79.20
mAP:        82.63
状态:       partial
日志:       logs/sysu/indoor/2026年05月26日-09:43:30/log/log.txt
```

旧 indoor baseline：

```text
logs/sysu/indoor/2026:04:20-03:26:53/log/log.txt
Rank-1: 76.03
mAP:    80.04
```

此前 indoor best 标注日志：

```text
logs/sysu/indoor/2026年05月25日-09:26:00（indoorbest）/log/log.txt
Rank-1: 78.45
mAP:    82.19
方法:   RG-MFD + RDL
```

结论：

```text
SYSU indoor 明显受益于 RG-MFD。
当前最高 79.20 仍是 partial，需要确认完整 120 epoch 后是否保持。
RDL 对 indoor 未必稳定，RG-MFD 单开目前更高。
```

### RegDB

当前汇总中 RegDB 最强仍是历史 Base/CRE：

```text
RegDB t2v:
Run:    2026:04:20-09:27:28_1
Rank-1: 86.17
mAP:    80.30

RegDB v2t:
Run:    2026:04:20-13:42:40_1
Rank-1: 87.14
mAP:    82.31
```

注意：

- 目前 RG-MFD 的 RegDB 日志存在，但多为 no_eval。
- 不能据此判断 RG-MFD 在 RegDB 无效。
- 需要补完整有效训练和测试日志。

### LLCM

当前 LLCM 最强：

```text
LLCM t2v:
Run:    2026年05月25日-17:23:04
方法:   RG-MFD
Rank-1: 47.44
mAP:    54.04

LLCM v2t:
Run:    2026年05月25日-17:39:33
方法:   RG-MFD
Rank-1: 56.05
mAP:    59.82
```

对照：

```text
LLCM t2v Base/CRE:
Rank-1: 47.10
mAP:    53.32

LLCM v2t 部分 Base/CRE 日志较弱或 partial，当前最强为 RG-MFD。
```

结论：

```text
LLCM 上 RG-MFD 有正向信号，但涨幅没有 SYSU 明显。
需要补充稳定完整实验和消融。
```

## 代码修改脉络

主要涉及文件：

```text
main.py
wsl.py
task/train.py
models/agw.py
models/__init__.py
sysu.sh
regdb.sh
llcm.sh
```

### main.py

新增或使用过的参数类型：

- CRE confidence 参数。
- CRE sample 参数。
- TRRM 参数。
- remain gate 参数。
- RDL 参数。
- RG-MFD 参数。

RG-MFD 相关：

```text
--enable-rgmfd
--rgmfd-reduction
--rgmfd-gate-scale
--rgmfd-start-epoch
--rgmfd-rel-weight
--rgmfd-orth-weight
--rgmfd-gate-reg-weight
--rgmfd-gate-target
```

RDL 相关：

```text
--enable-rdl
--rdl-warmup
--rdl-coverage-weight
--rdl-stability-weight
--rdl-common-boost
--rdl-specific-min
--rdl-remain-min
--rdl-remain-ratio-weight
```

### wsl.py

CRE 和 CMA 相关修改集中在这里。

历史上加入过：

- confidence-aware CRE。
- CRE 诊断。
- TRRM memory/streak 逻辑。
- remain gate 相关过滤。

当前结论：

```text
CRE 诊断非常有价值，应保留。
confidence-aware CRE 不适合作主线。
TRRM 可保留作消融或辅助探索。
```

### models/agw.py

加入 RG-MFD 模块。

核心作用：

- 在 GAP 后的 2048 维特征上做 gate。
- 输出 shared feature 和 specific feature。
- 训练时返回 `rgmfd_pack` 给 loss 使用。

关键对象：

```text
RGMFD
shared_BN
specific_BN
shared_features
specific_features
gate
```

### task/train.py

加入了：

- `_rgmfd_regularization_losses`
- `rgmfd_orth_loss`
- `rgmfd_gate_loss`
- `rgmfd_rel_loss`
- RDL 动态权重计算
- CRE/RDL 诊断日志输出

关键逻辑：

```text
epoch >= rgmfd_start_epoch 时才启用 RG-MFD 辅助损失。
```

这一点后来被证明很重要：

```text
SYSU all best 使用 rgmfd_start_epoch=20。
```

## 重要判断修正

### 修正一：最强 SYSU all 不是 RDL=0

曾经根据用户贴出的命令判断：

```text
enable_rdl=0 + enable_rgmfd=1 可能已经足够强。
```

后来用户指出 best 日志是：

```text
logs/sysu/all/2026年05月25日-18:11:48(best)/log/log.txt
```

读取日志后确认：

```text
enable_rdl=1
rgmfd_start_epoch=20
model_path=/root/saved_pretrain_sysu_resnet/model_20.pth
```

因此修正为：

```text
当前 SYSU all 最强是 RG-MFD + RDL + delayed start。
```

### 修正二：RDL 不是简单负作用

早期 `RG-MFD + RDL + start=0 + rel=0.2` 在 all 上 Rank-1 不强，但 indoor 涨。

后来 `RG-MFD + RDL + start=20 + rel=0.1` 在 SYSU all 最强。

所以 RDL 的定位应是：

```text
RDL 有潜力，但需要和 delayed RG-MFD loss 配合。
它不是主贡献本体，更像可靠性自适应训练策略。
```

### 修正三：RG-MFD 是主贡献，不是附属小模块

最开始想法是改 CRE。

现在应改成：

```text
CRE 提供可靠关系。
RG-MFD 利用可靠关系学习模态不变身份子空间。
RDL 辅助动态调整关系监督强度。
```

这样更像一区论文框架。

## 当前推荐论文叙事

推荐主标题方向：

```text
Reliability-Guided Modality-invariant Feature Decoupling for Weakly Supervised Visible-Infrared Person Re-identification
```

核心贡献可以写成三点：

1. 提出可靠关系引导的模态不变特征解耦框架 RG-MFD，将 CRE 发现的可靠跨模态关系从样本关系监督提升到特征子空间约束。
2. 设计 shared-specific feature decoupling gate，将混合视觉特征分解为模态共享身份表征和模态特定表征，并通过正交约束和 gate 正则防止信息纠缠与塌缩。
3. 引入关系可靠性自适应训练策略，在不同训练阶段根据关系覆盖率和稳定性调节跨模态监督强度，缓解早期伪关系噪声。

需要谨慎表述：

- confidence-aware CRE 不宜作为核心贡献。
- TRRM 可作为 explored variant 或 appendix 消融。
- RDL 是否作为正式贡献，取决于后续更多数据集是否稳定。

## 当前最应该补的实验

### 1. SYSU indoor 完整跑完

当前最高 indoor 日志是 partial：

```text
logs/sysu/indoor/2026年05月26日-09:43:30/log/log.txt
Rank-1: 79.20
mAP:    82.63
```

需要确认：

- 是否完整到 epoch 119。
- 最终 best 是否仍保持。
- 与 `RG-MFD + RDL + start=20` 对比是否稳定。

### 2. RegDB 补有效 RG-MFD 日志

当前 RegDB 最强还是 Base/CRE。

需要补：

```text
RegDB t2v: RG-MFD
RegDB t2v: RG-MFD + RDL + start=20
RegDB v2t: RG-MFD
RegDB v2t: RG-MFD + RDL + start=20
```

建议优先参数：

```text
--enable-rgmfd 1
--rgmfd-rel-weight 0.1
--rgmfd-orth-weight 0.05
--rgmfd-gate-reg-weight 0.01
```

RDL 是否开启建议分两组：

```text
--enable-rdl 0
--enable-rdl 1 --rgmfd-start-epoch 20
```

### 3. LLCM 补 RDL delayed start 完整日志

已有 LLCM RG-MFD 正向结果，但 RDL delayed start 多为 partial 或效果不清楚。

需要补完整：

```text
LLCM t2v: RG-MFD + RDL + start=20 + rel=0.1
LLCM v2t: RG-MFD + RDL + start=20 + rel=0.1
```

### 4. 核心消融表

建议必做消融：

```text
Base/CRE
+ confidence-aware CRE
+ TRRM
+ RG-MFD
+ RG-MFD + RDL
+ RG-MFD + RDL + delayed start
```

RG-MFD 内部消融：

```text
w/o rgmfd_rel_loss
w/o rgmfd_orth_loss
w/o rgmfd_gate_loss
rgmfd_start_epoch=0
rgmfd_start_epoch=20
rgmfd_rel_weight=0.0
rgmfd_rel_weight=0.05
rgmfd_rel_weight=0.1
rgmfd_rel_weight=0.2
```

### 5. 参数敏感性

优先只扫：

```text
rgmfd_rel_weight: 0.0 / 0.05 / 0.1 / 0.2
rgmfd_start_epoch: 0 / 20
enable_rdl: 0 / 1
```

暂不优先：

```text
milestone 95
过多 gate_scale/reduction 搜索
大范围 lr 搜索
```

## 当前建议命令模板

### SYSU all 当前 best 方向

```bash
python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestone 30 70 \
--lr 0.0003 \
--device 1 \
--search-mode all \
--cre-sample-rate 1.0 \
--enable-trrm 0 \
--enable-remain-gate 0 \
--enable-rgmfd 1 \
--rgmfd-start-epoch 20 \
--rgmfd-rel-weight 0.1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--enable-rdl 1 \
--model-path /root/saved_pretrain_sysu_resnet/model_20.pth
```

### SYSU indoor 当前更强方向

```bash
python3 main.py \
--dataset sysu \
--debug wsl \
--save-path sysu_agw \
--arch resnet \
--stage1-epoch 20 \
--milestone 30 70 \
--lr 0.0003 \
--device 1 \
--search-mode indoor \
--cre-sample-rate 1.0 \
--enable-trrm 0 \
--enable-remain-gate 0 \
--enable-rgmfd 1 \
--rgmfd-rel-weight 0.1 \
--rgmfd-orth-weight 0.05 \
--rgmfd-gate-reg-weight 0.01 \
--enable-rdl 0
```

注意：

- indoor 当前最高是 partial，需要完整验证。
- 为公平论文实验，最终 all/indoor 是否使用同一配置需要谨慎。
- 如果同一配置表现略低但更统一，可以论文主表用统一配置，附表报告 best variant。

## 当前风险点

### 1. SYSU 结果强，但 RegDB 尚未闭环

一区目标不能只靠 SYSU。

必须补：

- RegDB t2v
- RegDB v2t
- LLCM t2v
- LLCM v2t

### 2. RDL 在不同协议上可能不稳定

现象：

- SYSU all：RDL + delayed start 最强。
- SYSU indoor：当前 RDL=0 partial 更强。
- LLCM 的 RDL delayed start 日志还不完整。

所以论文里 RDL 要么作为辅助策略，要么需要更多稳定证据。

### 3. pretrain 影响需要说明

SYSU all best 使用：

```text
/root/saved_pretrain_sysu_resnet/model_20.pth
```

需要确认论文实验中：

- baseline 是否也使用同等预训练。
- 对比是否公平。
- 所有消融是否从同一 stage1 模型开始。

### 4. partial 日志不能直接作为最终主表

例如：

```text
SYSU indoor 2026年05月26日-09:43:30
```

当前 Rank-1 最高，但还标记为 partial。最终论文表必须用完整训练日志，除非确认训练已经结束但解析逻辑误判。

## 当前阶段结论

1. confidence-aware CRE 有尝试价值，但实验证明不是主涨点。
2. CRE 诊断非常关键，它证明了 common 后期高准确、高覆盖、高稳定。
3. TRRM 逻辑合理，但当前不是最强主线。
4. RG-MFD 是目前最有论文价值的贡献点。
5. RDL 不能简单放弃，它在 SYSU all best 中是有效组成部分。
6. delayed start 很关键，尤其是 `rgmfd_start_epoch=20`。
7. SYSU all 已经达到强结果，下一步重点是补全 RegDB/LLCM 和消融。

## 下一步执行清单

优先级从高到低：

1. 完整确认 SYSU indoor `RG-MFD, RDL=0, rel=0.1` 是否稳定到 120 epoch。
2. 跑 RegDB t2v/v2t 的 RG-MFD 和 RG-MFD+RDL delayed start。
3. 跑 LLCM t2v/v2t 的 RG-MFD+RDL delayed start 完整日志。
4. 做 SYSU all 消融：Base、RG-MFD、RG-MFD+RDL、start=0/20。
5. 做 RG-MFD 内部消融：w/o rel、w/o orth、w/o gate。
6. 只扫少量 `rgmfd_rel_weight`，不要过度调参。
7. 根据多数据集结果决定 RDL 是否写成正式贡献，还是写成辅助训练策略。

## 可直接用于论文的中文表述草稿

原始 CRE 能够发现跨模态可靠关系，但这些关系仅作为样本级伪监督使用，无法显式约束特征中的模态共享信息与模态特定信息。为此，我们提出可靠关系引导的模态不变特征解耦模块 RG-MFD。该模块通过轻量级门控机制将全局视觉表征分解为共享身份表征和模态特定表征，并利用 CRE 挖掘出的高置信 common relations 对共享表征进行跨模态对齐。同时，通过共享/特定特征正交约束和门控正则化，减少模态相关噪声对身份判别空间的干扰。实验表明，该策略能够在 SYSU-MM01 all-search 上显著提升 Rank-1 和 mAP。

## 可直接用于英文方法名

```text
Reliability-Guided Modality-invariant Feature Decoupling
```

可选缩写：

```text
RG-MFD
```

可选论文标题：

```text
Reliability-Guided Modality-invariant Feature Decoupling for Weakly Supervised Visible-Infrared Person Re-identification
```

## 文件索引

对话总结：

```text
conversation_history_2026-05-23.md
```

实验汇总：

```text
experiment_log_summary.md
```

当前 SYSU all best：

```text
logs/sysu/all/2026年05月25日-18:11:48(best)/log/log.txt
```

当前 SYSU indoor 最高 partial：

```text
logs/sysu/indoor/2026年05月26日-09:43:30/log/log.txt
```

旧 SYSU all baseline best：

```text
logs/sysu/all/2026年05月22日-17:31:06（best）/log/log.txt
```

旧 SYSU indoor baseline：

```text
logs/sysu/indoor/2026:04:20-03:26:53/log/log.txt
```

## 最终备注

当前研究已经从“能不能改 CRE 涨点”进入“如何把可靠关系转化为更强特征学习框架”的阶段。后续不要再把主精力放在 CRE 排序微调上，应围绕 RG-MFD 做多数据集验证和消融闭环。
