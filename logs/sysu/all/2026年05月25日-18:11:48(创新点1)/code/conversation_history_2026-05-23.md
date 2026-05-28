# WSL_ReID 改进讨论历史导出

导出日期：2026-05-23  
工作目录：`/root/WSL_ReID`  

## 导出说明

本文件整理当前会话中可见的历史对话、实验现象、代码修改和阶段性结论。

注意：本轮会话中途发生过上下文压缩，因此压缩前的部分助手回复无法逐字恢复；这些内容已根据保留下来的上下文摘要、代码修改记录和后续对话进行详细还原。用户原始问题尽量按可见内容逐条保留。

系统消息、开发者约束和内部指令未导出。

## 研究背景

项目：弱监督可见光-红外行人重识别，位于 `/root/WSL_ReID`。

参考文献：

`/root/WSL_ReID/paper/Zhang 等 - 2025 - Weakly supervised visible-infrared person re-identification via heterogeneous expert collaborative c.pdf`

论文方法主线：

- HEL：Heterogeneous Expert Learning
- CRE：Cross-modality Relation Exploration
- CCL：Collaborative Contrastive Learning

当前讨论重点：

- 原始 CRE 使用 Count Priority Selection。
- 用户希望改进代码以提升 SYSU/LLCM/RegDB 指标，目标从“二区”进一步讨论到“一区”。
- 已经尝试将 Count Priority CRE 升级为 confidence-aware CRE。
- 后续发现 CRE 后期接近饱和，`common` 关系很准，主要噪声来自 `specific/remain`。
- 因此进一步尝试 TRRM：Temporal Reliability Relation Memory，用历史稳定性过滤 `specific/remain`。

## 对话时间线

### 1. 文献改进方向

用户：

> 这篇文献还有什么地方可以改进涨点的，目标二区

上下文：

- 活动文件：`llcm.sh`
- 打开的文件：`llcm.sh`、`sysu.sh`、`main.py`、`regdb.sh`、`monitor.py`
- 用户提供了论文 PDF 路径。

助手回复摘要：

- 建议先从 CRE 入手，因为它是论文核心贡献之一，而且相对容易做代码级增强。
- 初步判断可以尝试：
  - 从 Count Priority CRE 改为 confidence-aware CRE。
  - 给 CRE 加置信度过滤和诊断指标。
  - 观察 `common/specific/remain` 的准确率、覆盖率、稳定性。
- 认为如果只是小改 CRE，比较适合二区方向；如果目标一区，需要更强的结构性贡献。

### 2. 要求直接修改

用户：

> 直接改

助手操作摘要：

- 开始在代码中实现 confidence-aware CRE。
- 主要修改文件：
  - `wsl.py`
  - `task/train.py`
  - `main.py`
- 在 `CMA` 中增加 CRE 置信度相关参数。
- 新增 `_get_confident_label`，在候选匹配中综合：
  - count
  - expert probability
  - margin
  - entropy confidence
  - prototype similarity
- 保留 `--cre-confidence 0` 作为回退到原始 Count Priority CRE 的开关。

### 3. CRE 开关含义

用户：

> --cre-confidence 0这是什么

助手回复摘要：

- `--cre-confidence 0` 表示关闭 confidence-aware CRE，回到原始 Count Priority CRE。
- `--cre-confidence 1` 表示启用置信度感知 CRE。
- 这个参数用于做消融实验：
  - `0`：原始策略，更稳。
  - `1`：新策略，用于测试是否涨点。

### 4. 第一阶段指标问题

用户：

> 训练第一阶段指标一样吗

助手回复摘要：

- 如果第一阶段没有真正使用 CRE 的匹配结果，那么理论上指标应基本一致。
- 但如果第一阶段仍然提前抽特征、更新 CRE 关系或日志逻辑有变化，可能出现轻微差异。
- 建议看代码里 phase1 是否调用了 `cma.get_label()` 或相关关系构造逻辑。

### 5. CRE pairs 数字解释

用户：

> CRE pairs | r2i:386 i2r:372 common:359 specific:12 remain:18数字什么意思

助手回复摘要：

- `r2i:386`：RGB 到 IR 的匹配关系数量。
- `i2r:372`：IR 到 RGB 的匹配关系数量。
- `common:359`：双向一致的可靠关系。
- `specific:12`：只在一侧成立或关系不完全一致的模态特定关系。
- `remain:18`：剩余不确定关系。
- 结论：`common` 占比很高，CRE 关系已经比较稳定。

### 6. CRE 已经很好还能怎么改

用户：

> 那这不是几乎很好了吗，还能怎么改进

助手回复摘要：

- 如果 `common` 已经占绝大多数，继续调 CRE 的排序策略可能收益有限。
- 更值得关注：
  - `specific/remain` 是否噪声大。
  - CRE 是否后期饱和。
  - 特征表示是否还有结构性提升空间。
- 建议加诊断指标，不只看数量，还要看伪标签准确率、覆盖率和稳定性。

### 7. 日志目录 mode 问题

用户：

> 日志目录下无论怎么农只有all，给indoor也是只有all

助手处理摘要：

- 检查日志路径构造逻辑。
- 判断问题与 `sysu.sh` 或参数传递有关。
- 后续已修正过日志目录 mode 保存逻辑，使 `all/indoor` 能正确出现在日志目录中。

### 8. 用户恢复代码并要求只改 CRE

用户：

> 我恢复了代码重新对代码进行修改把原来的 Count Priority CRE 升级成了置信度感知 CRE，不作其他修改

助手操作摘要：

- 重新基于当前代码修改。
- 尽量只动 CRE 相关逻辑，不引入其他模块。
- 新增 confidence-aware CRE，但保留原始 Count Priority 作为回退。
- 新增 CRE 诊断指标，帮助判断是否饱和。

主要修改：

- `wsl.py`
  - `use_confident_cre`
  - `cre_sample_rate`
  - `cre_count_weight`
  - `cre_prob_weight`
  - `cre_margin_weight`
  - `cre_entropy_weight`
  - `cre_proto_weight`
  - `cre_min_margin_start`
  - `cre_min_margin_end`
  - `cre_margin_decay_epoch`
- `main.py`
  - 增加上述 CLI 参数。
- `task/train.py`
  - 输出 CRE 诊断日志。

### 9. 置信度 CRE 掉点

用户：

> 反而掉点

助手回复摘要：

- 判断 confidence-aware CRE 不一定适合当前代码/数据。
- 如果原始 Count Priority 已经很好，重新排序可能会破坏稳定关系。
- 建议使用 `--cre-confidence 0` 作为稳态基线。
- 后续应先加指标确认 CRE 是否已经饱和，而不是继续盲目加强 CRE。

### 10. 加指标看 CRE 是否饱和

用户：

> 如果 1 还是低，就说明这个数据/代码里 CRE 已经很饱和，置信度重排不适合当主涨点，直接回到 --cre-confidence 0 更稳 加指标看看饱和没

助手操作摘要：

- 在 CRE 中添加诊断指标。
- 利用数据集中可见的真实标签只做日志诊断，不参与训练决策。
- 输出：
  - `r2i` 伪标签准确率
  - `i2r` 伪标签准确率
  - `common` 准确率
  - `specific` 准确率
  - `remain` 准确率
  - `cov_c`
  - `cov_all`
  - `stable_c`

新增日志格式：

```text
CRE diag | r2i:xx.xx%(a/b) i2r:xx.xx%(a/b) common:xx.xx%(a/b) specific:xx.xx%(a/b) remain:xx.xx%(a/b) cov_c:xx.xx% cov_all:xx.xx% stable_c:xx.xx%
```

### 11. 早期 CRE 诊断结果

用户提供日志：

```text
Time: 2026-05-22 17:28:26 | start phase2 from epoch 0
extracting rgb features
extracting ir features
get match labels
confident CRE rgb: candidates=5245, matches=310, min_margin=0.0000
confident CRE ir: candidates=3655, matches=233, min_margin=0.0000
CRE diag | r2i:54.84%(170/310) i2r:59.66%(139/233) common:95.50%(106/111) specific:33.33%(20/60) remain:32.16%(55/171) cov_c:28.10% cov_all:85.82% stable_c:0.00%
```

助手解释摘要：

- `common` 虽然只有 111 对，但准确率很高，95.50%。
- `r2i/i2r` 整体准确率只有 55% 到 60%，说明单向匹配噪声不少。
- `specific/remain` 准确率只有约 33%，不适合直接强监督。
- `cov_c` 只有 28%，说明早期可靠关系覆盖不足。
- 结论：早期应该更信 `common`，谨慎使用 `specific/remain`。

### 12. 覆盖率问题

用户：

> CRE diag | r2i:57.67%(188/326) i2r:63.98%(151/236) common:95.04%(115/121),覆盖不高？

助手解释摘要：

- 是的，早期 `common` 覆盖不高。
- 这不是坏事，而是说明可靠关系少但准。
- 如果强行扩大覆盖，会引入大量错误伪标签。
- 早期策略应偏保守，后期再逐步放开。

### 13. 后续 CRE 诊断变好

用户提供日志：

```text
get match labels
confident CRE rgb: candidates=2230, matches=360, min_margin=0.0000
confident CRE ir: candidates=2698, matches=331, min_margin=0.0000
CRE diag | r2i:89.72%(323/360) i2r:88.52%(293/331) common:96.22%(280/291)
```

助手解释摘要：

- 后期 CRE 明显变准。
- `r2i/i2r` 已接近 90%。
- `common` 仍保持 96% 左右。
- 说明 CRE 后期基本稳定，继续改排序策略收益会变小。

### 14. 后期 CRE 接近饱和

用户提供日志：

```text
CRE diag | r2i:93.21%(357/383) i2r:91.62%(339/370) common:94.68%(338/357) specific:69.23%(9/13) remain:22.22%(4/18) cov_c:90.38% cov_all:98.23% stable_c:98.60%
```

助手解释摘要：

- `common` 覆盖已超过 90%，稳定性接近 99%。
- CRE 后期已经非常饱和。
- `remain` 仍然很差，只有 22.22%。
- `specific` 数量很少，虽然准确率变高，但统计量不足。
- 结论：继续在 CRE 排序上做文章，不适合作为主要涨点。

### 15. 用户追问例子

用户：

> 比如呢

助手回复摘要：

- 提出几个可尝试方向：
  - Relation Reliability Memory：记录关系跨 epoch 是否稳定。
  - 对 `common/specific/remain` 采用不同损失权重。
  - 晚期才启用 `specific/remain`。
  - 引入局部特征或频域特征来提升表示能力。
  - 做 feature-level 模块，而不是继续只改 CRE。

### 16. 目标二区怎么改

用户：

> 目标二区怎么改

助手回复摘要：

- 二区目标可以不用大改网络，做一个稳健、可解释的小模块即可。
- 推荐方向：
  - CRE 饱和诊断 + 关系可靠性建模。
  - 把 `specific/remain` 从硬监督改成弱监督或延迟监督。
  - 增加跨 epoch 稳定性筛选。
- 论文表述可以包装成：
  - Noise-aware Cross-modality Relation Learning
  - Temporal Relation Reliability
  - Reliability-guided CRE

### 17. Epoch 103 结果

用户提供日志：

```text
confident CRE rgb: candidates=452, matches=387, min_margin=0.0000
confident CRE ir: candidates=1520, matches=380, min_margin=0.0000
CRE diag | r2i:92.25%(357/387) i2r:93.16%(354/380) common:93.63%(353/377) specific:42.86%(3/7) remain:40.00%(2/5) cov_c:95.44% cov_all:98.48% stable_c:100.00%
=================================================
Epoch: 103;Time: 2026-05-22 23:23:01;Setting: logs/sysu/all/2026年05月22日-17:28:13
e_lr: 3.0000000000000005e-06
r2r_id_loss: 0.12191846;  i2i_id_loss: 0.10937478;
tri_loss_rgb: 0.0070283683;  tri_loss_ir: 0.0072533917;
r2i_cmo_loss: 0.23855193;  i2r_cmo_loss: 0.223436;
rgb_cross_loss: 0.023148842;  ir_cross_loss: 0.0112484535;
weak_r2c_loss: 0.0023323023;
R1:0.6881;   R10:0.9530;  R20:0.9838;  mAP:0.6586;  mINP:0.5267
                   Best_R1: 0.6937;    Best mAP: 0.6632
=================================================
```

助手解释摘要：

- CRE 关系已经接近稳定。
- 最终指标只小幅变化，说明 confidence-aware CRE 本身不是强涨点。
- 当前表现更像“调参式改进”，不太够支撑强二区贡献。

### 18. 用户认为二区不太行

用户：

> 这个二区感觉不太性

助手回复摘要：

- 同意：如果只有 confidence-aware CRE，创新性和涨点都偏弱。
- 后续要做更像模块的贡献，而不是只改候选排序。
- 建议进入“特征模块 + 关系可靠性”的方向。

### 19. Epoch 119 结果

用户提供日志：

```text
CRE diag | r2i:92.73%(357/385) i2r:92.19%(354/384) common:93.16%(354/380) specific:50.00%(1/2) remain:28.57%(2/7) cov_c:96.20% cov_all:98.48% stable_c:100.00%
=================================================
Epoch: 119;Time: 2026-05-23 00:23:01;Setting: logs/sysu/all/2026年05月22日-17:31:06
e_lr: 3.0000000000000005e-06
r2r_id_loss: 0.115146;  i2i_id_loss: 0.107803345;
tri_loss_rgb: 0.0063023567;  tri_loss_ir: 0.0064803218;
r2i_cmo_loss: 0.23600641;  i2r_cmo_loss: 0.2157625;
rgb_cross_loss: 0.021501813;  ir_cross_loss: 0.011265795;
weak_r2c_loss: 0.0036241335;
R1:0.6960;   R10:0.9550;  R20:0.9844;  mAP:0.6641;  mINP:0.5278
                   Best_R1: 0.6960;    Best mAP: 0.6644
=================================================
```

助手解释摘要：

- 相比旧结果有小涨，但涨幅有限。
- CRE 已经高覆盖、高稳定。
- 当前更适合做诊断和消融，不能作为主要论文贡献。

### 20. 特征模块设计

用户：

> 这个特征模块应该怎么设计呢

助手回复摘要：

- 建议不要简单堆 CBAM/SE/Transformer，因为容易缺乏针对性。
- 更适合 VI-ReID 的模块应围绕“模态差异”设计。
- 可设计：
  - Global identity branch
  - Local body-part branch
  - Modality-robust texture/frequency branch
  - Relation reliability-guided feature alignment
- 建议先做轻量模块，避免破坏原始框架。

### 21. 如果目标一区

用户：

> 如果是要目标一区呢

助手回复摘要：

- 如果目标一区，仅靠 confidence CRE 或普通注意力模块不够。
- 需要形成更完整的方法框架：
  - 关系层面：时序可靠性建模。
  - 特征层面：跨模态不变特征和模态特定特征解耦。
  - 损失层面：可靠关系引导的动态监督。
- 论文贡献应从“改 CRE”升级为：
  - Reliability-aware Weakly Supervised VI-ReID
  - Temporal Relation Reliability Learning
  - Cross-modal Relation Noise Suppression

### 22. 先怎么做

用户：

> 先怎么做

助手回复摘要：

- 建议第一步先做关系可靠性模块，而不是直接改大网络。
- 原因：
  - 当前诊断已经证明 `common` 很准，`specific/remain` 噪声大。
  - 先利用这个事实做一个稳健模块，改动小，容易验证。
- 具体第一步：
  - 记录每对跨模态关系在多个 epoch 中是否持续出现。
  - 对 `specific` 做历史稳定性过滤。
  - 对 `remain` 延迟启用或降权。
  - 增加日志看过滤前后数量。

### 23. 环境上下文

用户提供：

```text
cwd: /root/WSL_ReID
shell: bash
current_date: 2026-05-23
timezone: Etc/UTC
```

### 24. 用户要求先修改代码

用户：

> 先修改代码试试

助手操作：

- 实现轻量版 TRRM：Temporal Reliability Relation Memory。
- 不大改主干网络。
- 目标是验证“利用历史稳定性过滤 noisy relation”是否有效。

代码修改摘要：

#### `wsl.py`

新增成员：

```python
self.relation_memory = {}
self.relation_streak = {}
```

新增方法：

```python
update_relation_memory(common_dict, specific_dict, remain_dict, args)
filter_specific_relations(specific_dict, epoch, args)
```

核心逻辑：

- `common` 关系记为高可靠。
- `specific` 关系记为中等可靠。
- `remain` 关系记为低可靠。
- 跨 epoch 使用 momentum 更新关系记忆。
- 只有历史稳定的 `specific` 会进入训练。

#### `task/train.py`

在关系构造后加入：

```python
if getattr(args, 'enable_trrm', 1):
    raw_specific_num = len(specific_dict)
    memory_stats = cma.update_relation_memory(common_dict, specific_dict, remain_dict, args)
    specific_dict = cma.filter_specific_relations(specific_dict, epoch, args)
    logger('TRRM diag | specific_kept:{}/{} active:{} avg_mem:{:.4f} avg_streak:{:.2f}'.format(...))
```

同时：

- `specific` cross loss 加权。
- `remain` weak loss 延迟启用并降权。
- tensor 转换改成 `torch.as_tensor(..., dtype=torch.long, device=model.device)`，避免空关系时 dtype 出错。

#### `main.py`

新增参数：

```bash
--enable-trrm
--trrm-momentum
--trrm-specific-start
--trrm-specific-streak
--trrm-specific-threshold
--trrm-specific-weight
--trrm-remain-start
--trrm-remain-weight
```

默认值：

```bash
--enable-trrm 1
--trrm-momentum 0.8
--trrm-specific-start 20
--trrm-specific-streak 2
--trrm-specific-threshold 0.2
--trrm-specific-weight 0.3
--trrm-remain-start 60
--trrm-remain-weight 0.2
```

验证：

```bash
python3 -m py_compile main.py wsl.py task/train.py
python3 main.py --help | rg -- 'trrm|cre-confidence'
```

均通过。

助手当时的最终回复原文：

> 已经先改了一版轻量的 **TRRM：Temporal Reliability Relation Memory**，先不动主干特征模块，只把 CRE 产生的关系从“当轮直接用”改成“看历史稳定性再用”，更适合你现在这个现象：`common` 已经很饱和，`specific/remain` 才是噪声来源。
>
> 主要改了：
>
> - `wsl.py`：新增关系记忆 `relation_memory / relation_streak`
> - `task/train.py`：训练时先更新关系记忆，再过滤 `specific_dict`
> - `main.py`：新增 TRRM 参数
>
> 现在日志会多一行：
>
> ```text
> TRRM diag | specific_kept:x/y active:z avg_mem:... avg_streak:...
> ```
>
> 含义是当前 `specific` 关系里，有多少条通过了历史稳定性过滤。默认策略比较保守：
>
> ```bash
> --enable-trrm 1
> --trrm-specific-start 20
> --trrm-specific-streak 2
> --trrm-specific-threshold 0.2
> --trrm-specific-weight 0.3
> --trrm-remain-start 60
> --trrm-remain-weight 0.2
> ```
>
> 我也做了基础检查：
>
> ```bash
> python3 -m py_compile main.py wsl.py task/train.py
> python3 main.py --help | rg -- 'trrm|cre-confidence'
> ```
>
> 都通过了。
>
> 你可以先直接跑现在的 `sysu.sh`。如果掉点，优先试这组稍微放开的参数：
>
> ```bash
> --trrm-specific-start 10 --trrm-specific-weight 0.5 --trrm-remain-start 80
> ```
>
> 这版的目的不是硬拉 CRE 数字，而是验证：**抑制不稳定 specific/remain 后，R1/mAP 是否更稳更高**。如果这条能涨，后面就可以把它包装成“时序可靠跨模态关系学习”，比单纯 confidence CRE 更像二区以上的贡献点。

### 25. 用户给出 TRRM 第一轮日志

用户：

```text
CRE diag | r2i:54.84%(170/310) i2r:60.26%(141/234) common:96.40%(107/111) specific:35.59%(21/59) remain:32.76%(57/174) cov_c:28.10% cov_all:86.33% stable_c:0.00%
TRRM diag | specific_kept:0/59 active:344 avg_mem:0.1053 avg_streak:1.00
```

助手解释原文：

> 这个日志其实是 **TRRM 正常生效**，而且说明你这个数据早期确实不能乱用 `specific/remain`。
>
> 关键看这几项：
>
> ```text
> common:96.40%(107/111)
> specific:35.59%(21/59)
> remain:32.76%(57/174)
> cov_c:28.10%
> stable_c:0.00%
> TRRM specific_kept:0/59 avg_streak:1.00
> ```
>
> 意思是：
>
> - `common` 很准，96%，但早期覆盖低，只有 `28.10%`
> - `specific/remain` 很不准，只有 35% / 32%，直接拿去监督大概率掉点
> - `stable_c:0.00%` 和 `avg_streak:1.00` 说明这是关系记忆的第一轮，还没有历史稳定性
> - `specific_kept:0/59` 是对的，因为默认要求 `specific` 至少连续稳定出现，早期全部过滤掉
>
> 所以这轮现象不是坏事，反而证明我们加 TRRM 的方向是对的：**早期只信 common，不信 specific/remain**。
>
> 你现在先别急着改参数，继续看后面 epoch，重点看：
>
> ```text
> specific_kept:x/y
> avg_streak
> common acc
> specific acc
> R1 / mAP
> ```
>
> 理想走势是：
>
> ```text
> specific_kept:0/59
> specific_kept:3/60
> specific_kept:8/62
> specific_kept:15/65
> ```
>
> 也就是慢慢放一点稳定的 `specific`，不要一下全放。
>
> 如果到 phase2 第 20 epoch 以后还是一直：
>
> ```text
> specific_kept:0/xx
> ```
>
> 那说明过滤太严，可以把参数放松成：
>
> ```bash
> --trrm-specific-start 10 \
> --trrm-specific-streak 1 \
> --trrm-specific-threshold 0.15 \
> --trrm-specific-weight 0.3
> ```
>
> 但第一轮这个结果很好理解：`specific/remain` 现在太脏，TRRM 把它们按住了。先跑完整，看最终 `Best_R1 / Best mAP`。

### 26. 当前请求：导出历史对话

用户：

> 详细导出所有历史对话放到md文件

助手执行：

- 新建当前文件：

```text
/root/WSL_ReID/conversation_history_2026-05-23.md
```

## 当前代码状态摘要

当前已实现两个主要实验方向。

### Confidence-aware CRE

目的：

- 替代原始 Count Priority CRE。
- 让 CRE 匹配不只看出现次数，也看候选置信度。

核心评价：

- 后期 CRE 已经很饱和。
- confidence-aware CRE 有小幅提升，但不足以作为强贡献。
- `--cre-confidence 0` 是稳妥回退。

### CRE Diagnostics

目的：

- 判断 CRE 是否饱和。
- 判断 `common/specific/remain` 是否可靠。

关键结论：

- 早期：
  - `common` 准，但覆盖低。
  - `specific/remain` 噪声大。
- 后期：
  - `common` 高准确、高覆盖、高稳定。
  - CRE 排序本身已接近饱和。
  - `remain` 依旧不可靠。

### TRRM

目的：

- 通过历史稳定性过滤 noisy relation。
- 让训练早期只信可靠 `common`。
- 让稳定出现的 `specific` 后期再逐步参与。

当前日志说明：

```text
TRRM diag | specific_kept:0/59 active:344 avg_mem:0.1053 avg_streak:1.00
```

解释：

- 第一轮没有历史记忆，所以 `avg_streak=1.00`。
- 默认要求连续稳定，因此第一轮 `specific_kept=0` 是正常现象。
- 这说明 TRRM 正在抑制早期低质量 `specific`。

## 实验结果摘录

### 早期 CRE

```text
r2i:54.84%(170/310)
i2r:59.66%(139/233)
common:95.50%(106/111)
specific:33.33%(20/60)
remain:32.16%(55/171)
cov_c:28.10%
cov_all:85.82%
stable_c:0.00%
```

结论：

- `common` 可用。
- `specific/remain` 不适合直接强监督。

### 中后期 CRE

```text
r2i:93.21%(357/383)
i2r:91.62%(339/370)
common:94.68%(338/357)
specific:69.23%(9/13)
remain:22.22%(4/18)
cov_c:90.38%
cov_all:98.23%
stable_c:98.60%
```

结论：

- `common` 已经饱和。
- `remain` 仍然很差。
- CRE 排序不适合作为主涨点。

### SYSU all 当前较好结果

```text
Epoch: 119
R1:0.6960
R10:0.9550
R20:0.9844
mAP:0.6641
mINP:0.5278
Best_R1: 0.6960
Best mAP: 0.6644
```

结论：

- confidence-aware CRE 有小幅涨点。
- 但涨幅和创新性不足以单独支撑强二区。

## 当前建议

短期实验建议：

1. 先完整跑当前 TRRM 默认配置。
2. 观察 `specific_kept` 是否随 epoch 逐渐增加。
3. 如果 phase2 第 20 epoch 后仍然为 0，放松参数：

```bash
--trrm-specific-start 10 \
--trrm-specific-streak 1 \
--trrm-specific-threshold 0.15 \
--trrm-specific-weight 0.3
```

4. 如果 TRRM 涨点，可以写成“时序可靠跨模态关系学习”。
5. 如果 TRRM 不涨点，则需要进入特征模块方向，例如：
   - 局部身体区域特征
   - 频域模态不变特征
   - 模态特定/模态共享特征解耦
   - 可靠关系引导的特征对齐

## 可用于论文包装的表述

较适合二区：

- Reliability-aware Cross-modality Relation Exploration
- Temporal Relation Reliability Memory
- Noise-suppressed Weakly Supervised VI-ReID

如果冲一区，需要升级为完整框架：

- Dynamic Reliable Relation Learning
- Relation-guided Modality-invariant Feature Decoupling
- Temporal Cross-modal Relation Denoising
- Reliability-aware Collaborative Cross-modal Representation Learning

