import torch
import torch.nn.functional as F
from models import Model
from datasets import SYSU
import time
import numpy as np
import random
import copy
from collections import OrderedDict
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy,pha_unwrapping
from models import Model

def _filter_remain_weak_inputs(scores, labels, args):
    total = scores.shape[0]
    empty_stats = {
        'total': total,
        'kept': 0,
        'keep_ratio': scores.new_tensor(0.0),
        'score': scores.new_tensor(0.0),
        'target_prob': scores.new_tensor(0.0),
    }
    if total == 0:
        return scores, labels, empty_stats
    if not getattr(args, 'enable_remain_gate', 0):
        stats = empty_stats.copy()
        stats['kept'] = total
        stats['keep_ratio'] = scores.new_tensor(1.0)
        return scores, labels, stats

    valid_rows = labels.sum(dim=1) > 0
    scores = scores[valid_rows]
    labels = labels[valid_rows]
    if scores.shape[0] == 0:
        return scores, labels, empty_stats

    with torch.no_grad():
        probs = torch.softmax(scores, dim=1)
        label_mask = labels.bool()
        label_count = label_mask.float().sum(dim=1).clamp_min(1.0)
        target_prob = (probs * label_mask.float()).sum(dim=1) / label_count
        top_probs = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        if top_probs.shape[1] == 1:
            margin = top_probs[:, 0]
        else:
            margin = top_probs[:, 0] - top_probs[:, 1]
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        certainty = 1.0 - entropy / np.log(probs.shape[1])

        gate_score = (
            target_prob
            + getattr(args, 'remain_gate_margin_weight', 0.5) * margin
            + getattr(args, 'remain_gate_entropy_weight', 0.3) * certainty
        )
        keep_mask = (
            (target_prob >= getattr(args, 'remain_gate_min_prob', 0.0))
            & (margin >= getattr(args, 'remain_gate_min_margin', 0.0))
            & (certainty >= getattr(args, 'remain_gate_min_certainty', 0.0))
        )
        candidate_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            selected_indices = candidate_indices
        else:
            candidate_scores = gate_score[candidate_indices]
            keep_num = candidate_indices.numel()
            keep_ratio = getattr(args, 'remain_gate_keep_ratio', 1.0)
            if 0 < keep_ratio < 1:
                keep_num = min(keep_num, max(1, int(np.ceil(candidate_indices.numel() * keep_ratio))))
            max_num = getattr(args, 'remain_gate_max_num', 0)
            if max_num > 0:
                keep_num = min(keep_num, max_num)
            top_indices = candidate_scores.topk(k=keep_num, largest=True).indices
            selected_indices = candidate_indices[top_indices]

    filtered_scores = scores[selected_indices]
    filtered_labels = labels[selected_indices]
    kept = filtered_scores.shape[0]
    stats = {
        'total': total,
        'kept': kept,
        'keep_ratio': scores.new_tensor(kept / max(1, total)),
        'score': gate_score[selected_indices].mean() if kept > 0 else scores.new_tensor(0.0),
        'target_prob': target_prob[selected_indices].mean() if kept > 0 else scores.new_tensor(0.0),
    }
    return filtered_scores, filtered_labels, stats

def _relation_dynamic_weights(args, epoch, cma, common_dict, specific_dict, remain_dict):
    stats = {
        'enabled': bool(getattr(args, 'enable_rdl', 0)),
        'common_weight': 1.0,
        'specific_weight': 1.0,
        'remain_weight': 1.0,
        'reliability': 0.0,
        'common_coverage': len(common_dict) / max(1, args.num_classes),
        'common_stability': 0.0,
        'remain_ratio': 0.0,
    }

    current_common = set((int(r), int(i)) for r, i in common_dict.items())
    previous_common = getattr(cma, 'prev_rdl_common_pairs', None)
    if previous_common is not None and len(current_common) > 0:
        stats['common_stability'] = len(current_common & previous_common) / len(current_common)
    cma.prev_rdl_common_pairs = current_common

    total_relations = max(1, len(common_dict) + len(specific_dict) + len(remain_dict))
    stats['remain_ratio'] = len(remain_dict) / total_relations

    if not stats['enabled']:
        return stats

    coverage_weight = getattr(args, 'rdl_coverage_weight', 0.5)
    stability_weight = getattr(args, 'rdl_stability_weight', 0.5)
    norm = max(1e-6, coverage_weight + stability_weight)
    reliability = (
        coverage_weight * stats['common_coverage']
        + stability_weight * stats['common_stability']
    ) / norm
    reliability = max(0.0, min(1.0, reliability))
    warmup = max(1, getattr(args, 'rdl_warmup', 5))
    progress = min(1.0, max(0.0, (epoch + 1) / warmup))

    common_target = 1.0 + getattr(args, 'rdl_common_boost', 0.05) * reliability
    specific_min = getattr(args, 'rdl_specific_min', 0.7)
    specific_target = specific_min + (1.0 - specific_min) * reliability
    remain_min = getattr(args, 'rdl_remain_min', 0.5)
    remain_target = remain_min + (1.0 - remain_min) * reliability
    remain_target *= max(
        remain_min,
        1.0 - getattr(args, 'rdl_remain_ratio_weight', 0.25) * stats['remain_ratio'],
    )

    stats['reliability'] = reliability
    stats['common_weight'] = 1.0 + progress * (common_target - 1.0)
    stats['specific_weight'] = 1.0 + progress * (specific_target - 1.0)
    stats['remain_weight'] = 1.0 + progress * (remain_target - 1.0)
    return stats

def _cosine_alignment_loss(features, targets):
    if features.numel() == 0 or targets.numel() == 0:
        return features.new_tensor(0.0)
    return (1.0 - F.cosine_similarity(features, targets, dim=1)).mean()

def _weighted_cosine_alignment_loss(features, targets, weights=None):
    if features.numel() == 0 or targets.numel() == 0:
        return features.new_tensor(0.0)
    losses = 1.0 - F.cosine_similarity(features, targets, dim=1)
    if weights is None:
        return losses.mean()
    weights = weights.to(features.device).float().view(-1)
    if weights.numel() != losses.numel() or weights.sum() <= 1e-6:
        return losses.mean()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

def _rgmfd_regularization_losses(args, rgmfd_pack):
    if rgmfd_pack is None or not getattr(args, 'enable_rgmfd', 0):
        return {}

    losses = {}
    shared = F.normalize(rgmfd_pack['shared_features'], dim=1)
    specific = F.normalize(rgmfd_pack['specific_features'], dim=1)
    orth_weight = getattr(args, 'rgmfd_orth_weight', 0.0)
    if orth_weight > 0:
        losses['rgmfd_orth_loss'] = orth_weight * (shared * specific).sum(dim=1).pow(2).mean()

    gate_reg_weight = getattr(args, 'rgmfd_gate_reg_weight', 0.0)
    if gate_reg_weight > 0:
        gate_target = getattr(args, 'rgmfd_gate_target', 0.5)
        gate_mean = rgmfd_pack['gate'].mean(dim=1)
        losses['rgmfd_gate_loss'] = gate_reg_weight * (gate_mean - gate_target).pow(2).mean()
    return losses

def _add_loss_dict(total_loss, losses, meter):
    for name, loss in losses.items():
        if torch.isnan(loss).any():
            continue
        meter.update({name: loss.data})
        total_loss = total_loss + loss
    return total_loss

def train(args, model: Model, dataset, *arg):
    """
    单 Epoch 的训练函数。
    负责执行前向计算、损失函数的构筑与梯度反向传播。
    依据 --debug 模式不同（wsl, baseline, sl 等），走不通的训练（监督或弱监督跨模态）计算图。
    """
    epoch = arg[0]
    cma:CMA = arg[1]   # cma: Cross Modal Match Aggregation 实例 (负责打伪标签和存储特征池)
    logger = arg[2] 
    enable_phase1 = arg[3] # 是否处于第一阶段（Phase 1通常用于初始化/预热各模态分支）
    common_loss_weight = 1.0
    specific_loss_weight = 1.0
    remain_loss_weight = 1.0
    use_rgmfd = bool(getattr(args, 'enable_rgmfd', 0)) and bool(getattr(model.model, 'enable_rgmfd', False))
    eot_graph = None

    # ======================================================
    # 1. 弱监督学习 (WSL) 伪标签匹配与关系划分阶段
    # ======================================================
    if 'wsl' in args.debug or not enable_phase1:
        # 获取当前模型在整个训练集上的特征，并更新标签库与距离矩阵
        cma.extract(args, model, dataset)        
        
        # 获取模态内的局部重组/聚类标签映射
        rgb_labeling_dict, ir_labeling_dict = \
            dataset.train_rgb.relabel_dict, dataset.train_ir.relabel_dict
            
        # 根据特征距离度量，计算可见光到红外(r2i)、红外到可见光(i2r) 的伪标签匹配配对
        r2i_pair_dict, i2r_pair_dict = cma.get_label(epoch)
        
        # 将关系粗暴地划分为三类：一致(common), 唯一/冲突片段(specific), 剩余弱对应(remain)
        common_dict, specific_dict, remain_dict = {},{},{}
        i2r_specific_dict, r2i_specific_dict, r2i_remain_dict, i2r_remain_dict = {},{},{},{}
        
        # 1.1 遍历 rgb->ir 的匹配
        for r,i in r2i_pair_dict.items():
            if i in i2r_pair_dict.keys() and i2r_pair_dict[i] == r:
                # 互为最近邻：纳入高质量的“一致匹配”(common)
                common_dict[r] = i
            elif r not in i2r_pair_dict.values() and i not in i2r_pair_dict.keys():
                # 单向包含或独立：纳入“模态特有匹配”(specific)
                r2i_specific_dict[r] = i
                specific_dict[r] = i
            else:
                # 冲突或多对一：纳入“剩余模棱两可匹配”(remain)
                r2i_remain_dict[r] = i
                remain_dict[r] = i
        
        # 1.2 遍历 ir->rgb 的匹配 (过滤掉已在 common 中的)
        for i,r in i2r_pair_dict.items():
            if (r,i) in common_dict.items():
                continue
            elif r not in r2i_pair_dict.values() and i not in r2i_pair_dict.keys():
                i2r_specific_dict[i] = r
                specific_dict[r] = i
            else:
                i2r_remain_dict[i] = r
                remain_dict[r] = i

        if getattr(args, 'enable_eot_rr', 0):
            eot_graph = cma.build_eot_relations(args)
            if eot_graph is not None:
                r2i_pair_dict = eot_graph['r2i_pair_dict']
                i2r_pair_dict = eot_graph['i2r_pair_dict']
                common_dict = eot_graph['common_dict']
                specific_dict = eot_graph['specific_dict']
                remain_dict = eot_graph['remain_dict']
                eot_stats = eot_graph['stats']
                logger(
                    'EOT-RR diag | active:{} max_rel:{:.4f} mean_rel:{:.4f} '
                    'common:{}({:.4f}) specific:{}({:.4f}) remain:{}({:.4f})'.format(
                        eot_stats['active'],
                        eot_stats['max_rel'],
                        eot_stats['mean_rel'],
                        eot_stats['common_num'],
                        eot_stats['common_mean'],
                        eot_stats['specific_num'],
                        eot_stats['specific_mean'],
                        eot_stats['remain_num'],
                        eot_stats['remain_mean'],
                    )
                )

        diagnostics = cma.relation_diagnostics(
            r2i_pair_dict, i2r_pair_dict, common_dict, specific_dict, remain_dict
        )
        if diagnostics:
            logger(
                'CRE diag | r2i:{:.2f}%({}/{}) i2r:{:.2f}%({}/{}) '
                'common:{:.2f}%({}/{}) specific:{:.2f}%({}/{}) remain:{:.2f}%({}/{}) '
                'cov_c:{:.2f}% cov_all:{:.2f}% stable_c:{:.2f}%'.format(
                    diagnostics['r2i']['acc'] * 100,
                    diagnostics['r2i']['correct'],
                    diagnostics['r2i']['valid'],
                    diagnostics['i2r']['acc'] * 100,
                    diagnostics['i2r']['correct'],
                    diagnostics['i2r']['valid'],
                    diagnostics['common']['acc'] * 100,
                    diagnostics['common']['correct'],
                    diagnostics['common']['valid'],
                    diagnostics['specific']['acc'] * 100,
                    diagnostics['specific']['correct'],
                    diagnostics['specific']['valid'],
                    diagnostics['remain']['acc'] * 100,
                    diagnostics['remain']['correct'],
                    diagnostics['remain']['valid'],
                    diagnostics['common_coverage'] * 100,
                    diagnostics['all_coverage'] * 100,
                    diagnostics['common_stability'] * 100,
                )
            )

        rdl_stats = _relation_dynamic_weights(args, epoch, cma, common_dict, specific_dict, remain_dict)
        if rdl_stats['enabled']:
            common_loss_weight = rdl_stats['common_weight']
            specific_loss_weight = rdl_stats['specific_weight']
            remain_loss_weight = rdl_stats['remain_weight']
            logger(
                'RDL diag | rel:{:.3f} cov_c:{:.2f}% stable_c:{:.2f}% remain_r:{:.2f}% '
                'w_c:{:.3f} w_s:{:.3f} w_r:{:.3f}'.format(
                    rdl_stats['reliability'],
                    rdl_stats['common_coverage'] * 100,
                    rdl_stats['common_stability'] * 100,
                    rdl_stats['remain_ratio'] * 100,
                    common_loss_weight,
                    specific_loss_weight,
                    remain_loss_weight,
                )
            )

        if getattr(args, 'enable_trrm', 1):
            raw_specific_num = len(specific_dict)
            memory_stats = cma.update_relation_memory(common_dict, specific_dict, remain_dict, args)
            specific_dict = cma.filter_specific_relations(specific_dict, epoch, args)
            specific_loss_weight = getattr(args, 'trrm_specific_weight', 0.3)
            ramp_epochs = getattr(args, 'trrm_specific_ramp', 0)
            if ramp_epochs > 0:
                start_epoch = getattr(args, 'trrm_specific_start', 20)
                ramp_ratio = min(1.0, max(0.0, (epoch - start_epoch + 1) / ramp_epochs))
                specific_loss_weight *= ramp_ratio
            logger(
                'TRRM diag | specific_kept:{}/{} active:{} avg_mem:{:.4f} avg_streak:{:.2f}'.format(
                    len(specific_dict),
                    raw_specific_num,
                    memory_stats['active'],
                    memory_stats['avg_memory'],
                    memory_stats['avg_streak'],
                )
            )

        # 1.3 利用划分好的字典构建各个关系类型的稀疏连接矩阵 (rm)
        all_rm = torch.zeros((args.num_classes,args.num_classes)).to(model.device) # 总体连接矩阵
        common_rm = all_rm.clone()   # M_c : 互为最近邻的一对一关系
        specific_rm = all_rm.clone() # M_s : 特有对应关系
        remain_rm = all_rm.clone()   # M_w : 冲突的对应关系
        r2i_rm = all_rm.clone()      # M_{v->r}
        i2r_rm = all_rm.clone()      # M_{r->v}
        eot_reliability = None
        if eot_graph is not None:
            eot_reliability = torch.as_tensor(
                eot_graph['reliability'], dtype=torch.float32, device=model.device
            )

        def relation_weight(r, i):
            if eot_reliability is None:
                return 1.0
            return eot_reliability[int(r), int(i)]
        
        for r, i in common_dict.items(): 
            common_rm[r,i] += relation_weight(r, i)
        for r, i in specific_dict.items():
            specific_rm[r,i] += relation_weight(r, i)
        for r, i in r2i_pair_dict.items():
            r2i_rm[r,i] += relation_weight(r, i)
        for i, r in i2r_pair_dict.items():
            i2r_rm[i,r] += relation_weight(r, i)
        for r, i in remain_dict.items():
            remain_rm[r,i] += relation_weight(r, i)

        # 将一致矩阵叠加给特定矩阵，方便后续特定损失调用
        specific_rm = specific_rm + common_rm
        
        # 收集不同关联分类下匹配成功的具体身份 ID 列表
        matched_rgb, matched_ir = list(r2i_pair_dict.keys()), list(i2r_pair_dict.keys())
        common_matched_rgb, common_matched_ir = list(common_dict.keys()), list(common_dict.values())
        specific_matched_rgb, specific_matched_ir = list(specific_dict.keys()), list(specific_dict.values())
        remain_matched_rgb, remain_matched_ir = list(remain_dict.keys()), list(remain_dict.values())
        all_matched_rgb = list(set(common_matched_rgb + specific_matched_rgb + remain_matched_rgb))
        all_matched_ir = list(set(common_matched_ir + specific_matched_ir + remain_matched_ir))
        
        # 转换到设备张量 Tensor 以便 GPU 加速
        matched_rgb = torch.as_tensor(matched_rgb, dtype=torch.long, device=model.device)
        matched_ir = torch.as_tensor(matched_ir, dtype=torch.long, device=model.device)
        common_matched_rgb = torch.as_tensor(common_matched_rgb, dtype=torch.long, device=model.device)
        common_matched_ir = torch.as_tensor(common_matched_ir, dtype=torch.long, device=model.device)
        specific_matched_rgb = torch.as_tensor(specific_matched_rgb, dtype=torch.long, device=model.device)
        specific_matched_ir = torch.as_tensor(specific_matched_ir, dtype=torch.long, device=model.device)

        remain_matched_rgb = torch.as_tensor(remain_matched_rgb, dtype=torch.long, device=model.device)
        remain_matched_ir = torch.as_tensor(remain_matched_ir, dtype=torch.long, device=model.device)
        all_matched_rgb = torch.as_tensor(all_matched_rgb, dtype=torch.long, device=model.device)
        all_matched_ir = torch.as_tensor(all_matched_ir, dtype=torch.long, device=model.device)

        # 控制分类器开关 (Phase 2等高阶阶段启用融合分类器)
        if not model.enable_cls3:
            model.enable_cls3 = True
    # ======================================================
    # 2. 网络前向传播与模型损失计算主循环
    # ======================================================
    model.set_train()
    meter = MultiItemAverageMeter()
    bt = args.batch_pidnum*args.pid_numsample
    rgb_loader, ir_loader = dataset.get_train_loader()

    nan_batch_counter = 0
    # 用 zip 将可见光数据 (rgb) 和红外数据 (ir) 拼接遍历迭代
    for (rgb_imgs, ca_imgs, color_info), (ir_imgs, aug_imgs,ir_info) in zip(rgb_loader, ir_loader):
        # 梯度清零，根据阶段选择相应的优化器
        if enable_phase1:
            model.optimizer_phase1.zero_grad()
        else:
            model.optimizer_phase2.zero_grad()
            
        # 将张量送入设定设备
        rgb_imgs, ca_imgs = rgb_imgs.to(model.device), ca_imgs.to(model.device)
        color_imgs = torch.cat((rgb_imgs, ca_imgs), dim = 0)
        
        # 提取真实或伪造 Ground Truth 的特征
        rgb_gts, ir_gts = color_info[:,-1], ir_info[:,-1] 
        rgb_ids, ir_ids = color_info[:,1], ir_info[:,1]
        rgb_ids = torch.cat((rgb_ids,rgb_ids)).to(model.device)
        
        # 不同数据集格式拼装兼容
        if args.dataset == 'regdb':
            ir_imgs, aug_imgs = ir_imgs.to(model.device), aug_imgs.to(model.device)
            ir_imgs = torch.cat((ir_imgs, aug_imgs), dim = 0)
            ir_ids = torch.cat((ir_ids,ir_ids)).to(model.device)
        else:
            ir_imgs = ir_imgs.to(model.device)
            ir_ids = ir_ids.to(model.device)
            
        # >> 执行前向传播获得特征和Logits分布 <<
        if use_rgmfd:
            gap_features, bn_features, rgmfd_pack = model.model(color_imgs, ir_imgs, return_rg=True)
        else:
            gap_features, bn_features = model.model(color_imgs, ir_imgs)
            rgmfd_pack = None
        # classifier1：可见光专家 W_v => 输出 rgb 的分类概率 (包含跨模态 p_{v->r})
        rgbcls_out, _l2_features = model.classifier1(bn_features)
        # classifier2：红外专家 W_r => 输出 ir 的分类概率 (包含跨模态 p_{r->v})
        ircls_out, _l2_features = model.classifier2(bn_features)

        # 拆分出纯净特征
        rgb_features, ir_features = gap_features[:2*bt], gap_features[2*bt:]
        
        # 拆分各专家在各个模态样本上的打分 Logits:
        # r2r_cls (VIS预测VIS), i2i_cls (IR预测IR), r2i_cls (VIS预测IR), i2r_cls (IR预测VIS)
        r2r_cls, i2i_cls, r2i_cls, i2r_cls =\
              rgbcls_out[:2*bt], ircls_out[2*bt:], ircls_out[:2*bt], rgbcls_out[2*bt:]
        if 'wsl' in args.debug:
            if enable_phase1:
                # ----------------
                # 第一阶段 Phase 1: 各自模态专家进行内模态预热
                # 计算模态内的交叉熵与特征维度三元组损失 (ID_Loss & Triplet_Loss)
                # ----------------
                r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
                i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
                
                total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
                if epoch >= getattr(args, 'rgmfd_start_epoch', 0):
                    total_loss = _add_loss_dict(
                        total_loss,
                        _rgmfd_regularization_losses(args, rgmfd_pack),
                        meter,
                    )
                
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data,
                            'r2r_tri_loss':r2r_tri_loss.data,
                            'i2i_tri_loss':i2i_tri_loss.data})
            else:
                # ----------------
                # 第二阶段 Phase 2: 开始执行跨模态协作与对齐
                # 建立融合分类器和双向蒸馏损失
                # ----------------
                r2c_cls = model.classifier3(bn_features)[0][:2*bt]
                i2c_cls = model.classifier3(bn_features)[0][2*bt:]
                dtd_features = bn_features.detach() # 脱离计算图，用于蒸馏(distillation)指导
                
                # 提取不传递梯度的辅助预测指导量
                dtd_rgbcls_out = model.classifier1(dtd_features)[0]
                dtd_ircls_out = model.classifier2(dtd_features)[0]
                dtd_r2r_cls, dtd_i2r_cls = dtd_rgbcls_out[:2*bt], dtd_rgbcls_out[2*bt:]
                dtd_r2i_cls, dtd_i2i_cls = dtd_ircls_out[:2*bt], dtd_ircls_out[2*bt:]
                
                # 计算跨模态的基础 ID 误差
                r2r_id_loss = model.pid_criterion(dtd_r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(dtd_i2i_cls, ir_ids)
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data})
                total_loss = r2r_id_loss + i2i_id_loss
                if epoch >= getattr(args, 'rgmfd_start_epoch', 0):
                    total_loss = _add_loss_dict(
                        total_loss,
                        _rgmfd_regularization_losses(args, rgmfd_pack),
                        meter,
                    )
                
                common_rgb_indices = torch.isin(rgb_ids, common_matched_rgb)
                common_ir_indices = torch.isin(ir_ids, common_matched_ir)
                ###############################################################
                if args.debug == 'wsl':
                    # ================ A. Triplet 跨模态正负样本发掘 ================
                    # 利用 "common" (互为最近邻的一对一) 的可靠标签，构造跨模态三元组
                    tri_rgb_indices = torch.isin(rgb_ids, common_matched_rgb)
                    tri_ir_indices = torch.isin(ir_ids, common_matched_ir)
                    selected_tri_rgb_ids = rgb_ids[tri_rgb_indices]
                    selected_tri_ir_ids = ir_ids[tri_ir_indices]
                    
                    # 依据 common_rm 记录的映射字典，将一组伪标签映射到对方的域
                    translated_tri_rgb_label = torch.nonzero(common_rm[selected_tri_rgb_ids])[:,-1]
                    translated_tri_ir_label = torch.nonzero(common_rm.T[selected_tri_ir_ids])[:,-1]
                
                    selected_tri_rgb_features = rgb_features[tri_rgb_indices]
                    selected_tri_ir_features = ir_features[tri_ir_indices]
                    matched_tri_rgb_features = torch.cat((selected_tri_rgb_features,ir_features),dim=0)
                    matched_tri_ir_features = torch.cat((rgb_features,selected_tri_ir_features),dim=0)
                    matched_tri_rgb_labels = torch.cat((translated_tri_rgb_label,ir_ids),dim=0)
                    matched_tri_ir_labels = torch.cat((rgb_ids,translated_tri_ir_label),dim=0)
                    
                    # 混合特征送入计算跨模态的三元组误差距离
                    tri_loss_rgb = common_loss_weight * args.tri_weight * model.tri_criterion(matched_tri_rgb_features, matched_tri_rgb_labels)
                    tri_loss_ir = common_loss_weight * args.tri_weight * model.tri_criterion(matched_tri_ir_features, matched_tri_ir_labels)
                    meter.update({'tri_loss_rgb':tri_loss_rgb.data,
                                'tri_loss_ir':tri_loss_ir.data})
                    total_loss += tri_loss_rgb + tri_loss_ir

                    # ================ B. CMO 跨模态优化损失 (Cross Modal Optimization loss) ================
                    selected_common_rgb_ids = rgb_ids[common_rgb_indices]
                    selected_common_ir_ids = ir_ids[common_ir_indices]
                    translated_cmo_rgb_label = torch.nonzero(common_rm[selected_common_rgb_ids])[:,-1]
                    translated_cmo_ir_label = torch.nonzero(common_rm.T[selected_common_ir_ids])[:,-1]
                    
                    # 将本次 Batch 的特征与标号放入 CMA(Cross Modal Match Aggregation) 进行动量更新
                    cma.update(bn_features[:2*bt], bn_features[2*bt:], rgb_ids, ir_ids)
                    
                    # 利用信息熵(Entropy)做基于强度的权值重新分配：
                    # 当模型对某个跨模态打分极为自信（低熵），代表它跨越差距难度低，分配更大比重
                    r2i_entropy = infoEntropy(r2i_cls)
                    i2r_entropy = infoEntropy(i2r_cls)
                    w_r2i = r2i_entropy/(r2i_entropy+i2r_entropy)
                    w_i2r = i2r_entropy/(r2i_entropy+i2r_entropy)
                    
                    # 取出 CMA Memory 里面的聚合旧特征，作为监督原型指导当前特征的学习
                    selected_rgb_memory = cma.vis_memory[translated_cmo_ir_label].detach()
                    selected_ir_memory = cma.ir_memory[translated_cmo_rgb_label].detach()
                    
                    # 取分类器对该原型的分布响应
                    mem_r2i_cls,_ = model.classifier2(selected_rgb_memory)
                    mem_i2r_cls,_ = model.classifier1(selected_ir_memory)
                    
                    # MSE(均方误差) 将当前分支的模态间互蒸馏结果向原型对齐
                    cmo_criterion = torch.nn.MSELoss()

                    # 验证并叠加 CMO 损失
                    if (selected_tri_ir_ids.shape[0]!=0):
                        r2i_cmo_loss = common_loss_weight * w_r2i * cmo_criterion(dtd_i2i_cls[common_ir_indices],mem_r2i_cls)
                        if torch.isnan(r2i_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'r2i_cmo_loss':r2i_cmo_loss.data})
                            total_loss += r2i_cmo_loss
                    if (selected_tri_rgb_ids.shape[0]!=0):
                        i2r_cmo_loss = common_loss_weight * w_i2r * cmo_criterion(dtd_r2r_cls[common_rgb_indices],mem_i2r_cls)
                        if torch.isnan(i2r_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'i2r_cmo_loss':i2r_cmo_loss.data})
                            total_loss += i2r_cmo_loss

                    if rgmfd_pack is not None and epoch >= getattr(args, 'rgmfd_start_epoch', 0):
                        rel_losses = []
                        shared_bn = rgmfd_pack['shared_bn']
                        if selected_common_rgb_ids.shape[0] != 0:
                            selected_rgb_shared = shared_bn[:2*bt][common_rgb_indices]
                            rgb_rel_weights = common_rm[selected_common_rgb_ids, translated_cmo_rgb_label].detach()
                            rel_losses.append(
                                _weighted_cosine_alignment_loss(
                                    selected_rgb_shared, selected_ir_memory, rgb_rel_weights
                                )
                            )
                        if selected_common_ir_ids.shape[0] != 0:
                            selected_ir_shared = shared_bn[2*bt:][common_ir_indices]
                            ir_rel_weights = common_rm.T[selected_common_ir_ids, translated_cmo_ir_label].detach()
                            rel_losses.append(
                                _weighted_cosine_alignment_loss(
                                    selected_ir_shared, selected_rgb_memory, ir_rel_weights
                                )
                            )
                        if len(rel_losses) > 0:
                            rgmfd_rel_loss = (
                                common_loss_weight
                                * getattr(args, 'rgmfd_rel_weight', 0.0)
                                * sum(rel_losses) / len(rel_losses)
                            )
                            if torch.isnan(rgmfd_rel_loss).any():
                                nan_batch_counter += 1
                            else:
                                meter.update({'rgmfd_rel_loss': rgmfd_rel_loss.data})
                                total_loss += rgmfd_rel_loss

                # ================ C. 中后期处理模糊(remain)弱监督边界 ================
                remain_start_epoch = getattr(args, 'trrm_remain_start', 60) if getattr(args, 'enable_trrm', 1) else 30
                remain_weight = getattr(args, 'trrm_remain_weight', 0.2) if getattr(args, 'enable_trrm', 1) else 1.0
                remain_weight *= remain_loss_weight
                if epoch >= remain_start_epoch:
                    remain_rgb_indices = torch.isin(rgb_ids, remain_matched_rgb)
                    remain_ir_indices = torch.isin(ir_ids, remain_matched_ir)
                    remain_rgb_ids = rgb_ids[remain_rgb_indices]
                    remain_ir_ids = ir_ids[remain_ir_indices]
                    remain_r2c_cls = r2c_cls[remain_rgb_indices]
                    remain_i2c_cls = i2c_cls[remain_ir_indices]
                    
                    # 针对一对多或多对多重叠标签的数据点，用 weak_criterion（可能是一种软标签或正则化损失）温和约束
                    remain_labels = remain_rm[remain_rgb_ids]
                    remain_r2c_cls, remain_labels, remain_gate_stats = _filter_remain_weak_inputs(
                        remain_r2c_cls, remain_labels, args
                    )
                    if getattr(args, 'enable_remain_gate', 0) and remain_gate_stats['total'] > 0:
                        meter.update({
                            'remain_gate_keep': remain_gate_stats['keep_ratio'],
                            'remain_gate_score': remain_gate_stats['score'],
                            'remain_gate_prob': remain_gate_stats['target_prob'],
                        })
                    if remain_r2c_cls.shape[0] > 0:
                        weak_r2c_loss = remain_weight * args.weak_weight*model.weak_criterion(remain_r2c_cls, remain_labels)
                        if torch.isnan(weak_r2c_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'weak_r2c_loss':weak_r2c_loss.data})
                            total_loss += weak_r2c_loss
        if enable_phase1:
            total_loss.backward()
            model.optimizer_phase1.step()
        else:                
            if args.debug == 'wsl':
                # ================ D. 模态特有身份的惩罚 (Modal Specific Pseudo Labels) ================
                # 处理被划分为 specific (单向包含映射) 里的节点，并用软标签监督更新
                specific_rgb_indices = torch.isin(rgb_ids, specific_matched_rgb)
                specific_ir_indices = torch.isin(ir_ids, specific_matched_ir)
                
                # 剔除掉属于 common(高质量互映射) 后剩下的所谓 specific
                rgb_indices = specific_rgb_indices ^ common_rgb_indices
                ir_indices = specific_ir_indices ^ common_ir_indices

                selected_ir_ids = ir_ids[ir_indices]
                selected_rgb_ids = rgb_ids[rgb_indices]
                selected_i2c_cls = i2c_cls[ir_indices]
                selected_r2c_cls = r2c_cls[rgb_indices]

                # 计算针对可见光特征跨入特有集合的 ID 分类损失
                if (selected_rgb_ids.shape[0]>0):
                    specific_weight = specific_loss_weight
                    rgb_cross_loss = specific_weight * model.pid_criterion(selected_r2c_cls, specific_rm[selected_rgb_ids])
                    if torch.isnan(rgb_cross_loss).any():
                        nan_batch_counter+=1
                    else:
                        meter.update({'rgb_cross_loss':rgb_cross_loss.data})
                        total_loss += rgb_cross_loss
                # 红外特征对自身分配到标签库的 ID 分类损失
                ir_cross_loss = model.pid_criterion(i2c_cls, ir_ids)
                meter.update({'ir_cross_loss':ir_cross_loss.data})
                total_loss+= ir_cross_loss
                    
            elif args.debug == 'baseline':
                # ================ E. 纯净基线(Baseline) 监督模式 ================
                # 不跑复杂的伪标签蒸馏，直接做跨模态基本损失
                    r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                    i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                    r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
                    i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
                    
                    total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
                    
                    meter.update({'r2r_id_loss':r2r_id_loss.data,
                                'i2i_id_loss':i2i_id_loss.data,
                                'r2r_tri_loss':r2r_tri_loss.data,
                                'i2i_tri_loss':i2i_tri_loss.data})
            
            elif args.debug == 'sl':
                # ================ F. SL(Supervised Learning 全监督模式) ================
                # 将两种模态并在一起拉近距离的主流全监督策略
                rgb_gts = torch.cat((rgb_gts,rgb_gts)).to(model.device)
                ir_gts = torch.cat((ir_gts,ir_gts)).to(model.device)
                gts = torch.cat((rgb_gts,ir_gts))

                id_loss = model.pid_criterion(rgbcls_out, gts)
                tri_loss = model.tri_criterion(gap_features, gts)
                total_loss = id_loss + args.tri_weight*tri_loss
                
                meter.update({'id_loss': id_loss.data,
                                'tri_loss': tri_loss.data})

            else:
                raise RuntimeError('Debug mode {} not found!'.format(args.debug))
        
            # Phase 2 主优化器梯度的向后传播更新
            total_loss.backward()
            model.optimizer_phase2.step()
            
    return meter.get_val(), meter.get_str()

def relabel(select_ids, source_labels, target_labels):
    """
    重标记工具函数：
    用来在内存字典中将预测出的身份 ID 或者原始 Batch ID 重新映射到联合模态分配出的 target ID 空间中。
    Input: source_labels, target_labels
    Output: 对应 target 模态体系内的新 ID 值 select_ids
    """
    key_to_value = torch.full((torch.max(source_labels) + 1,), -1, dtype=torch.long).to(source_labels.device)
    key_to_value[source_labels] = target_labels
    
    select_ids = key_to_value[select_ids]
    return select_ids

def hate_nan(loss, condition,logger):
    """
    检查损失的安全性。遇到极端梯度崩溃 (NaN Loss) 抛出异常日志，而不是悄悄污染模型。
    """
    if torch.isnan(loss):
        if condition:
            logger('no matched labels')
        else:
            logger('nan loss detected')
        return torch.tensor(0.0).to(loss.device)
    else:
        return loss
