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


def weighted_soft_ce(logits, soft_targets, weights):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    weights = weights.float()
    valid = weights > 0
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    logits = logits[valid]
    soft_targets = soft_targets[valid]
    weights = weights[valid]
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-12)


def sparse_targets(targets, topk):
    if targets.numel() == 0:
        return targets
    k = min(topk, targets.size(1))
    values, indices = torch.topk(targets, k=k, dim=1)
    sparse = torch.zeros_like(targets)
    sparse.scatter_(1, indices, values)
    sparse = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return sparse


def weighted_mse(inputs, targets, weights):
    if inputs.numel() == 0:
        return inputs.new_tensor(0.0)
    weights = weights.float()
    valid = weights > 0
    if valid.sum() == 0:
        return inputs.new_tensor(0.0)
    diff = (inputs[valid] - targets[valid]).pow(2).mean(dim=1)
    return (diff * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-12)

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

    # ======================================================
    # 1. 弱监督学习 (WSL) 伪标签匹配与关系划分阶段
    # ======================================================
    if 'wsl' in args.debug or not enable_phase1:
        # 获取当前模型在整个训练集上的特征，并更新标签库与距离矩阵
        cma.extract(args, model, dataset)        
        cma.get_label(epoch)

        row_scores = cma.row_scores.to(model.device)
        col_scores = cma.col_scores.to(model.device)
        common_rm = cma.common_rm.to(model.device)
        specific_rm = cma.specific_rm.to(model.device)
        specific_col_rm = cma.specific_col_rm.to(model.device)
        remain_rm = cma.remain_rm.to(model.device)
        remain_col_rm = cma.remain_col_rm.to(model.device)
        row_assign = cma.row_assign.to(model.device)
        col_assign = cma.col_assign.to(model.device)
        row_conf = cma.row_conf.to(model.device)
        col_conf = cma.col_conf.to(model.device)
        common_matched_rgb = torch.nonzero(cma.common_row_mask.to(model.device), as_tuple=False).flatten()
        common_matched_ir = torch.nonzero(cma.common_col_mask.to(model.device), as_tuple=False).flatten()
        specific_matched_rgb = torch.nonzero(cma.specific_row_mask.to(model.device), as_tuple=False).flatten()
        specific_matched_ir = torch.nonzero(cma.specific_col_mask.to(model.device), as_tuple=False).flatten()
        remain_matched_rgb = torch.nonzero(cma.remain_row_mask.to(model.device), as_tuple=False).flatten()
        remain_matched_ir = torch.nonzero(cma.remain_col_mask.to(model.device), as_tuple=False).flatten()

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
        gap_features, bn_features = model.model(color_imgs, ir_imgs)
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
                
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data,
                            'r2r_tri_loss':r2r_tri_loss.data,
                            'i2i_tri_loss':i2i_tri_loss.data})
            else:
                # ----------------
                # 第二阶段 Phase 2: 基于 BMPR 可靠性路由做渐进伪监督
                r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
                i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data})
                total_loss = r2r_id_loss + i2i_id_loss

                rgb_route_targets = sparse_targets(row_scores[rgb_ids], cma.topk)
                ir_route_targets = sparse_targets(col_scores[ir_ids], cma.topk)
                rgb_route_weights = row_conf[rgb_ids]
                ir_route_weights = col_conf[ir_ids]
                common_rgb_indices = torch.isin(rgb_ids, common_matched_rgb)
                common_ir_indices = torch.isin(ir_ids, common_matched_ir)
                specific_rgb_indices = torch.isin(rgb_ids, specific_matched_rgb)
                specific_ir_indices = torch.isin(ir_ids, specific_matched_ir)
                remain_rgb_indices = torch.isin(rgb_ids, remain_matched_rgb)
                remain_ir_indices = torch.isin(ir_ids, remain_matched_ir)
                ###############################################################
                if args.debug == 'wsl':
                    # ================ A. BMPR 渐进式跨模态软监督 ================
                    rgb_bmpr_loss = weighted_soft_ce(r2i_cls, rgb_route_targets, rgb_route_weights)
                    ir_bmpr_loss = weighted_soft_ce(i2r_cls, ir_route_targets, ir_route_weights)
                    meter.update({'rgb_bmpr_loss': rgb_bmpr_loss.data,
                                  'ir_bmpr_loss': ir_bmpr_loss.data})
                    total_loss += rgb_bmpr_loss + ir_bmpr_loss

                    # ================ B. 高可靠 common 的跨模态三元组 ================
                    tri_rgb_indices = torch.isin(rgb_ids, common_matched_rgb)
                    tri_ir_indices = torch.isin(ir_ids, common_matched_ir)
                    selected_tri_rgb_ids = rgb_ids[tri_rgb_indices]
                    selected_tri_ir_ids = ir_ids[tri_ir_indices]
                    
                    translated_tri_rgb_label = row_assign[selected_tri_rgb_ids]
                    translated_tri_ir_label = col_assign[selected_tri_ir_ids]
                
                    selected_tri_rgb_features = rgb_features[tri_rgb_indices]
                    selected_tri_ir_features = ir_features[tri_ir_indices]
                    matched_tri_rgb_features = torch.cat((selected_tri_rgb_features,ir_features),dim=0)
                    matched_tri_ir_features = torch.cat((rgb_features,selected_tri_ir_features),dim=0)
                    matched_tri_rgb_labels = torch.cat((translated_tri_rgb_label,ir_ids),dim=0)
                    matched_tri_ir_labels = torch.cat((rgb_ids,translated_tri_ir_label),dim=0)
                    
                    # 混合特征送入计算跨模态的三元组误差距离
                    tri_loss_rgb = args.tri_weight * model.tri_criterion(matched_tri_rgb_features, matched_tri_rgb_labels)
                    tri_loss_ir = args.tri_weight * model.tri_criterion(matched_tri_ir_features, matched_tri_ir_labels)
                    meter.update({'tri_loss_rgb':tri_loss_rgb.data,
                                'tri_loss_ir':tri_loss_ir.data})
                    total_loss += tri_loss_rgb + tri_loss_ir

                    # ================ C. 高可靠 common 的原型蒸馏 ================
                    selected_common_rgb_ids = rgb_ids[common_rgb_indices]
                    selected_common_ir_ids = ir_ids[common_ir_indices]
                    
                    # 将本次 Batch 的特征与标号放入 CMA(Cross Modal Match Aggregation) 进行动量更新
                    cma.update(bn_features[:2*bt], bn_features[2*bt:], rgb_ids, ir_ids)

                    # 验证并叠加 CMO 损失
                    if selected_common_rgb_ids.shape[0] != 0:
                        matched_ir_memory = cma.ir_memory.detach()[row_assign[selected_common_rgb_ids]]
                        mem_r2i_cls,_ = model.classifier2(matched_ir_memory)
                        r2i_cmo_loss = weighted_mse(
                            r2i_cls[common_rgb_indices],
                            mem_r2i_cls,
                            row_conf[selected_common_rgb_ids]
                        )
                        if torch.isnan(r2i_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'r2i_cmo_loss':r2i_cmo_loss.data})
                            total_loss += r2i_cmo_loss
                    if selected_common_ir_ids.shape[0] != 0:
                        matched_rgb_memory = cma.vis_memory.detach()[col_assign[selected_common_ir_ids]]
                        mem_i2r_cls,_ = model.classifier1(matched_rgb_memory)
                        i2r_cmo_loss = weighted_mse(
                            i2r_cls[common_ir_indices],
                            mem_i2r_cls,
                            col_conf[selected_common_ir_ids]
                        )
                        if torch.isnan(i2r_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'i2r_cmo_loss':i2r_cmo_loss.data})
                            total_loss += i2r_cmo_loss

                # ================ D. remain 的弱监督收尾 ================
                if epoch >= 30:
                    remain_rgb_ids = rgb_ids[remain_rgb_indices]
                    remain_ir_ids = ir_ids[remain_ir_indices]
                    remain_r2i_cls = r2i_cls[remain_rgb_indices]
                    remain_i2r_cls = i2r_cls[remain_ir_indices]
                    
                    # 针对一对多或多对多重叠标签的数据点，用 weak_criterion（可能是一种软标签或正则化损失）温和约束
                    if remain_rgb_ids.shape[0] > 0:
                        weak_r2c_loss = args.weak_weight*model.weak_criterion(remain_r2i_cls, remain_rm[remain_rgb_ids])
                        if torch.isnan(weak_r2c_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'weak_r2c_loss':weak_r2c_loss.data})
                            total_loss += weak_r2c_loss
                    if remain_ir_ids.shape[0] > 0:
                        weak_i2c_loss = args.weak_weight*model.weak_criterion(remain_i2r_cls, remain_col_rm[remain_ir_ids])
                        if torch.isnan(weak_i2c_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'weak_i2c_loss':weak_i2c_loss.data})
                            total_loss += weak_i2c_loss
        if enable_phase1:
            total_loss.backward()
            model.optimizer_phase1.step()
        else:                
            if args.debug == 'wsl':
                # ================ E. 中可靠 specific 的软监督 ================
                selected_ir_ids = ir_ids[specific_ir_indices]
                selected_rgb_ids = rgb_ids[specific_rgb_indices]
                selected_i2r_cls = i2r_cls[specific_ir_indices]
                selected_r2i_cls = r2i_cls[specific_rgb_indices]

                # 计算针对可见光特征跨入特有集合的 BCPT 软监督损失
                if selected_rgb_ids.shape[0] > 0:
                    rgb_cross_loss = weighted_soft_ce(
                        selected_r2i_cls,
                        specific_rm[selected_rgb_ids],
                        row_conf[selected_rgb_ids]
                    )
                    if torch.isnan(rgb_cross_loss).any():
                        nan_batch_counter+=1
                    else:
                        meter.update({'rgb_cross_loss':rgb_cross_loss.data})
                        total_loss += rgb_cross_loss
                if selected_ir_ids.shape[0] > 0:
                    ir_cross_loss = weighted_soft_ce(
                        selected_i2r_cls,
                        specific_col_rm[selected_ir_ids],
                        col_conf[selected_ir_ids]
                    )
                    meter.update({'ir_cross_loss':ir_cross_loss.data})
                    total_loss += ir_cross_loss
                    
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
