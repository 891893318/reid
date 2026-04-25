import torch
from models import Model
from datasets import SYSU
import time
import numpy as np
import random
import copy
import math
import torch.nn.functional as F
from collections import OrderedDict
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy,pha_unwrapping
from models import Model


def _cosine_distance(inputs, targets):
    inputs = F.normalize(inputs, dim=1)
    targets = F.normalize(targets, dim=1)
    return 1.0 - (inputs * targets).sum(dim=1)


def _entropy_weight(prob):
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=1)
    normalizer = math.log(prob.size(1) + 1e-12)
    return 1.0 - entropy / normalizer


def _soft_relation_alignment(cma, source_feats, source_labels, source_cams, source_modal,
                             target_modal, args):
    _, prob, target_proto = cma.get_cross_modal_relation(
        source_modal=source_modal,
        target_modal=target_modal,
        source_feats=source_feats,
        source_labels=source_labels,
        source_cams=source_cams,
        lambda_instance=args.scrc_lambda_instance,
        lambda_cam=args.scrc_lambda_cam,
        lambda_global=args.scrc_lambda_global,
        temperature=args.scrc_relation_temp,
    )
    weights = _entropy_weight(prob)
    norm_source = F.normalize(source_feats, dim=1)
    norm_target = F.normalize(target_proto, dim=2)
    distances = 1.0 - torch.einsum('bd,bkd->bk', norm_source, norm_target)
    loss = (weights * (prob * distances).sum(dim=1)).mean()
    return loss, prob


def _prototype_regularization(cma, feats, labels, cams, modal):
    global_proto, cam_proto = cma.gather_same_modal_prototypes(modal, labels, cams)
    return _cosine_distance(feats, global_proto).mean() + _cosine_distance(feats, cam_proto).mean()


def _bidirectional_consistency(cma, args):
    vis_global = F.normalize(cma.vis_memory, dim=1)
    ir_global = F.normalize(cma.ir_memory, dim=1)
    scores = torch.matmul(vis_global, ir_global.t())
    p = torch.softmax(scores / args.scrc_relation_temp, dim=1)
    q = torch.softmax(scores.t() / args.scrc_relation_temp, dim=1).t()
    return F.kl_div(torch.log(p.clamp_min(1e-12)), q.detach(), reduction='batchmean') + \
        F.kl_div(torch.log(q.clamp_min(1e-12)), p.detach(), reduction='batchmean')

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

        # 1.3 利用划分好的字典构建各个关系类型的稀疏连接矩阵 (rm)
        all_rm = torch.zeros((args.num_classes,args.num_classes)).to(model.device) # 总体连接矩阵
        common_rm = all_rm.clone()   # M_c : 互为最近邻的一对一关系
        specific_rm = all_rm.clone() # M_s : 特有对应关系
        remain_rm = all_rm.clone()   # M_w : 冲突的对应关系
        r2i_rm = all_rm.clone()      # M_{v->r}
        i2r_rm = all_rm.clone()      # M_{r->v}
        
        for r, i in common_dict.items(): 
            common_rm[r,i] += 1
        for r, i in specific_dict.items():
            specific_rm[r,i] += 1
        for r, i in r2i_pair_dict.items():
            r2i_rm[r,i] += 1
        for i, r in i2r_pair_dict.items():
            i2r_rm[i,r] += 1
        for r, i in remain_dict.items():
            remain_rm[r,i] += 1

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
        matched_rgb = torch.tensor(matched_rgb).to(model.device)
        matched_ir = torch.tensor(matched_ir).to(model.device)
        common_matched_rgb = torch.tensor(common_matched_rgb).to(model.device)
        common_matched_ir = torch.tensor(common_matched_ir).to(model.device)
        specific_matched_rgb = torch.tensor(specific_matched_rgb).to(model.device)
        specific_matched_ir = torch.tensor(specific_matched_ir).to(model.device)

        remain_matched_rgb = torch.tensor(remain_matched_rgb).to(model.device)
        remain_matched_ir = torch.tensor(remain_matched_ir).to(model.device)
        all_matched_rgb = torch.tensor(all_matched_rgb).to(model.device)
        all_matched_ir = torch.tensor(all_matched_ir).to(model.device)

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
        rgb_cams, ir_cams = color_info[:,2], ir_info[:,2]
        rgb_ids = torch.cat((rgb_ids,rgb_ids)).to(model.device)
        rgb_cams = torch.cat((rgb_cams, rgb_cams)).to(model.device)
        
        # 不同数据集格式拼装兼容
        if args.dataset == 'regdb':
            ir_imgs, aug_imgs = ir_imgs.to(model.device), aug_imgs.to(model.device)
            ir_imgs = torch.cat((ir_imgs, aug_imgs), dim = 0)
            ir_ids = torch.cat((ir_ids,ir_ids)).to(model.device)
            ir_cams = torch.cat((ir_cams, ir_cams)).to(model.device)
        else:
            ir_imgs = ir_imgs.to(model.device)
            ir_ids = ir_ids.to(model.device)
            ir_cams = ir_cams.to(model.device)
            
        # >> 执行前向传播获得特征和Logits分布 <<
        gap_features, bn_features = model.model(color_imgs, ir_imgs)
        # classifier1：可见光专家 W_v => 输出 rgb 的分类概率 (包含跨模态 p_{v->r})
        rgbcls_out, _l2_features = model.classifier1(bn_features)
        # classifier2：红外专家 W_r => 输出 ir 的分类概率 (包含跨模态 p_{r->v})
        ircls_out, _l2_features = model.classifier2(bn_features)

        # 拆分出纯净特征
        rgb_features, ir_features = gap_features[:2*bt], gap_features[2*bt:]
        rgb_bn_features, ir_bn_features = bn_features[:2*bt], bn_features[2*bt:]
        
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
                proto_rgb_loss = args.scrc_proto_weight * _prototype_regularization(
                    cma, rgb_bn_features, rgb_ids, rgb_cams, 'rgb'
                )
                proto_ir_loss = args.scrc_proto_weight * _prototype_regularization(
                    cma, ir_bn_features, ir_ids, ir_cams, 'ir'
                )

                total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss + proto_rgb_loss + proto_ir_loss
                
                meter.update({'r2r_id_loss':r2r_id_loss.data,
                            'i2i_id_loss':i2i_id_loss.data,
                            'r2r_tri_loss':r2r_tri_loss.data,
                            'i2i_tri_loss':i2i_tri_loss.data,
                            'proto_rgb_loss': proto_rgb_loss.data,
                            'proto_ir_loss': proto_ir_loss.data})
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
                    tri_loss_rgb = args.tri_weight * model.tri_criterion(matched_tri_rgb_features, matched_tri_rgb_labels)
                    tri_loss_ir = args.tri_weight * model.tri_criterion(matched_tri_ir_features, matched_tri_ir_labels)
                    meter.update({'tri_loss_rgb':tri_loss_rgb.data,
                                'tri_loss_ir':tri_loss_ir.data})
                    total_loss += tri_loss_rgb + tri_loss_ir

                    # ================ B. CMO 跨模态优化损失 (Cross Modal Optimization loss) ================
                    selected_common_rgb_ids = rgb_ids[common_rgb_indices]
                    selected_common_ir_ids = ir_ids[common_ir_indices]
                    translated_cmo_rgb_label = torch.nonzero(common_rm[selected_common_rgb_ids])[:,-1]
                    translated_cmo_ir_label = torch.nonzero(common_rm.T[selected_common_ir_ids])[:,-1]
                    
                    # 将本次 Batch 的特征与标号放入 CMA(Cross Modal Match Aggregation) 进行动量更新
                    cma.update(rgb_bn_features, ir_bn_features, rgb_ids, ir_ids, rgb_cams, ir_cams)
                    
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
                        r2i_cmo_loss = w_r2i * cmo_criterion(dtd_i2i_cls[common_ir_indices],mem_r2i_cls)
                        if torch.isnan(r2i_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'r2i_cmo_loss':r2i_cmo_loss.data})
                            total_loss += r2i_cmo_loss
                    if (selected_tri_rgb_ids.shape[0]!=0):
                        i2r_cmo_loss = w_i2r * cmo_criterion(dtd_r2r_cls[common_rgb_indices],mem_i2r_cls)
                        if torch.isnan(i2r_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'i2r_cmo_loss':i2r_cmo_loss.data})
                            total_loss += i2r_cmo_loss

                    scrc_cm_rgb, rgb_relation_prob = _soft_relation_alignment(
                        cma, rgb_bn_features, rgb_ids, rgb_cams, 'rgb', 'ir', args
                    )
                    scrc_cm_ir, ir_relation_prob = _soft_relation_alignment(
                        cma, ir_bn_features, ir_ids, ir_cams, 'ir', 'rgb', args
                    )
                    scrc_cm_loss = args.scrc_cm_weight * (scrc_cm_rgb + scrc_cm_ir)
                    scrc_bi_loss = args.scrc_bi_weight * _bidirectional_consistency(cma, args)
                    scrc_proto_rgb = args.scrc_proto_weight * _prototype_regularization(
                        cma, rgb_bn_features, rgb_ids, rgb_cams, 'rgb'
                    )
                    scrc_proto_ir = args.scrc_proto_weight * _prototype_regularization(
                        cma, ir_bn_features, ir_ids, ir_cams, 'ir'
                    )
                    total_loss += scrc_cm_loss + scrc_bi_loss + scrc_proto_rgb + scrc_proto_ir
                    meter.update({
                        'scrc_cm_loss': scrc_cm_loss.data,
                        'scrc_bi_loss': scrc_bi_loss.data,
                        'scrc_proto_rgb': scrc_proto_rgb.data,
                        'scrc_proto_ir': scrc_proto_ir.data,
                    })

                # ================ C. 中后期处理模糊(remain)弱监督边界 ================
                if epoch >= 30:
                    remain_rgb_indices = torch.isin(rgb_ids, remain_matched_rgb)
                    remain_ir_indices = torch.isin(ir_ids, remain_matched_ir)
                    remain_rgb_ids = rgb_ids[remain_rgb_indices]
                    remain_ir_ids = ir_ids[remain_ir_indices]
                    remain_r2c_cls = r2c_cls[remain_rgb_indices]
                    remain_i2c_cls = i2c_cls[remain_ir_indices]
                    
                    # 针对一对多或多对多重叠标签的数据点，用 weak_criterion（可能是一种软标签或正则化损失）温和约束
                    if (remain_rgb_indices.shape[0]>0):
                        weak_r2c_loss = args.weak_weight*model.weak_criterion(remain_r2c_cls, remain_rm[remain_rgb_ids])
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
                    rgb_cross_loss = model.pid_criterion(selected_r2c_cls, specific_rm[selected_rgb_ids])
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
