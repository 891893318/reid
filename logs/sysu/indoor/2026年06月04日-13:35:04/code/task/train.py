import torch
import torch.nn.functional as F
from models import Model
from wsl import CMA
from utils import MultiItemAverageMeter, infoEntropy

def _cosine_alignment_loss(features, targets, weights=None):
    if features.numel() == 0 or targets.numel() == 0:
        return features.new_tensor(0.0)
    losses = 1.0 - F.cosine_similarity(features, targets, dim=1)
    if weights is None:
        return losses.mean()
    weights = weights.to(losses.device, dtype=losses.dtype).detach().clamp_min(0.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

def _weighted_mse_loss(inputs, targets, weights=None):
    if inputs.numel() == 0 or targets.numel() == 0:
        return inputs.new_tensor(0.0)
    losses = F.mse_loss(inputs, targets, reduction='none').mean(dim=1)
    if weights is None:
        return losses.mean()
    weights = weights.to(losses.device, dtype=losses.dtype).detach().clamp_min(0.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

def _weighted_weak_loss(scores, labels):
    if scores.numel() == 0 or labels.numel() == 0:
        return scores.new_tensor(0.0)
    eps = 1e-10
    probs = F.softmax(scores, dim=1)
    labels = labels.to(scores.device, dtype=scores.dtype).detach().clamp_min(0.0)
    positive_mask = labels > 0
    row_weights = labels.max(dim=1).values
    valid_rows = row_weights > 0
    if not valid_rows.any():
        return scores.new_tensor(0.0)

    positive_probs = probs.masked_fill(~positive_mask, 1.0)
    label_probs = positive_probs.min(dim=1, keepdim=True).values
    negative_mask = (probs < label_probs).to(scores.dtype)
    losses = -((1.0 - probs + eps).log() * negative_mask).sum(dim=1)
    return (losses[valid_rows] * row_weights[valid_rows]).sum() / row_weights[valid_rows].sum().clamp_min(1e-6)

def _shared_proto_contrast_loss(
    features,
    target_labels,
    prototypes,
    temperature=0.07,
    weights=None,
    hard_k=0,
    hard_weight=0.0,
):
    if features.numel() == 0 or target_labels.numel() == 0 or prototypes.numel() == 0:
        return features.new_tensor(0.0)

    target_labels = target_labels.to(features.device, dtype=torch.long)
    valid = (target_labels >= 0) & (target_labels < prototypes.shape[0])
    if not valid.any():
        return features.new_tensor(0.0)

    features = F.normalize(features[valid], dim=1)
    target_labels = target_labels[valid]
    prototypes = F.normalize(prototypes.to(features.device).detach(), dim=1)
    logits = torch.matmul(features, prototypes.t()) / max(float(temperature), 1e-6)
    losses = F.cross_entropy(logits, target_labels, reduction='none')

    if hard_k > 0 and hard_weight > 0 and prototypes.shape[0] > 1:
        hard_k = min(int(hard_k), prototypes.shape[0] - 1)
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        neg_logits = logits.detach().clone()
        neg_logits[row_idx, target_labels] = -float('inf')
        hard_idx = neg_logits.topk(k=hard_k, dim=1).indices
        hard_logits = logits.gather(1, hard_idx)
        pos_logits = logits[row_idx, target_labels].unsqueeze(1)
        losses = losses + float(hard_weight) * F.softplus(hard_logits - pos_logits).mean(dim=1)

    if weights is None:
        return losses.mean()

    weights = weights.to(losses.device, dtype=losses.dtype).detach()[valid].clamp_min(0.0)
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
    use_rgmfd = bool(getattr(args, 'enable_rgmfd', 0)) and bool(getattr(model.model, 'enable_rgmfd', False))
    use_rgspc = (
        bool(getattr(args, 'enable_rgspc', 0))
        and use_rgmfd
        and not enable_phase1
        and epoch >= getattr(args, 'rgspc_start_epoch', 40)
    )
    use_rgrc = (
        bool(getattr(args, 'enable_rgrc', 0))
        and not enable_phase1
        and epoch >= getattr(args, 'rgrc_start_epoch', 0)
    )
    use_raecl = (
        bool(getattr(args, 'enable_raecl', 0))
        and not enable_phase1
        and not use_rgrc
        and epoch >= getattr(args, 'raecl_start_epoch', 0)
    )

    # ======================================================
    # 1. 弱监督学习 (WSL) 伪标签匹配与关系划分阶段
    # ======================================================
    if 'wsl' in args.debug or not enable_phase1:
        # 获取当前模型在整个训练集上的特征，并更新标签库与距离矩阵
        cma.extract(args, model, dataset)        
        
        # 根据特征距离度量，计算可见光到红外(r2i)、红外到可见光(i2r) 的伪标签匹配配对
        r2i_pair_dict, i2r_pair_dict = cma.get_label(epoch)
        
        # 将关系粗暴地划分为三类：一致(common), 唯一/冲突片段(specific), 剩余弱对应(remain)
        common_dict, specific_dict, remain_dict = {},{},{}
        
        # 1.1 遍历 rgb->ir 的匹配
        for r,i in r2i_pair_dict.items():
            if i in i2r_pair_dict.keys() and i2r_pair_dict[i] == r:
                # 互为最近邻：纳入高质量的“一致匹配”(common)
                common_dict[r] = i
            elif r not in i2r_pair_dict.values() and i not in i2r_pair_dict.keys():
                # 单向包含或独立：纳入“模态特有匹配”(specific)
                specific_dict[r] = i
            else:
                # 冲突或多对一：纳入“剩余模棱两可匹配”(remain)
                remain_dict[r] = i
        
        # 1.2 遍历 ir->rgb 的匹配 (过滤掉已在 common 中的)
        for i,r in i2r_pair_dict.items():
            if (r,i) in common_dict.items():
                continue
            elif r not in r2i_pair_dict.values() and i not in r2i_pair_dict.keys():
                specific_dict[r] = i
            else:
                remain_dict[r] = i

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

        # 1.3 利用划分好的字典构建各个关系类型的稀疏连接矩阵 (rm)
        all_rm = torch.zeros((args.num_classes,args.num_classes)).to(model.device) # 总体连接矩阵
        common_rm = all_rm.clone()   # M_c : 互为最近邻的一对一关系
        specific_rm = all_rm.clone() # M_s : 特有对应关系
        remain_rm = all_rm.clone()   # M_w : 冲突的对应关系
        
        for r, i in common_dict.items(): 
            common_rm[r,i] += 1
        for r, i in specific_dict.items():
            specific_rm[r,i] += 1
        for r, i in remain_dict.items():
            remain_rm[r,i] += 1

        common_rel_weight_rm = None
        if use_rgrc:
            rgrc_mats, rgrc_stats = cma.relation_correction(
                args, common_dict, specific_dict, remain_dict, model.device
            )
            common_rm = rgrc_mats['common']
            specific_rm = rgrc_mats['specific']
            remain_rm = rgrc_mats['remain']
            common_rel_weight_rm = common_rm
            logger(
                'RGRC diag | common:{}/{} tail:{} raw:{:.3f} w:{:.3f} '
                'specific:{}/{} raw:{:.3f} w:{:.3f} '
                'remain:{}/{} raw:{:.3f} w:{:.3f}'.format(
                    rgrc_stats['common']['kept'],
                    rgrc_stats['common']['num'],
                    rgrc_stats['common']['suppressed'],
                    rgrc_stats['common']['raw_mean'],
                    rgrc_stats['common']['weight_mean'],
                    rgrc_stats['specific']['kept'],
                    rgrc_stats['specific']['num'],
                    rgrc_stats['specific']['raw_mean'],
                    rgrc_stats['specific']['weight_mean'],
                    rgrc_stats['remain']['kept'],
                    rgrc_stats['remain']['num'],
                    rgrc_stats['remain']['raw_mean'],
                    rgrc_stats['remain']['weight_mean'],
                )
            )
        elif use_raecl:
            common_rel_weight_rm, raecl_stats = cma.relation_reliability(args, common_dict, model.device)
            if raecl_stats['num'] > 0:
                logger(
                    'RAECL diag | pairs:{} raw:{:.3f} weight:{:.3f} min:{:.3f} max:{:.3f}'.format(
                        raecl_stats['num'],
                        raecl_stats['raw_mean'],
                        raecl_stats['weight_mean'],
                        raecl_stats['weight_min'],
                        raecl_stats['weight_max'],
                    )
                )

        common_pairs = torch.nonzero(common_rm > 0, as_tuple=False)
        specific_pairs = torch.nonzero(specific_rm > 0, as_tuple=False)
        remain_pairs = torch.nonzero(remain_rm > 0, as_tuple=False)

        # 将一致矩阵叠加给特定矩阵，方便后续特定损失调用
        specific_rm = specific_rm + common_rm
        
        # 收集不同关联分类下匹配成功的具体身份 ID 列表
        common_matched_rgb = common_pairs[:, 0] if common_pairs.numel() > 0 else []
        common_matched_ir = common_pairs[:, 1] if common_pairs.numel() > 0 else []
        specific_matched_rgb = specific_pairs[:, 0] if specific_pairs.numel() > 0 else []
        remain_matched_rgb = remain_pairs[:, 0] if remain_pairs.numel() > 0 else []
        
        # 转换到设备张量 Tensor 以便 GPU 加速
        common_matched_rgb = torch.as_tensor(common_matched_rgb, dtype=torch.long, device=model.device)
        common_matched_ir = torch.as_tensor(common_matched_ir, dtype=torch.long, device=model.device)
        specific_matched_rgb = torch.as_tensor(specific_matched_rgb, dtype=torch.long, device=model.device)
        remain_matched_rgb = torch.as_tensor(remain_matched_rgb, dtype=torch.long, device=model.device)

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
                dtd_r2r_cls = dtd_rgbcls_out[:2*bt]
                dtd_i2i_cls = dtd_ircls_out[2*bt:]
                
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
                    rgb_relation_weights = None
                    ir_relation_weights = None
                    if common_rel_weight_rm is not None:
                        if selected_common_rgb_ids.shape[0] != 0:
                            rgb_relation_weights = common_rel_weight_rm[
                                selected_common_rgb_ids, translated_cmo_rgb_label
                            ]
                        if selected_common_ir_ids.shape[0] != 0:
                            ir_relation_weights = common_rel_weight_rm[
                                translated_cmo_ir_label, selected_common_ir_ids
                            ]
                        weight_tensors = [
                            weights for weights in (rgb_relation_weights, ir_relation_weights)
                            if weights is not None and weights.numel() > 0
                        ]
                        if len(weight_tensors) > 0:
                            pair_weight_name = 'rgrc_pair_weight' if use_rgrc else 'raecl_pair_weight'
                            meter.update({pair_weight_name: torch.cat(weight_tensors).mean().data})
                    rgb_cmo_weights = rgb_relation_weights
                    ir_cmo_weights = ir_relation_weights
                    if use_rgrc and not bool(getattr(args, 'rgrc_weight_cmo', 0)):
                        rgb_cmo_weights = None
                        ir_cmo_weights = None
                    
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
                    
                    # 验证并叠加 CMO 损失
                    if selected_common_ir_ids.shape[0] != 0:
                        r2i_cmo_loss = w_r2i * _weighted_mse_loss(
                            dtd_i2i_cls[common_ir_indices],
                            mem_r2i_cls,
                            ir_cmo_weights,
                        )
                        if torch.isnan(r2i_cmo_loss).any():
                            nan_batch_counter+=1
                        else:
                            meter.update({'r2i_cmo_loss':r2i_cmo_loss.data})
                            total_loss += r2i_cmo_loss
                    if selected_common_rgb_ids.shape[0] != 0:
                        i2r_cmo_loss = w_i2r * _weighted_mse_loss(
                            dtd_r2r_cls[common_rgb_indices],
                            mem_i2r_cls,
                            rgb_cmo_weights,
                        )
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
                            rel_losses.append(
                                _cosine_alignment_loss(
                                    selected_rgb_shared,
                                    selected_ir_memory,
                                    rgb_relation_weights,
                                )
                            )
                        if selected_common_ir_ids.shape[0] != 0:
                            selected_ir_shared = shared_bn[2*bt:][common_ir_indices]
                            rel_losses.append(
                                _cosine_alignment_loss(
                                    selected_ir_shared,
                                    selected_rgb_memory,
                                    ir_relation_weights,
                                )
                            )
                        if len(rel_losses) > 0:
                            rgmfd_rel_loss = (
                                getattr(args, 'rgmfd_rel_weight', 0.0)
                                * sum(rel_losses) / len(rel_losses)
                            )
                            if torch.isnan(rgmfd_rel_loss).any():
                                nan_batch_counter += 1
                            else:
                                meter.update({'rgmfd_rel_loss': rgmfd_rel_loss.data})
                                total_loss += rgmfd_rel_loss

                        if use_rgspc:
                            proto_losses = []
                            temperature = getattr(args, 'rgspc_temperature', 0.07)
                            hard_k = getattr(args, 'rgspc_hard_k', 0)
                            hard_weight = getattr(args, 'rgspc_hard_weight', 0.0)
                            if selected_common_rgb_ids.shape[0] != 0:
                                selected_rgb_shared = shared_bn[:2*bt][common_rgb_indices]
                                proto_losses.append(
                                    _shared_proto_contrast_loss(
                                        selected_rgb_shared,
                                        translated_cmo_rgb_label,
                                        cma.ir_memory,
                                        temperature=temperature,
                                        weights=rgb_relation_weights,
                                        hard_k=hard_k,
                                        hard_weight=hard_weight,
                                    )
                                )
                            if selected_common_ir_ids.shape[0] != 0:
                                selected_ir_shared = shared_bn[2*bt:][common_ir_indices]
                                proto_losses.append(
                                    _shared_proto_contrast_loss(
                                        selected_ir_shared,
                                        translated_cmo_ir_label,
                                        cma.vis_memory,
                                        temperature=temperature,
                                        weights=ir_relation_weights,
                                        hard_k=hard_k,
                                        hard_weight=hard_weight,
                                    )
                                )
                            if len(proto_losses) > 0:
                                rgspc_loss = (
                                    getattr(args, 'rgspc_weight', 0.0)
                                    * sum(proto_losses) / len(proto_losses)
                                )
                                if torch.isnan(rgspc_loss).any():
                                    nan_batch_counter += 1
                                else:
                                    meter.update({'rgspc_loss': rgspc_loss.data})
                                    total_loss += rgspc_loss

                # ================ C. 中后期处理模糊(remain)弱监督边界 ================
                if epoch >= 30:
                    remain_rgb_indices = torch.isin(rgb_ids, remain_matched_rgb)
                    remain_rgb_ids = rgb_ids[remain_rgb_indices]
                    remain_r2c_cls = r2c_cls[remain_rgb_indices]

                    # 针对一对多或多对多重叠标签的数据点，用 weak_criterion 温和约束
                    remain_labels = remain_rm[remain_rgb_ids]
                    if remain_r2c_cls.shape[0] > 0:
                        if use_rgrc:
                            weak_r2c_loss = args.weak_weight * _weighted_weak_loss(remain_r2c_cls, remain_labels)
                        else:
                            weak_r2c_loss = args.weak_weight * model.weak_criterion(remain_r2c_cls, remain_labels)
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
                
                # 剔除掉属于 common(高质量互映射) 后剩下的所谓 specific
                rgb_indices = specific_rgb_indices ^ common_rgb_indices

                selected_rgb_ids = rgb_ids[rgb_indices]
                selected_r2c_cls = r2c_cls[rgb_indices]

                # 计算针对可见光特征跨入特有集合的 ID 分类损失
                if selected_rgb_ids.shape[0] > 0:
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
