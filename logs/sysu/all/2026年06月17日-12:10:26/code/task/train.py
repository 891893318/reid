import torch
import torch.nn.functional as F

from models import Model
from models.relation import (
    build_uprt_posterior,
    posterior_classifier_loss,
    posterior_cross_modal_loss,
)
from utils import MultiItemAverageMeter
from wsl import CMA


def _update_hcl_state(args, epoch, cma, common_dict, allow_activation):
    current_pairs = set((int(rgb), int(ir)) for rgb, ir in common_dict.items())
    previous_pairs = getattr(cma, 'previous_hcl_pairs', None)
    stability = 0.0
    if previous_pairs is not None and current_pairs:
        stability = len(current_pairs & previous_pairs) / len(current_pairs)
    cma.previous_hcl_pairs = current_pairs

    pair_streaks = getattr(cma, 'hcl_pair_streaks', {})
    pair_streaks = {pair: pair_streaks.get(pair, 0) + 1 for pair in current_pairs}
    cma.hcl_pair_streaks = pair_streaks
    min_pair_streak = max(1, getattr(args, 'hcl_pair_streak', 2))
    stable_pairs = {pair for pair, streak in pair_streaks.items() if streak >= min_pair_streak}

    coverage = len(current_pairs) / max(1, args.num_classes)
    stable_coverage = len(stable_pairs) / max(1, args.num_classes)
    ready = (
        allow_activation
        and bool(getattr(args, 'enable_hcl', 0))
        and stable_coverage >= getattr(args, 'hcl_min_coverage', 0.2)
        and stability >= getattr(args, 'hcl_min_stability', 0.7)
    )
    cma.hcl_ready_streak = getattr(cma, 'hcl_ready_streak', 0) + 1 if ready else 0

    if not getattr(cma, 'hcl_active', False):
        required_epochs = max(1, getattr(args, 'hcl_ready_epochs', 3))
        if cma.hcl_ready_streak >= required_epochs:
            cma.hcl_active = True
            cma.hcl_active_epoch = epoch

    active = bool(getattr(cma, 'hcl_active', False))
    warmup = max(1, getattr(args, 'hcl_warmup_epochs', 5))
    scale = 0.0
    if active:
        warmup_scale = max(0.0, min(1.0, (epoch - cma.hcl_active_epoch + 1) / warmup))
        full_coverage = max(1e-6, getattr(args, 'hcl_full_coverage', 0.7))
        coverage_scale = min(1.0, stable_coverage / full_coverage)
        scale = warmup_scale * coverage_scale

    return {
        'active': active,
        'coverage': coverage,
        'stable_coverage': stable_coverage,
        'stability': stability,
        'pairs': stable_pairs,
        'streak': cma.hcl_ready_streak,
        'scale': scale,
    }


def _hard_negative_infonce(anchors, positive_labels, memory, topk, temperature):
    if anchors.numel() == 0:
        return None, {}

    memory = memory.detach()
    memory_valid = memory.norm(dim=1) > 1e-12
    safe_labels = positive_labels.clamp(min=0, max=memory.shape[0] - 1)
    valid = (
        (positive_labels >= 0)
        & (positive_labels < memory.shape[0])
        & memory_valid[safe_labels]
    )
    if not valid.any() or memory_valid.sum() < 2:
        return None, {}

    anchors = F.normalize(anchors[valid], dim=1)
    positive_labels = positive_labels[valid]
    memory = F.normalize(memory, dim=1)
    similarities = torch.matmul(anchors, memory.t())
    positive_similarities = similarities.gather(1, positive_labels.unsqueeze(1))

    negative_mask = memory_valid.unsqueeze(0).expand_as(similarities).clone()
    negative_mask.scatter_(1, positive_labels.unsqueeze(1), False)
    negative_similarities = similarities.masked_fill(~negative_mask, float('-inf'))
    negative_count = max(1, int(memory_valid.sum().item()) - 1)
    hard_negative_count = min(max(1, topk), negative_count)
    hard_negatives = negative_similarities.topk(hard_negative_count, dim=1).values

    logits = torch.cat((positive_similarities, hard_negatives), dim=1)
    logits = logits / max(temperature, 1e-6)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, targets)
    stats = {
        'anchors': logits.shape[0],
        'positive_similarity': positive_similarities.mean().detach(),
        'hard_negative_similarity': hard_negatives.mean().detach(),
    }
    return loss, stats


def _posterior_top1_infonce(
    anchors,
    labels,
    memory,
    posterior,
    transported_mass,
    topk,
    temperature,
    min_confidence,
):
    if anchors.numel() == 0:
        return None, {}

    memory = memory.detach()
    memory_valid = memory.norm(dim=1) > 1e-12
    targets = posterior[labels].detach()
    positive_confidence, positive_labels = targets.max(dim=1)
    sample_weights = transported_mass[labels].detach() * positive_confidence
    valid = (
        (positive_confidence >= min_confidence)
        & (positive_labels >= 0)
        & (positive_labels < memory.shape[0])
        & memory_valid[positive_labels]
        & (sample_weights > 0)
    )
    if not valid.any() or memory_valid.sum() < 2:
        return None, {}

    anchors = F.normalize(anchors[valid], dim=1)
    positive_labels = positive_labels[valid]
    positive_confidence = positive_confidence[valid]
    sample_weights = sample_weights[valid]
    memory = F.normalize(memory, dim=1)
    similarities = torch.matmul(anchors, memory.t())
    positive_similarities = similarities.gather(1, positive_labels.unsqueeze(1))

    negative_mask = memory_valid.unsqueeze(0).expand_as(similarities).clone()
    negative_mask.scatter_(1, positive_labels.unsqueeze(1), False)
    negative_similarities = similarities.masked_fill(~negative_mask, float('-inf'))
    negative_count = max(1, int(memory_valid.sum().item()) - 1)
    hard_negative_count = min(max(1, int(topk)), negative_count)
    hard_negatives = negative_similarities.topk(hard_negative_count, dim=1).values

    logits = torch.cat((positive_similarities, hard_negatives), dim=1)
    logits = logits / max(float(temperature), 1e-6)
    target_indices = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    per_sample = F.cross_entropy(logits, target_indices, reduction='none')
    sample_weights = sample_weights / sample_weights.max().clamp_min(1e-12)
    loss = (sample_weights * per_sample).sum() / sample_weights.sum().clamp_min(1e-12)
    stats = {
        'anchors': logits.new_tensor(float(logits.shape[0])),
        'confidence': positive_confidence.mean().detach(),
        'positive_similarity': positive_similarities.mean().detach(),
        'hard_negative_similarity': hard_negatives.mean().detach(),
    }
    return loss, stats


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


def _build_relations(r2i_pair_dict, i2r_pair_dict):
    common_dict, specific_dict, remain_dict = {}, {}, {}
    for rgb, ir in r2i_pair_dict.items():
        if ir in i2r_pair_dict and i2r_pair_dict[ir] == rgb:
            common_dict[rgb] = ir
        elif rgb not in i2r_pair_dict.values() and ir not in i2r_pair_dict:
            specific_dict[rgb] = ir
        else:
            remain_dict[rgb] = ir

    for ir, rgb in i2r_pair_dict.items():
        if common_dict.get(rgb) == ir:
            continue
        if rgb not in r2i_pair_dict.values() and ir not in r2i_pair_dict:
            specific_dict[rgb] = ir
        else:
            remain_dict[rgb] = ir
    return common_dict, specific_dict, remain_dict


def _log_relations(logger, diagnostics):
    if not diagnostics:
        return
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


def train(args, model: Model, dataset, *arg):
    epoch = arg[0]
    cma: CMA = arg[1]
    logger = arg[2]
    enable_phase1 = arg[3]
    use_rgmfd = bool(getattr(args, 'enable_rgmfd', 0)) and bool(
        getattr(model.model, 'enable_rgmfd', False)
    )

    if args.debug == 'wsl':
        cma.extract(
            args,
            model,
            dataset,
            collect_uprt=bool(getattr(args, 'enable_uprt', 0)) and not enable_phase1,
        )
        r2i_pair_dict, i2r_pair_dict = cma.get_label(epoch)
        common_dict, specific_dict, remain_dict = _build_relations(r2i_pair_dict, i2r_pair_dict)
        diagnostics = cma.relation_diagnostics(
            r2i_pair_dict, i2r_pair_dict, common_dict, specific_dict, remain_dict
        )
        _log_relations(logger, diagnostics)

        uprt_state = None
        uprt_scale = 0.0
        if getattr(args, 'enable_uprt', 0) and not enable_phase1:
            uprt_state = build_uprt_posterior(
                args, cma, common_dict, specific_dict, remain_dict, epoch=epoch
            )
            start_epoch = getattr(args, 'uprt_start_epoch', 5)
            if epoch >= start_epoch:
                warmup = max(1, getattr(args, 'uprt_warmup_epochs', 10))
                uprt_scale = min(1.0, (epoch - start_epoch + 1) / warmup)
            stats = uprt_state['stats']
            logger(
                'UPRT diag | active:{} scale:{:.2f} rel:{:.3f} mass:{:.6f} '
                'entropy:{:.3f} raw_ent:{:.3f} common_p:{:.3f} '
                'cand:{:.2f}% rec_cand:{:.2f}% prior:{:.2f}% rec:{:.3f}'.format(
                    int(uprt_scale > 0),
                    uprt_scale,
                    stats['mean_reliability'].item(),
                    stats['mean_mass'].item(),
                    stats['posterior_entropy'].item(),
                    stats['raw_posterior_entropy'].item(),
                    stats['common_probability'].item(),
                    stats['candidate_ratio'].item() * 100,
                    stats['recovery_candidate_ratio'].item() * 100,
                    stats['prior_ratio'].item() * 100,
                    stats['recovery_weight'].item(),
                )
            )

        hcl_state = {'active': False}
        if getattr(args, 'enable_hcl', 0):
            hcl_state = _update_hcl_state(args, epoch, cma, common_dict, not enable_phase1)
            logger(
                'HCL diag | active:{} streak:{} cov_c:{:.2f}% stable_cov:{:.2f}% stable_c:{:.2f}% scale:{:.2f}'.format(
                    int(hcl_state['active']),
                    hcl_state['streak'],
                    hcl_state['coverage'] * 100,
                    hcl_state['stable_coverage'] * 100,
                    hcl_state['stability'] * 100,
                    hcl_state['scale'],
                )
            )

            rgb_to_ir = torch.full((args.num_classes,), -1, dtype=torch.long, device=model.device)
            ir_to_rgb = torch.full((args.num_classes,), -1, dtype=torch.long, device=model.device)
            for rgb, ir in hcl_state['pairs']:
                rgb_to_ir[rgb] = ir
                ir_to_rgb[ir] = rgb

    model.set_train()
    meter = MultiItemAverageMeter()
    bt = args.batch_pidnum * args.pid_numsample
    rgb_loader, ir_loader = dataset.get_train_loader()

    for (rgb_imgs, ca_imgs, color_info), (ir_imgs, aug_imgs, ir_info) in zip(rgb_loader, ir_loader):
        optimizer = model.optimizer_phase1 if enable_phase1 else model.optimizer_phase2
        optimizer.zero_grad()

        rgb_imgs, ca_imgs = rgb_imgs.to(model.device), ca_imgs.to(model.device)
        color_imgs = torch.cat((rgb_imgs, ca_imgs), dim=0)
        rgb_gts, ir_gts = color_info[:, -1], ir_info[:, -1]
        rgb_ids = torch.cat((color_info[:, 1], color_info[:, 1])).to(model.device)
        ir_ids = ir_info[:, 1]

        if args.dataset == 'regdb':
            ir_imgs, aug_imgs = ir_imgs.to(model.device), aug_imgs.to(model.device)
            ir_imgs = torch.cat((ir_imgs, aug_imgs), dim=0)
            ir_ids = torch.cat((ir_ids, ir_ids)).to(model.device)
        else:
            ir_imgs = ir_imgs.to(model.device)
            ir_ids = ir_ids.to(model.device)

        if use_rgmfd:
            gap_features, bn_features, rgmfd_pack = model.model(color_imgs, ir_imgs, return_rg=True)
        else:
            gap_features, bn_features = model.model(color_imgs, ir_imgs)
            rgmfd_pack = None

        rgbcls_out, _ = model.classifier1(bn_features)
        ircls_out, _ = model.classifier2(bn_features)
        rgb_features, ir_features = gap_features[:2 * bt], gap_features[2 * bt:]
        r2r_cls, i2i_cls = rgbcls_out[:2 * bt], ircls_out[2 * bt:]

        if args.debug == 'wsl' and enable_phase1:
            r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
            i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
            r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
            i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
            total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
            meter.update({
                'r2r_id_loss': r2r_id_loss.data,
                'i2i_id_loss': i2i_id_loss.data,
                'r2r_tri_loss': r2r_tri_loss.data,
                'i2i_tri_loss': i2i_tri_loss.data,
            })
        elif args.debug == 'wsl':
            phase2_id_weight = getattr(args, 'phase2_id_weight', 1.0)
            r2r_id_loss = phase2_id_weight * model.pid_criterion(r2r_cls, rgb_ids)
            i2i_id_loss = phase2_id_weight * model.pid_criterion(i2i_cls, ir_ids)
            total_loss = r2r_id_loss + i2i_id_loss
            meter.update({'r2r_id_loss': r2r_id_loss.data, 'i2i_id_loss': i2i_id_loss.data})

            if getattr(args, 'enable_hcl', 0):
                cma.update(bn_features[:2 * bt], bn_features[2 * bt:], rgb_ids, ir_ids)
            if uprt_state is not None and uprt_scale > 0:
                shared_bn = rgmfd_pack['shared_bn'] if rgmfd_pack is not None else bn_features
                rgb_count = rgb_ids.shape[0]
                rgb_uprt_loss = posterior_cross_modal_loss(
                    shared_bn[:rgb_count],
                    rgb_ids,
                    cma.uprt_ir_memory,
                    uprt_state['v2t'],
                    uprt_state['vis_mass'],
                    getattr(args, 'uprt_temperature', 0.07),
                )
                ir_uprt_loss = posterior_cross_modal_loss(
                    shared_bn[rgb_count:],
                    ir_ids,
                    cma.uprt_vis_memory,
                    uprt_state['t2v'],
                    uprt_state['ir_mass'],
                    getattr(args, 'uprt_temperature', 0.07),
                )
                if rgb_uprt_loss is not None and ir_uprt_loss is not None:
                    uprt_loss = (
                        getattr(args, 'uprt_weight', 0.1)
                        * uprt_scale
                        * 0.5
                        * (rgb_uprt_loss + ir_uprt_loss)
                    )
                    total_loss = total_loss + uprt_loss
                    meter.update({'uprt_loss': uprt_loss.data})
                cls_weight = getattr(args, 'uprt_cls_weight', 0.0)
                if cls_weight > 0:
                    rgb_uprt_cls_loss = posterior_classifier_loss(
                        ircls_out[:rgb_count],
                        rgb_ids,
                        uprt_state['v2t'],
                        uprt_state['vis_mass'],
                        getattr(args, 'uprt_cls_temperature', 1.0),
                    )
                    ir_uprt_cls_loss = posterior_classifier_loss(
                        rgbcls_out[rgb_count:],
                        ir_ids,
                        uprt_state['t2v'],
                        uprt_state['ir_mass'],
                        getattr(args, 'uprt_cls_temperature', 1.0),
                    )
                    if rgb_uprt_cls_loss is not None and ir_uprt_cls_loss is not None:
                        uprt_cls_loss = (
                            cls_weight
                            * uprt_scale
                            * 0.5
                            * (rgb_uprt_cls_loss + ir_uprt_cls_loss)
                        )
                        total_loss = total_loss + uprt_cls_loss
                        meter.update({'uprt_cls_loss': uprt_cls_loss.data})
                hard_weight = getattr(args, 'uprt_hard_weight', 0.0)
                hard_start_epoch = getattr(args, 'uprt_hard_start_epoch', 40)
                if hard_weight > 0 and epoch >= hard_start_epoch:
                    hard_warmup = max(1, getattr(args, 'uprt_hard_warmup_epochs', 10))
                    hard_scale = min(1.0, (epoch - hard_start_epoch + 1) / hard_warmup)
                    rgb_hard_loss, rgb_hard_stats = _posterior_top1_infonce(
                        shared_bn[:rgb_count],
                        rgb_ids,
                        cma.uprt_ir_memory,
                        uprt_state['v2t'],
                        uprt_state['vis_mass'],
                        getattr(args, 'uprt_hard_topk', 20),
                        getattr(args, 'uprt_hard_temperature', 0.07),
                        getattr(args, 'uprt_hard_min_confidence', 0.85),
                    )
                    ir_hard_loss, ir_hard_stats = _posterior_top1_infonce(
                        shared_bn[rgb_count:],
                        ir_ids,
                        cma.uprt_vis_memory,
                        uprt_state['t2v'],
                        uprt_state['ir_mass'],
                        getattr(args, 'uprt_hard_topk', 20),
                        getattr(args, 'uprt_hard_temperature', 0.07),
                        getattr(args, 'uprt_hard_min_confidence', 0.85),
                    )
                    hard_losses = []
                    hard_stats = []
                    if rgb_hard_loss is not None:
                        hard_losses.append(rgb_hard_loss)
                        hard_stats.append(rgb_hard_stats)
                    if ir_hard_loss is not None:
                        hard_losses.append(ir_hard_loss)
                        hard_stats.append(ir_hard_stats)
                    if hard_losses:
                        uprt_hard_loss = hard_weight * uprt_scale * hard_scale * sum(hard_losses) / len(hard_losses)
                        total_loss = total_loss + uprt_hard_loss
                        meter.update({
                            'uprt_hard_loss': uprt_hard_loss.data,
                            'uprt_hard_conf': sum(s['confidence'] for s in hard_stats) / len(hard_stats),
                            'uprt_hard_pos_sim': sum(s['positive_similarity'] for s in hard_stats) / len(hard_stats),
                            'uprt_hard_neg_sim': sum(s['hard_negative_similarity'] for s in hard_stats) / len(hard_stats),
                            'uprt_hard_anchors': sum(s['anchors'] for s in hard_stats),
                        })
            if hcl_state['active']:
                hcl_losses = []
                hcl_stats = []
                rgb_hcl_loss, rgb_stats = _hard_negative_infonce(
                    bn_features[:2 * bt],
                    rgb_to_ir[rgb_ids],
                    cma.ir_memory,
                    getattr(args, 'hcl_topk', 20),
                    getattr(args, 'hcl_temperature', 0.07),
                )
                ir_hcl_loss, ir_stats = _hard_negative_infonce(
                    bn_features[2 * bt:],
                    ir_to_rgb[ir_ids],
                    cma.vis_memory,
                    getattr(args, 'hcl_topk', 20),
                    getattr(args, 'hcl_temperature', 0.07),
                )
                if rgb_hcl_loss is not None:
                    hcl_losses.append(rgb_hcl_loss)
                    hcl_stats.append(rgb_stats)
                if ir_hcl_loss is not None:
                    hcl_losses.append(ir_hcl_loss)
                    hcl_stats.append(ir_stats)
                if hcl_losses:
                    hcl_loss = (
                        getattr(args, 'hcl_weight', 0.1)
                        * hcl_state['scale']
                        * sum(hcl_losses) / len(hcl_losses)
                    )
                    total_loss = total_loss + hcl_loss
                    meter.update({
                        'hcl_loss': hcl_loss.data,
                        'hcl_pos_sim': sum(s['positive_similarity'] for s in hcl_stats) / len(hcl_stats),
                        'hcl_hard_neg_sim': sum(s['hard_negative_similarity'] for s in hcl_stats) / len(hcl_stats),
                    })
        elif args.debug == 'baseline':
            r2r_id_loss = model.pid_criterion(r2r_cls, rgb_ids)
            i2i_id_loss = model.pid_criterion(i2i_cls, ir_ids)
            r2r_tri_loss = args.tri_weight * model.tri_criterion(rgb_features, rgb_ids)
            i2i_tri_loss = args.tri_weight * model.tri_criterion(ir_features, ir_ids)
            total_loss = r2r_id_loss + i2i_id_loss + r2r_tri_loss + i2i_tri_loss
            meter.update({
                'r2r_id_loss': r2r_id_loss.data,
                'i2i_id_loss': i2i_id_loss.data,
                'r2r_tri_loss': r2r_tri_loss.data,
                'i2i_tri_loss': i2i_tri_loss.data,
            })
        elif args.debug == 'sl':
            rgb_gts = torch.cat((rgb_gts, rgb_gts)).to(model.device)
            if args.dataset == 'regdb':
                ir_gts = torch.cat((ir_gts, ir_gts)).to(model.device)
            else:
                ir_gts = ir_gts.to(model.device)
            gts = torch.cat((rgb_gts, ir_gts))
            id_loss = model.pid_criterion(rgbcls_out, gts)
            tri_loss = args.tri_weight * model.tri_criterion(gap_features, gts)
            total_loss = id_loss + tri_loss
            meter.update({'id_loss': id_loss.data, 'tri_loss': tri_loss.data})
        else:
            raise RuntimeError('Debug mode {} not found!'.format(args.debug))

        if epoch >= getattr(args, 'rgmfd_start_epoch', 0):
            total_loss = _add_loss_dict(
                total_loss,
                _rgmfd_regularization_losses(args, rgmfd_pack),
                meter,
            )
        total_loss.backward()
        optimizer.step()

    return meter.get_val(), meter.get_str()
