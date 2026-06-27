import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter, OrderedDict
from utils import fliplr
class CMA(nn.Module):
    '''
    Cross modal Match Aggregation
    '''
    def __init__(self, args):
        super(CMA, self).__init__()
        # self.inited = False
        self.device = torch.device(args.device)
        self.not_saved = True
        # self.threshold = 0.8
        self.num_classes = args.num_classes
        self.T = args.temperature # softmax temperature
        self.sigma = args.sigma # momentum update factor
        self.cre_sample_rate = getattr(args, "cre_sample_rate", 1.0)
        # memory of visible and infrared modal
        self.register_buffer('vis_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('ir_memory',torch.zeros(self.num_classes,2048))

    @torch.no_grad()
    # def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode):
    def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode, rgb_features=None, ir_features=None,
             rgb_gt=None, ir_gt=None):
    # vis: vis sample(v2i scores or vis features) ir: ir sample
        self.mode = mode
        self.not_saved = False
        if self.mode != 'scores' and self.mode != 'features':
            raise ValueError('invalid mode!')
        elif self.mode == 'scores': # predict scores
            vis = torch.nn.functional.softmax(self.T*vis,dim=1)
            ir = torch.nn.functional.softmax(self.T*ir,dim=1)
        ###############################
        # save features in memory bank
        if rgb_features is not None and ir_features is not None:
            # Prepare empty memory banks on the device
            self.vis_memory = self.vis_memory.to(self.device)
            self.ir_memory = self.ir_memory.to(self.device)
            
            # Get unique labels and process RGB and IR features
            label_set = torch.unique(rgb_ids)
            
            for label in label_set:
                # Select RGB features for the current label
                rgb_mask = (rgb_ids == label)
                ir_mask = (ir_ids == label)
                # .any() check True in bool tensor
                if rgb_mask.any():
                    rgb_selected = rgb_features[rgb_mask]
                    self.vis_memory[label] = rgb_selected.mean(dim=0)
                
                if ir_mask.any():
                    ir_selected = ir_features[ir_mask]
                    self.ir_memory[label] = ir_selected.mean(dim=0)
        ################################
        vis = vis.detach().cpu().numpy()
        ir = ir.detach().cpu().numpy()
        rgb_ids, ir_ids = rgb_ids.cpu(), ir_ids.cpu()
            
        self.vis, self.ir = vis, ir
        self.rgb_ids, self.ir_ids = rgb_ids, ir_ids
        self.rgb_idx, self.ir_idx = rgb_idx, ir_idx
        if rgb_gt is not None and ir_gt is not None:
            self.rgb_gt = rgb_gt.detach().cpu()
            self.ir_gt = ir_gt.detach().cpu()
            self.rgb_gt_by_label = self._label_to_gt(self.rgb_ids, self.rgb_gt)
            self.ir_gt_by_label = self._label_to_gt(self.ir_ids, self.ir_gt)
        
    @torch.no_grad()
    def update(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        rgb_set = torch.unique(rgb_labels)
        ir_set = torch.unique(ir_labels)
        for i in rgb_set:
            rgb_mask = (rgb_labels == i)
            selected_rgb = rgb_feats[rgb_mask].mean(dim=0)
            self.vis_memory[i] = (1-self.sigma)*self.vis_memory[i] + self.sigma * selected_rgb
        for i in ir_set:
            ir_mask = (ir_labels == i)
            selected_ir = ir_feats[ir_mask].mean(dim=0)
            self.ir_memory[i] = (1-self.sigma)*self.ir_memory[i] + self.sigma * selected_ir

    def get_label(self, epoch=None):
        if self.not_saved:# pass if 
            pass
        else:
            print('get match labels')
            if self.mode == 'features':
                dists = np.matmul(self.vis, self.ir.T)
                v2i_dict, i2v_dict = self._get_label(dists,'dist', epoch)

            elif self.mode == 'scores':
                v2i_dict, _ = self._get_label(self.vis,'rgb', epoch)
                i2v_dict, _ = self._get_label(self.ir,'ir', epoch)
                self.v2i = v2i_dict
                self.i2v = i2v_dict
            return v2i_dict, i2v_dict

    def _get_label(self,dists,mode,epoch=None):
        return self._get_count_label(dists, mode)

    def _get_count_label(self,dists,mode):
        sample_rate = self.cre_sample_rate
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]# flat to 1d and sort
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)# sort index return to 2d, like ([0,1,2],[1,2,0])
        idx1, idx2 = sorted_2d[0], sorted_2d[1]# sorted idx of dim0 and dim1
        dists = dists[idx1, idx2]
        idx_length = int(np.ceil(sample_rate*dists.shape[0]/self.num_classes))
        idx_length = max(1, min(idx_length, dists.shape[0]))
        dists = dists[:idx_length]

        if mode=='dist': # multiply the instance features of the two modalities
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                            np.array(self.ir_ids)[idx2[:idx_length]])]
            
        elif mode=='rgb': # classify score of RGB (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]

        elif mode=='ir': # classify score of IR (v2i)
            convert_label = [(i,j) for i,j in zip(np.array(self.ir_ids)[idx1[:idx_length]],\
                                                  idx2[:idx_length])]
        else:
            raise AttributeError('invalid mode!')
        convert_label_cnt = Counter(convert_label)
        convert_label_cnt_sorted = sorted(convert_label_cnt.items(),key = lambda x:x[1],reverse = True)
        length = len(convert_label_cnt_sorted)
        in_rgb_label=[]
        in_ir_label=[]
        v2i = OrderedDict()
        i2v = OrderedDict()

        length_ratio = 1
        for i in range(int(length*length_ratio)):
            key = convert_label_cnt_sorted[i][0] 
            value = convert_label_cnt_sorted[i][1]
            # if key[0] == -1 or key[1] == -1:
            #     continue
            if key[0] in in_rgb_label or key[1] in in_ir_label:
                continue
            in_rgb_label.append(key[0])
            in_ir_label.append(key[1])
            v2i[key[0]] = key[1]
            i2v[key[1]] = key[0]
            # v2i[key[0]][key[1]] = 1
            
        return v2i, i2v # only v2i/i2v is used in scores mode

    def _label_to_gt(self, labels, gts):
        label_to_gt = {}
        for label, gt in zip(labels.tolist(), gts.tolist()):
            label_to_gt[int(label)] = int(gt)
        return label_to_gt

    def _pair_quality(self, pair_dict, direction='r2i'):
        correct = 0
        total = 0
        missing = 0
        for src, dst in pair_dict.items():
            if direction == 'r2i':
                src_gt = self.rgb_gt_by_label.get(int(src), None)
                dst_gt = self.ir_gt_by_label.get(int(dst), None)
            elif direction == 'i2r':
                src_gt = self.ir_gt_by_label.get(int(src), None)
                dst_gt = self.rgb_gt_by_label.get(int(dst), None)
            else:
                raise AttributeError('invalid direction!')
            if src_gt is None or dst_gt is None:
                missing += 1
                continue
            total += 1
            correct += int(src_gt == dst_gt)
        acc = correct / total if total > 0 else 0.0
        return {'num': len(pair_dict), 'valid': total, 'correct': correct, 'acc': acc, 'missing': missing}

    def relation_diagnostics(self, r2i_pair_dict, i2r_pair_dict, common_dict, specific_dict, remain_dict):
        if not hasattr(self, 'rgb_gt_by_label') or not hasattr(self, 'ir_gt_by_label'):
            return {}

        diagnostics = {
            'r2i': self._pair_quality(r2i_pair_dict, 'r2i'),
            'i2r': self._pair_quality(i2r_pair_dict, 'i2r'),
            'common': self._pair_quality(common_dict, 'r2i'),
            'specific': self._pair_quality(specific_dict, 'r2i'),
            'remain': self._pair_quality(remain_dict, 'r2i'),
        }
        current_common = set((int(r), int(i)) for r, i in common_dict.items())
        if hasattr(self, 'prev_common_pairs') and len(current_common) > 0:
            diagnostics['common_stability'] = len(current_common & self.prev_common_pairs) / len(current_common)
        else:
            diagnostics['common_stability'] = 0.0
        diagnostics['common_coverage'] = len(common_dict) / self.num_classes
        diagnostics['all_coverage'] = len(set(common_dict.keys()) | set(specific_dict.keys()) | set(remain_dict.keys())) / self.num_classes
        self.prev_common_pairs = current_common
        return diagnostics

    def _ids_to_numpy(self, ids):
        if hasattr(ids, 'numpy'):
            return ids.numpy()
        return np.asarray(ids)

    def _score_stats(self, scores, labels, source_label, target_label):
        labels = self._ids_to_numpy(labels)
        mask = labels == int(source_label)
        if not np.any(mask):
            return 0.0, 0.0

        rows = scores[mask]
        target_label = int(target_label)
        if target_label < 0 or target_label >= rows.shape[1]:
            return 0.0, 0.0

        target_probs = rows[:, target_label]
        if rows.shape[1] == 1:
            margins = target_probs
        else:
            other_scores = rows.copy()
            other_scores[:, target_label] = -np.inf
            margins = target_probs - other_scores.max(axis=1)
        return float(target_probs.mean()), float(np.maximum(margins, 0.0).mean())

    def _prototype_score(self, rgb_label, ir_label):
        rgb_proto = self.vis_memory[int(rgb_label)].detach()
        ir_proto = self.ir_memory[int(ir_label)].detach()
        if rgb_proto.norm().item() <= 1e-12 or ir_proto.norm().item() <= 1e-12:
            return 0.5
        sim = F.cosine_similarity(rgb_proto.unsqueeze(0), ir_proto.unsqueeze(0), dim=1).item()
        return max(0.0, min(1.0, 0.5 * (sim + 1.0)))

    def _relation_consistency_score(self, relation_type, key, previous_pairs):
        type_priors = {
            'common': 1.0,
            'specific': 0.65,
            'remain': 0.35,
        }
        temporal_score = 0.0
        if key in previous_pairs.get(relation_type, set()):
            temporal_score = 1.0
        elif any(key in pairs for pairs in previous_pairs.values()):
            temporal_score = 0.6
        return 0.5 * type_priors.get(relation_type, 0.5) + 0.5 * temporal_score

    def _rgrc_score(self, args, relation_type, rgb_label, ir_label, previous_pairs, previous_scores):
        r2i_prob, r2i_margin = self._score_stats(self.vis, self.rgb_ids, rgb_label, ir_label)
        i2r_prob, i2r_margin = self._score_stats(self.ir, self.ir_ids, ir_label, rgb_label)
        prob_score = 0.5 * (r2i_prob + i2r_prob)
        margin_score = 0.5 * (r2i_margin + i2r_margin)
        prob_weight = getattr(args, 'rgrc_expert_prob_weight', 0.7)
        margin_weight = getattr(args, 'rgrc_expert_margin_weight', 0.3)
        expert_norm = max(1e-6, prob_weight + margin_weight)
        expert_score = (prob_weight * prob_score + margin_weight * margin_score) / expert_norm
        shared_score = self._prototype_score(rgb_label, ir_label)
        key = (int(rgb_label), int(ir_label))
        consistency_score = self._relation_consistency_score(relation_type, key, previous_pairs)

        alpha = getattr(args, 'rgrc_alpha', 0.4)
        beta = getattr(args, 'rgrc_beta', 0.3)
        gamma = getattr(args, 'rgrc_gamma', 0.3)
        norm = max(1e-6, alpha + beta + gamma)
        raw_score = (
            alpha * expert_score
            + beta * shared_score
            + gamma * consistency_score
        ) / norm
        raw_score = max(0.0, min(1.0, raw_score))

        momentum = getattr(args, 'rgrc_momentum', 0.8)
        if key in previous_scores:
            raw_score = momentum * previous_scores[key] + (1.0 - momentum) * raw_score
        return max(0.0, min(1.0, raw_score))

    def _rgrc_weight(self, score, min_weight):
        return max(0.0, min(1.0, min_weight + (1.0 - min_weight) * score))

    def _rgrc_bucket_stats(self, scores, weights, kept, total):
        if total == 0:
            return {
                'num': 0,
                'kept': 0,
                'raw_mean': 0.0,
                'weight_mean': 0.0,
                'weight_min': 0.0,
                'weight_max': 0.0,
            }
        return {
            'num': total,
            'kept': kept,
            'raw_mean': float(np.mean(scores)) if scores else 0.0,
            'weight_mean': float(np.mean(weights)) if weights else 0.0,
            'weight_min': float(np.min(weights)) if weights else 0.0,
            'weight_max': float(np.max(weights)) if weights else 0.0,
        }

    def relation_correction(self, args, common_dict, specific_dict, remain_dict, device):
        relation_mats = {
            'common': torch.zeros((self.num_classes, self.num_classes), device=device),
            'specific': torch.zeros((self.num_classes, self.num_classes), device=device),
            'remain': torch.zeros((self.num_classes, self.num_classes), device=device),
        }
        relation_dicts = {
            'common': common_dict,
            'specific': specific_dict,
            'remain': remain_dict,
        }
        thresholds = {
            'common': getattr(args, 'rgrc_common_threshold', 0.0),
            'specific': getattr(args, 'rgrc_specific_threshold', 0.35),
            'remain': getattr(args, 'rgrc_remain_threshold', 0.55),
        }
        min_weights = {
            'common': getattr(args, 'rgrc_common_min_weight', 0.6),
            'specific': getattr(args, 'rgrc_specific_min_weight', 0.35),
            'remain': getattr(args, 'rgrc_remain_min_weight', 0.2),
        }

        previous_pairs = getattr(self, 'rgrc_prev_pairs', {})
        previous_scores = getattr(self, 'rgrc_score_memory', {})
        next_pairs = {}
        next_scores = {}
        stats = {}

        for relation_type, pair_dict in relation_dicts.items():
            scores = []
            weights = []
            kept = 0
            current_pairs = set()
            for rgb_label, ir_label in pair_dict.items():
                key = (int(rgb_label), int(ir_label))
                current_pairs.add(key)
                score = self._rgrc_score(
                    args, relation_type, rgb_label, ir_label, previous_pairs, previous_scores
                )
                scores.append(score)
                next_scores[key] = score
                if score < thresholds[relation_type]:
                    continue
                weight = self._rgrc_weight(score, min_weights[relation_type])
                relation_mats[relation_type][key[0], key[1]] = weight
                weights.append(weight)
                kept += 1
            next_pairs[relation_type] = current_pairs
            stats[relation_type] = self._rgrc_bucket_stats(scores, weights, kept, len(pair_dict))

        self.rgrc_prev_pairs = next_pairs
        self.rgrc_score_memory = next_scores
        return relation_mats, stats

    def relation_reliability(self, args, common_dict, device):
        weights = torch.zeros((self.num_classes, self.num_classes), device=device)
        if len(common_dict) == 0:
            return weights, {
                'num': 0,
                'raw_mean': 0.0,
                'weight_mean': 0.0,
                'weight_min': 0.0,
                'weight_max': 0.0,
            }

        prob_weight = getattr(args, 'raecl_prob_weight', 0.45)
        margin_weight = getattr(args, 'raecl_margin_weight', 0.20)
        proto_weight = getattr(args, 'raecl_proto_weight', 0.25)
        stability_weight = getattr(args, 'raecl_stability_weight', 0.10)
        norm = max(1e-6, prob_weight + margin_weight + proto_weight + stability_weight)
        momentum = getattr(args, 'raecl_momentum', 0.8)
        min_weight = getattr(args, 'raecl_min_weight', 0.6)

        previous_pairs = getattr(self, 'raecl_prev_common_pairs', set())
        previous_scores = getattr(self, 'raecl_score_memory', {})
        current_pairs = set()
        next_scores = {}
        raw_values = []
        weight_values = []

        for rgb_label, ir_label in common_dict.items():
            key = (int(rgb_label), int(ir_label))
            current_pairs.add(key)
            r2i_prob, r2i_margin = self._score_stats(self.vis, self.rgb_ids, rgb_label, ir_label)
            i2r_prob, i2r_margin = self._score_stats(self.ir, self.ir_ids, ir_label, rgb_label)
            prob_score = 0.5 * (r2i_prob + i2r_prob)
            margin_score = 0.5 * (r2i_margin + i2r_margin)
            proto_score = self._prototype_score(rgb_label, ir_label)
            stability_score = 1.0 if key in previous_pairs else 0.0
            raw_score = (
                prob_weight * prob_score
                + margin_weight * margin_score
                + proto_weight * proto_score
                + stability_weight * stability_score
            ) / norm
            raw_score = max(0.0, min(1.0, raw_score))
            score = raw_score
            if key in previous_scores:
                score = momentum * previous_scores[key] + (1.0 - momentum) * raw_score
            score = max(0.0, min(1.0, score))
            final_weight = min_weight + (1.0 - min_weight) * score
            weights[key[0], key[1]] = final_weight
            next_scores[key] = score
            raw_values.append(raw_score)
            weight_values.append(final_weight)

        self.raecl_prev_common_pairs = current_pairs
        self.raecl_score_memory = next_scores

        return weights, {
            'num': len(weight_values),
            'raw_mean': float(np.mean(raw_values)),
            'weight_mean': float(np.mean(weight_values)),
            'weight_min': float(np.min(weight_values)),
            'weight_max': float(np.max(weight_values)),
        }

    def extract(self, args, model, dataset):
        '''
        Output: BN_features, labels, cls
        '''
        # save epoch
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader() 
        with torch.no_grad():
            
            rgb_features, rgb_labels, rgb_gt, r2i_cls, rgb_idx = self._extract_feature(model, rgb_loader,'rgb')
            ir_features, ir_labels, ir_gt, i2r_cls, ir_idx = self._extract_feature(model, ir_loader,'ir')

        # # //match by cls and save features to memory bank
        self.save(r2i_cls, i2r_cls, rgb_labels, ir_labels, rgb_idx,\
                 ir_idx, 'scores', rgb_features, ir_features, rgb_gt, ir_gt)
        
    def _extract_feature(self, model, loader, modal):

        print('extracting {} features'.format(modal))

        saved_features, saved_labels, saved_cls= None, None, None
        saved_gts, saved_idx= None, None
        for imgs_list, infos in loader:
            labels = infos[:,1]
            idx = infos[:,0]
            gts = infos[:,-1].to(model.device)
            if imgs_list.__class__.__name__ != 'list':
                imgs = imgs_list
                imgs, labels, idx = \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            else:
                ori_imgs, ca_imgs = imgs_list[0], imgs_list[1]
                if len(ori_imgs.shape) < 4:
                    ori_imgs = ori_imgs.unsqueeze(0)
                    ca_imgs = ca_imgs.unsqueeze(0)

                imgs = torch.cat((ori_imgs,ca_imgs),dim=0)
                labels = torch.cat((labels,labels),dim=0)
                idx = torch.cat((idx,idx),dim=0)
                gts= torch.cat((gts,gts),dim=0).to(model.device)
                imgs, labels, idx= \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            _, bn_features = model.model(imgs) # _:gap feature

            if modal == 'rgb':
                cls, l2_features = model.classifier2(bn_features)
            elif modal == 'ir':
                cls, l2_features = model.classifier1(bn_features)
            l2_features = l2_features.detach().cpu()

            if saved_features is None: 
                # saved_features, saved_labels, saved_cls, saved_idx = l2_features, labels, cls, idx
                saved_features, saved_labels, saved_cls, saved_idx = bn_features, labels, cls, idx

                saved_gts = gts
            else:
                # saved_features = torch.cat((saved_features, l2_features), dim=0)
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)

                saved_gts = torch.cat((saved_gts, gts), dim=0)
        return saved_features, saved_labels, saved_gts, saved_cls, saved_idx
