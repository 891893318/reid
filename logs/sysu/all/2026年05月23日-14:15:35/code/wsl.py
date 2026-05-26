import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter, OrderedDict
from sklearn.preprocessing import normalize
import time
import pickle
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
        self.use_confident_cre = bool(getattr(args, "cre_confidence", 1))
        self.cre_sample_rate = getattr(args, "cre_sample_rate", 1.0)
        self.cre_count_weight = getattr(args, "cre_count_weight", 1.0)
        self.cre_prob_weight = getattr(args, "cre_prob_weight", 0.5)
        self.cre_margin_weight = getattr(args, "cre_margin_weight", 0.5)
        self.cre_entropy_weight = getattr(args, "cre_entropy_weight", 0.3)
        self.cre_proto_weight = getattr(args, "cre_proto_weight", 0.5)
        self.cre_min_margin_start = getattr(args, "cre_min_margin_start", 0.0)
        self.cre_min_margin_end = getattr(args, "cre_min_margin_end", 0.0)
        self.cre_margin_decay_epoch = max(1, getattr(args, "cre_margin_decay_epoch", 30))
        self.relation_memory = {}
        self.relation_streak = {}
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
        if self.use_confident_cre and mode in ('rgb', 'ir'):
            return self._get_confident_label(dists, mode, epoch)
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

    def _get_confident_label(self, dists, mode, epoch=None):
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]
        idx1, idx2 = np.unravel_index(sorted_1d, dists_shape)
        flat_scores = dists[idx1, idx2]
        idx_length = int(np.ceil(self.cre_sample_rate * flat_scores.shape[0] / self.num_classes))
        idx_length = max(1, min(idx_length, flat_scores.shape[0]))

        entropy_conf = self._entropy_confidence(dists)
        margins = self._entry_margins(dists, idx1[:idx_length], idx2[:idx_length])
        proto_sim = self._prototype_similarity()
        label_stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0])

        for row, col, score, margin in zip(
            idx1[:idx_length], idx2[:idx_length], flat_scores[:idx_length], margins
        ):
            key = self._convert_pair(row, col, mode)
            label_stats[key][0] += 1
            label_stats[key][1] += float(score)
            label_stats[key][2] += float(max(margin, 0.0))
            label_stats[key][3] += float(entropy_conf[row])

        candidates = self._score_candidates(label_stats, proto_sim, mode, epoch)
        if len(candidates) == 0:
            return self._get_count_label(dists, mode)

        in_rgb_label=[]
        in_ir_label=[]
        v2i = OrderedDict()
        i2v = OrderedDict()

        for key, _, _, _, _, _, _ in candidates:
            if key[0] in in_rgb_label or key[1] in in_ir_label:
                continue
            in_rgb_label.append(key[0])
            in_ir_label.append(key[1])
            v2i[key[0]] = key[1]
            i2v[key[1]] = key[0]

        print(
            'confident CRE {}: candidates={}, matches={}, min_margin={:.4f}'.format(
                mode, len(candidates), len(v2i), self._dynamic_min_margin(epoch)
            )
        )
        return v2i, i2v

    def _score_candidates(self, label_stats, proto_sim, mode, epoch=None):
        if len(label_stats) == 0:
            return []

        max_count = max(v[0] for v in label_stats.values())
        min_margin = self._dynamic_min_margin(epoch)
        candidates = []
        fallback_candidates = []

        for key, (cnt, prob_sum, margin_sum, entropy_sum) in label_stats.items():
            count_score = np.log1p(cnt) / np.log1p(max_count)
            prob_score = prob_sum / cnt
            margin_score = margin_sum / cnt
            entropy_score = entropy_sum / cnt
            proto_score = self._lookup_proto_score(proto_sim, key, mode)
            relation_score = (
                self.cre_count_weight * count_score
                + self.cre_prob_weight * prob_score
                + self.cre_margin_weight * margin_score
                + self.cre_entropy_weight * entropy_score
                + self.cre_proto_weight * proto_score
            )
            item = (key, relation_score, cnt, prob_score, margin_score, entropy_score, proto_score)
            fallback_candidates.append(item)
            if margin_score >= min_margin:
                candidates.append(item)

        if len(candidates) == 0:
            candidates = fallback_candidates
        # Keep Count Priority and raw expert score as the primary decision
        # rules; confidence only refines truly close candidates.
        return sorted(candidates, key=lambda x:(x[2], x[3], x[1]), reverse=True)

    def _convert_pair(self, row, col, mode):
        if mode == 'rgb':
            return (int(self.rgb_ids[row].item()), int(col))
        if mode == 'ir':
            return (int(self.ir_ids[row].item()), int(col))
        raise AttributeError('invalid mode!')

    def _entry_margins(self, dists, rows, cols):
        if dists.shape[1] == 1:
            return dists[rows, cols]

        top2 = np.partition(dists, -2, axis=1)[:, -2:]
        second_best = top2[:, 0]
        best = top2[:, 1]
        scores = dists[rows, cols]
        other_best = np.where(np.isclose(scores, best[rows]), second_best[rows], best[rows])
        return scores - other_best

    def _entropy_confidence(self, dists):
        eps = 1e-12
        probs = np.clip(dists, eps, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        max_entropy = np.log(max(dists.shape[1], 2))
        return 1.0 - entropy / max_entropy

    def _prototype_similarity(self):
        vis_memory = self.vis_memory.detach()
        ir_memory = self.ir_memory.detach()
        vis_memory = torch.nn.functional.normalize(vis_memory, dim=1)
        ir_memory = torch.nn.functional.normalize(ir_memory, dim=1)
        sim = torch.matmul(vis_memory, ir_memory.t()).detach().cpu().numpy()
        return (sim + 1.0) / 2.0

    def _lookup_proto_score(self, proto_sim, key, mode):
        if mode == 'rgb':
            return float(proto_sim[key[0], key[1]])
        if mode == 'ir':
            return float(proto_sim[key[1], key[0]])
        return 0.0

    def _dynamic_min_margin(self, epoch=None):
        if epoch is None:
            return self.cre_min_margin_end
        progress = min(max(float(epoch), 0.0) / self.cre_margin_decay_epoch, 1.0)
        return self.cre_min_margin_start + (
            self.cre_min_margin_end - self.cre_min_margin_start
        ) * progress

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

    def update_relation_memory(self, common_dict, specific_dict, remain_dict, args):
        momentum = getattr(args, 'trrm_momentum', 0.8)
        current_scores = {}
        for r, i in common_dict.items():
            current_scores[(int(r), int(i))] = 1.0
        for r, i in specific_dict.items():
            current_scores[(int(r), int(i))] = 0.6
        for r, i in remain_dict.items():
            current_scores[(int(r), int(i))] = 0.2

        previous_keys = set(self.relation_memory.keys())
        current_keys = set(current_scores.keys())
        for key in current_keys:
            old_score = self.relation_memory.get(key, 0.0)
            self.relation_memory[key] = momentum * old_score + (1.0 - momentum) * current_scores[key]
            self.relation_streak[key] = self.relation_streak.get(key, 0) + 1

        for key in previous_keys - current_keys:
            self.relation_memory[key] = momentum * self.relation_memory[key]
            self.relation_streak[key] = 0
            if self.relation_memory[key] < 1e-4:
                self.relation_memory.pop(key, None)
                self.relation_streak.pop(key, None)

        if len(current_keys) == 0:
            return {'active': 0, 'avg_memory': 0.0, 'avg_streak': 0.0}
        avg_memory = sum(self.relation_memory[k] for k in current_keys) / len(current_keys)
        avg_streak = sum(self.relation_streak[k] for k in current_keys) / len(current_keys)
        return {'active': len(current_keys), 'avg_memory': avg_memory, 'avg_streak': avg_streak}

    def filter_specific_relations(self, specific_dict, epoch, args):
        if not getattr(args, 'enable_trrm', 1):
            return specific_dict
        if epoch < getattr(args, 'trrm_specific_start', 20):
            return {}

        min_streak = getattr(args, 'trrm_specific_streak', 2)
        min_memory = getattr(args, 'trrm_specific_threshold', 0.2)
        max_ratio = getattr(args, 'trrm_specific_max_ratio', 0.5)
        max_num = getattr(args, 'trrm_specific_max_num', 0)
        candidates = []
        for r, i in specific_dict.items():
            key = (int(r), int(i))
            memory = self.relation_memory.get(key, 0.0)
            streak = self.relation_streak.get(key, 0)
            if streak >= min_streak and memory >= min_memory:
                candidates.append((r, i, memory, streak))

        candidates.sort(key=lambda item: (item[2], item[3]), reverse=True)
        keep_num = len(candidates)
        if 0 < max_ratio < 1:
            keep_num = min(keep_num, max(1, int(np.ceil(len(specific_dict) * max_ratio))))
        if max_num > 0:
            keep_num = min(keep_num, max_num)

        filtered = OrderedDict()
        for r, i, _, _ in candidates[:keep_num]:
            filtered[r] = i
        return filtered

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
