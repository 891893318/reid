import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter, OrderedDict

class CMA(nn.Module):
    '''
    Cross modal Match Aggregation
    '''
    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device(args.device)
        self.not_saved = True
        self.num_classes = args.num_classes
        self.T = args.temperature # softmax temperature
        self.sigma = args.sigma # momentum update factor
        self.pred_weight = getattr(args, 'bcpt_pred_weight', 0.45)
        self.feat_weight = getattr(args, 'bcpt_feat_weight', 0.40)
        self.conf_weight = getattr(args, 'bcpt_conf_weight', 0.15)
        self.ot_reg = getattr(args, 'bcpt_ot_reg', 0.07)
        self.ot_iters = getattr(args, 'bcpt_ot_iters', 30)
        self.high_th = getattr(args, 'bcpt_high_th', 0.55)
        self.mid_th = getattr(args, 'bcpt_mid_th', 0.25)
        self.topk = getattr(args, 'bcpt_topk', 3)
        # memory of visible and infrared modal
        self.register_buffer('vis_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('ir_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('transport_row', torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer('transport_col', torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer('common_rm', torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer('specific_rm', torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer('remain_rm', torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer('remain_col_rm', torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer('row_conf', torch.zeros(self.num_classes))
        self.register_buffer('col_conf', torch.zeros(self.num_classes))
        self.register_buffer('row_assign', torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer('col_assign', torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer('common_row_mask', torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer('common_col_mask', torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer('specific_row_mask', torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer('specific_col_mask', torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer('remain_row_mask', torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer('remain_col_mask', torch.zeros(self.num_classes, dtype=torch.bool))

    @torch.no_grad()
    def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode, rgb_features=None, ir_features=None):
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
            
            # Get the union of visible and infrared labels for prototype updates.
            label_set = torch.unique(torch.cat((rgb_ids, ir_ids), dim=0))
            
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
        self.vis_scores = vis.detach().float().cpu()
        self.ir_scores = ir.detach().float().cpu()
        self.rgb_ids = rgb_ids.detach().long().cpu()
        self.ir_ids = ir_ids.detach().long().cpu()
            
        self.rgb_idx, self.ir_idx = rgb_idx, ir_idx
        
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
                dists = np.matmul(self.vis_scores.numpy(), self.ir_scores.numpy().T)
                v2i_dict, i2v_dict = self._get_label(dists,'dist')
            elif self.mode == 'scores':
                v2i_dict, i2v_dict = self._get_bcpt_label()
                self.v2i = v2i_dict
                self.i2v = i2v_dict
            return v2i_dict, i2v_dict

    @torch.no_grad()
    def _build_class_stats(self, scores, labels):
        scores = scores.to(self.device)
        labels = labels.to(self.device)
        proto = torch.zeros(self.num_classes, scores.size(1), device=self.device)
        counts = torch.bincount(labels, minlength=self.num_classes).float().to(self.device)
        proto.index_add_(0, labels, scores)
        proto = proto / counts.unsqueeze(1).clamp_min(1.0)
        entropy = -(proto.clamp_min(1e-12) * proto.clamp_min(1e-12).log()).sum(dim=1)
        active = counts > 0
        return proto, counts, entropy, active

    @torch.no_grad()
    def _sinkhorn(self, cost, a, b, mask):
        kernel = torch.exp(-cost / max(self.ot_reg, 1e-4)) * mask.float()
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        eps = 1e-8
        for _ in range(self.ot_iters):
            kv = kernel @ v + eps
            u = torch.where(a > 0, a / kv, torch.zeros_like(a))
            ktu = kernel.t() @ u + eps
            v = torch.where(b > 0, b / ktu, torch.zeros_like(b))
        transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        return transport * mask.float()

    @torch.no_grad()
    def _topk_mask(self, matrix):
        if matrix.numel() == 0:
            return matrix
        k = min(self.topk, matrix.size(1))
        values, indices = torch.topk(matrix, k=k, dim=1)
        sparse = torch.zeros_like(matrix)
        sparse.scatter_(1, indices, values)
        return sparse

    @torch.no_grad()
    def _get_bcpt_label(self):
        vis_proto, vis_count, vis_entropy, vis_active = self._build_class_stats(self.vis_scores, self.rgb_ids)
        ir_proto, ir_count, ir_entropy, ir_active = self._build_class_stats(self.ir_scores, self.ir_ids)

        vis_memory = F.normalize(self.vis_memory.to(self.device), dim=1)
        ir_memory = F.normalize(self.ir_memory.to(self.device), dim=1)
        feat_score = torch.matmul(vis_memory, ir_memory.t())
        feat_score = ((feat_score + 1.0) * 0.5).clamp(0.0, 1.0)

        pred_score = 0.5 * (vis_proto + ir_proto.t())
        conf_score = 1.0 - (vis_entropy.unsqueeze(1) + ir_entropy.unsqueeze(0)) / (2.0 * np.log(self.num_classes + 1e-12))
        conf_score = conf_score.clamp(0.0, 1.0)

        active_mask = vis_active.unsqueeze(1) & ir_active.unsqueeze(0)
        cost = (
            self.pred_weight * (1.0 - pred_score) +
            self.feat_weight * (1.0 - feat_score) +
            self.conf_weight * (1.0 - conf_score)
        )
        high_cost = cost[active_mask].max() + 1.0 if active_mask.any() else torch.tensor(1.0, device=self.device)
        cost = cost.masked_fill(~active_mask, high_cost)

        a = vis_count / vis_count.sum().clamp_min(1.0)
        b = ir_count / ir_count.sum().clamp_min(1.0)
        transport = self._sinkhorn(cost, a, b, active_mask)
        row_transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-12)
        col_transport = transport.t() / transport.t().sum(dim=1, keepdim=True).clamp_min(1e-12)

        row_conf, row_assign = row_transport.max(dim=1)
        col_conf, col_assign = col_transport.max(dim=1)

        class_ids = torch.arange(self.num_classes, device=self.device)
        mutual_rows = col_assign[row_assign] == class_ids
        mutual_cols = row_assign[col_assign] == class_ids

        common_row_mask = vis_active & (row_conf >= self.high_th) & mutual_rows
        common_col_mask = ir_active & (col_conf >= self.high_th) & mutual_cols
        specific_row_mask = vis_active & (row_conf >= self.high_th) & (~common_row_mask)
        specific_col_mask = ir_active & (col_conf >= self.high_th) & (~common_col_mask)
        remain_row_mask = vis_active & (row_conf >= self.mid_th) & (row_conf < self.high_th)
        remain_col_mask = ir_active & (col_conf >= self.mid_th) & (col_conf < self.high_th)

        common_rm = torch.zeros_like(row_transport)
        if common_row_mask.any():
            common_rm[common_row_mask, row_assign[common_row_mask]] = row_conf[common_row_mask]

        specific_rm = row_transport * specific_row_mask.float().unsqueeze(1)
        remain_rm = self._topk_mask(row_transport * remain_row_mask.float().unsqueeze(1))
        remain_col_rm = self._topk_mask(col_transport * remain_col_mask.float().unsqueeze(1))

        self.transport_row.copy_(row_transport)
        self.transport_col.copy_(col_transport)
        self.common_rm.copy_(common_rm)
        self.specific_rm.copy_(specific_rm)
        self.remain_rm.copy_(remain_rm)
        self.remain_col_rm.copy_(remain_col_rm)
        self.row_conf.copy_(row_conf)
        self.col_conf.copy_(col_conf)
        self.row_assign.copy_(row_assign)
        self.col_assign.copy_(col_assign)
        self.common_row_mask.copy_(common_row_mask)
        self.common_col_mask.copy_(common_col_mask)
        self.specific_row_mask.copy_(specific_row_mask)
        self.specific_col_mask.copy_(specific_col_mask)
        self.remain_row_mask.copy_(remain_row_mask)
        self.remain_col_mask.copy_(remain_col_mask)

        v2i = OrderedDict()
        for idx in torch.nonzero(vis_active, as_tuple=False).flatten():
            v2i[int(idx.item())] = int(row_assign[idx].item())
        i2v = OrderedDict()
        for idx in torch.nonzero(ir_active, as_tuple=False).flatten():
            i2v[int(idx.item())] = int(col_assign[idx].item())
        return v2i, i2v
    # TODO
    def _get_label(self,dists,mode):
        sample_rate = 1
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]# flat to 1d and sort
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)# sort index return to 2d, like ([0,1,2],[1,2,0])
        idx1, idx2 = sorted_2d[0], sorted_2d[1]# sorted idx of dim0 and dim1
        dists = dists[idx1, idx2]
        idx_length = int(np.ceil(sample_rate*dists.shape[0]/self.num_classes))
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
        lambda_cm=0.1
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
                 ir_idx, 'scores', rgb_features, ir_features)
        
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
