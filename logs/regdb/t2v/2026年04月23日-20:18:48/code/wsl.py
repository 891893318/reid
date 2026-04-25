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
        self.num_cams = getattr(args, 'num_cams', 1)
        self.T = args.temperature # softmax temperature
        self.sigma = args.sigma # momentum update factor
        # memory of visible and infrared modal
        self.register_buffer('vis_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('ir_memory',torch.zeros(self.num_classes,2048))
        self.register_buffer('vis_cam_memory', torch.zeros(self.num_classes, self.num_cams, 2048))
        self.register_buffer('ir_cam_memory', torch.zeros(self.num_classes, self.num_cams, 2048))
        self.register_buffer('vis_cam_valid', torch.zeros(self.num_classes, self.num_cams, dtype=torch.bool))
        self.register_buffer('ir_cam_valid', torch.zeros(self.num_classes, self.num_cams, dtype=torch.bool))

    @torch.no_grad()
    # def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode):
    def save(self,vis,ir,rgb_ids,ir_ids,rgb_idx,ir_idx,mode, rgb_features=None, ir_features=None,
             rgb_cams=None, ir_cams=None):
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
            self.vis_cam_memory = self.vis_cam_memory.to(self.device)
            self.ir_cam_memory = self.ir_cam_memory.to(self.device)
            self.vis_cam_valid = self.vis_cam_valid.to(self.device)
            self.ir_cam_valid = self.ir_cam_valid.to(self.device)
            
            # Get unique labels and process RGB and IR features
            self._set_memory(self.vis_memory, self.vis_cam_memory, self.vis_cam_valid,
                             rgb_features, rgb_ids, rgb_cams)
            self._set_memory(self.ir_memory, self.ir_cam_memory, self.ir_cam_valid,
                             ir_features, ir_ids, ir_cams)
        ################################
        vis = vis.detach().cpu().numpy()
        ir = ir.detach().cpu().numpy()
        rgb_ids, ir_ids = rgb_ids.cpu(), ir_ids.cpu()
            
        self.vis, self.ir = vis, ir
        self.rgb_ids, self.ir_ids = rgb_ids, ir_ids
        self.rgb_idx, self.ir_idx = rgb_idx, ir_idx
        
    @torch.no_grad()
    def update(self, rgb_feats, ir_feats, rgb_labels, ir_labels, rgb_cams=None, ir_cams=None):
        self._momentum_update(self.vis_memory, self.vis_cam_memory, self.vis_cam_valid,
                              rgb_feats, rgb_labels, rgb_cams)
        self._momentum_update(self.ir_memory, self.ir_cam_memory, self.ir_cam_valid,
                              ir_feats, ir_labels, ir_cams)

    @torch.no_grad()
    def _set_memory(self, global_memory, cam_memory, cam_valid, feats, labels, cams):
        label_set = torch.unique(labels)
        for label in label_set:
            label = int(label.item())
            label_mask = labels == label
            global_memory[label] = feats[label_mask].mean(dim=0)
            if cams is None:
                continue
            for cam in torch.unique(cams[label_mask]):
                cam = int(cam.item())
                cam_mask = label_mask & (cams == cam)
                cam_memory[label, cam] = feats[cam_mask].mean(dim=0)
                cam_valid[label, cam] = True

    @torch.no_grad()
    def _momentum_update(self, global_memory, cam_memory, cam_valid, feats, labels, cams):
        label_set = torch.unique(labels)
        for label in label_set:
            label = int(label.item())
            label_mask = labels == label
            selected = feats[label_mask].mean(dim=0)
            global_memory[label] = (1 - self.sigma) * global_memory[label] + self.sigma * selected
            if cams is None:
                continue
            for cam in torch.unique(cams[label_mask]):
                cam = int(cam.item())
                cam_mask = label_mask & (cams == cam)
                selected_cam = feats[cam_mask].mean(dim=0)
                if cam_valid[label, cam]:
                    cam_memory[label, cam] = (1 - self.sigma) * cam_memory[label, cam] + self.sigma * selected_cam
                else:
                    cam_memory[label, cam] = selected_cam
                    cam_valid[label, cam] = True

    def get_global_memory(self, modal):
        return self.vis_memory if modal == 'rgb' else self.ir_memory

    def get_cam_memory(self, modal):
        if modal == 'rgb':
            return self.vis_cam_memory, self.vis_cam_valid
        return self.ir_cam_memory, self.ir_cam_valid

    def gather_same_modal_prototypes(self, modal, labels, cams):
        global_memory = self.get_global_memory(modal)
        cam_memory, cam_valid = self.get_cam_memory(modal)
        labels = labels.long()
        cams = cams.long().clamp(min=0, max=self.num_cams - 1)
        global_proto = global_memory[labels]
        cam_proto = cam_memory[labels, cams]
        valid = cam_valid[labels, cams].unsqueeze(1)
        cam_proto = torch.where(valid, cam_proto, global_proto)
        return global_proto, cam_proto

    def get_cross_modal_relation(self, source_modal, target_modal, source_feats, source_labels, source_cams,
                                 lambda_instance=0.5, lambda_cam=0.25, lambda_global=0.25, temperature=0.07):
        source_global_memory = self.get_global_memory(source_modal)
        target_global_memory = self.get_global_memory(target_modal)
        target_cam_memory, target_cam_valid = self.get_cam_memory(target_modal)

        source_labels = source_labels.long()
        source_cams = source_cams.long().clamp(min=0, max=self.num_cams - 1)

        source_global = source_global_memory[source_labels]
        target_same_cam = target_cam_memory[:, source_cams].permute(1, 0, 2)
        target_same_cam_valid = target_cam_valid[:, source_cams].permute(1, 0).unsqueeze(-1)
        target_global_expand = target_global_memory.unsqueeze(0).expand_as(target_same_cam)
        target_fused = torch.where(target_same_cam_valid, target_same_cam, target_global_expand)

        norm_source_feats = torch.nn.functional.normalize(source_feats, dim=1)
        norm_source_global = torch.nn.functional.normalize(source_global, dim=1)
        norm_target_global = torch.nn.functional.normalize(target_global_memory, dim=1)
        norm_target_fused = torch.nn.functional.normalize(target_fused, dim=2)

        score_instance = torch.matmul(norm_source_feats, norm_target_global.t())
        score_global = torch.matmul(norm_source_global, norm_target_global.t())
        score_cam = torch.einsum('bd,bkd->bk', norm_source_feats, norm_target_fused)
        score = lambda_instance * score_instance + lambda_cam * score_cam + lambda_global * score_global
        prob = torch.softmax(score / temperature, dim=1)
        return score, prob, target_fused

    def get_label(self, epoch=None):
        if self.not_saved:# pass if 
            pass
        else:
            print('get match labels')
            if self.mode == 'features':
                dists = np.matmul(self.vis, self.ir.T)
                v2i_dict, i2v_dict = self._get_label(dists,'dist')

            elif self.mode == 'scores':
                v2i_dict, _ = self._get_label(self.vis,'rgb')
                i2v_dict, _ = self._get_label(self.ir,'ir')
                self.v2i = v2i_dict
                self.i2v = i2v_dict
            return v2i_dict, i2v_dict
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
            
            rgb_features, rgb_labels, rgb_cams, rgb_gt, r2i_cls, rgb_idx = self._extract_feature(model, rgb_loader,'rgb')
            ir_features, ir_labels, ir_cams, ir_gt, i2r_cls, ir_idx = self._extract_feature(model, ir_loader,'ir')

        # # //match by cls and save features to memory bank
        self.save(r2i_cls, i2r_cls, rgb_labels, ir_labels, rgb_idx,\
                 ir_idx, 'scores', rgb_features, ir_features, rgb_cams, ir_cams)
        
    def _extract_feature(self, model, loader, modal):

        print('extracting {} features'.format(modal))

        saved_features, saved_labels, saved_cls= None, None, None
        saved_gts, saved_idx, saved_cams = None, None, None
        for imgs_list, infos in loader:
            labels = infos[:,1]
            cams = infos[:,2]
            idx = infos[:,0]
            gts = infos[:,-1].to(model.device)
            if imgs_list.__class__.__name__ != 'list':
                imgs = imgs_list
                imgs, labels, idx, cams = \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device), cams.to(model.device)
            else:
                ori_imgs, ca_imgs = imgs_list[0], imgs_list[1]
                if len(ori_imgs.shape) < 4:
                    ori_imgs = ori_imgs.unsqueeze(0)
                    ca_imgs = ca_imgs.unsqueeze(0)

                imgs = torch.cat((ori_imgs,ca_imgs),dim=0)
                labels = torch.cat((labels,labels),dim=0)
                cams = torch.cat((cams, cams), dim=0)
                idx = torch.cat((idx,idx),dim=0)
                gts= torch.cat((gts,gts),dim=0).to(model.device)
                imgs, labels, idx, cams= \
                    imgs.to(model.device), labels.to(model.device), idx.to(model.device), cams.to(model.device)
            _, bn_features = model.model(imgs) # _:gap feature

            if modal == 'rgb':
                cls, l2_features = model.classifier2(bn_features)
            elif modal == 'ir':
                cls, l2_features = model.classifier1(bn_features)
            l2_features = l2_features.detach().cpu()

            if saved_features is None: 
                # saved_features, saved_labels, saved_cls, saved_idx = l2_features, labels, cls, idx
                saved_features, saved_labels, saved_cls, saved_idx, saved_cams = bn_features, labels, cls, idx, cams

                saved_gts = gts
            else:
                # saved_features = torch.cat((saved_features, l2_features), dim=0)
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)
                saved_cams = torch.cat((saved_cams, cams), dim=0)

                saved_gts = torch.cat((saved_gts, gts), dim=0)
        return saved_features, saved_labels, saved_cams, saved_gts, saved_cls, saved_idx
