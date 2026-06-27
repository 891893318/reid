import torch
import os
import re

from .classifier import Image_Classifier

from utils import os_walk
from .agw import AGW
from .clip_model import CLIP
from .optim import WarmupMultiStepLR
from .loss import TripletLoss_WRT
_models = {
    "resnet": AGW,  # visual encoder AGW, no text encoder
    "clip-resnet": CLIP,  # resnet50 + transformer
    "vit": 0,  # visual encoder vit, no text encoder
    #"clip-vit": CLIP,  # both vit-b/16 + transformer
}


def create(args):
    """
    Create a dataset instance.
    """
    if args.arch not in _models:
        raise KeyError("Unknown backbone:", args.arch)
    print('loading {} dataset ...'.format(args.dataset))
    return Model(args)


class Model:
    def __init__(self, args):
        self.mode = args.mode
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model_root = os.path.join(args.save_path, "models/")
        self.stage1_save_path = os.path.join(self.model_root, "stage1/")
        self.save_path = os.path.join(self.model_root, "stage2/")
        os.makedirs(self.stage1_save_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.milestones = args.milestones
        self.resume = args.resume
        self.args = args

        self.model = _models[args.arch](args).to(self.device)
        self.classifier1 = Image_Classifier(args).to(self.device) # RGB classifier
        self.classifier2 = Image_Classifier(args).to(self.device) # IR classifier

        self._init_optimizer()
        self._init_criterion()

    def _init_optimizer(self):
        ###########################
        ###专家学习率适配###
        params_phase1 = []
        params_phase2 = []
        for part in (self.model, self.classifier1, self.classifier2):
            for key, value in part.named_parameters():
                if value.requires_grad and key.find("classifier") != -1:
                    params_phase1 += [{"params": [value], 
                                "lr": 2 * self.lr,
                                "weight_decay": self.weight_decay}]
                elif value.requires_grad:
                    params_phase1 += [{"params": [value], 
                                "lr": self.lr,
                                "weight_decay": self.weight_decay}]
        params_phase2.extend(params_phase1)
        self.optimizer_phase1 = torch.optim.Adam(params_phase1)
        self.optimizer_phase2 = torch.optim.Adam(params_phase2)
        self.scheduler_phase1 = WarmupMultiStepLR(self.optimizer_phase1, self.milestones,
                                           gamma=0.1, warmup_factor=0.01, warmup_iters=10, mode='cls')
        self.scheduler_phase2 = WarmupMultiStepLR(self.optimizer_phase2, self.milestones,
                                           gamma=0.1, warmup_factor=0.01, warmup_iters=10, mode='cls')
    def _init_criterion(self):
        self.pid_criterion = torch.nn.CrossEntropyLoss()
        self.tri_criterion = TripletLoss_WRT()

    def set_train(self):
        self.model.train()
        self.classifier1.train()
        self.classifier2.train()

    def set_eval(self):
        self.model.eval()
        self.classifier1.eval()
        self.classifier2.eval()

    def save_model(self, save_epoch, is_best_rank=False, is_best_map=False):
        if not is_best_rank and not is_best_map:
            return

        root, _, files = os_walk(self.save_path)
        for file in files:
            if re.fullmatch(r'model_\d+\.pth', file):
                os.remove(os.path.join(root, file))

        all_state_dict = {'backbone': self.model.state_dict(),
                          'classifier1': self.classifier1.state_dict(),
                          'classifier2': self.classifier2.state_dict()}
        torch.save(all_state_dict, os.path.join(self.save_path, f'model_{save_epoch}.pth'))
        if is_best_rank:
            torch.save(all_state_dict, os.path.join(self.save_path, 'best_r1.pth'))
        if is_best_map:
            torch.save(all_state_dict, os.path.join(self.save_path, 'best_map.pth'))
        
    def resume_model(self, specified_model=None):
        '''
        # load the weights from existed file
        model_epoch: the epoch of the model to load(optional)
        '''
        if specified_model is None:
            root, _, files = os_walk(self.save_path)
            self.resume_epoch = 0
            if len(files) > 0:
                indexes = []
                for file in files:
                    match = re.fullmatch(r'model_(\d+)\.pth', file)
                    if match:
                        indexes.append(int(match.group(1)))
                if indexes:
                    indexes = sorted(list(set(indexes)), reverse=False)
                    model_path = os.path.join(self.save_path, 'model_{}.pth'.format(indexes[-1]))
                    
                    if self.resume or self.mode == 'test':
                        loaded_dict = torch.load(model_path,map_location=self.device)
                        self.model.load_state_dict(loaded_dict['backbone'], strict=False)
                        self.classifier1.load_state_dict(loaded_dict['classifier1'], strict=False)
                        self.classifier2.load_state_dict(loaded_dict['classifier2'], strict=False)
                        start_epoch = indexes[-1]
                        self.resume_epoch = start_epoch
                        print(f'load {model_path}')
                    else:
                        os.remove(model_path)
                        print('existed model files removed!')
                else:
                    print('valid model file not existed!')
            else:
                print('valid model file not existed!')
            print(f'from {self.resume_epoch} epoch training')

        else:
            model_path = specified_model
            loaded_dict = torch.load(model_path,map_location=self.device)
            self.model.load_state_dict(loaded_dict['backbone'], strict=False)
            self.classifier1.load_state_dict(loaded_dict['classifier1'], strict=False)
            self.classifier2.load_state_dict(loaded_dict['classifier2'], strict=False)
            
            # match = re.search(r'\d+', specified_model)
            # self.resume_epoch = int(match.group())
            
            self.resume_epoch = 0
            print(f'load {model_path}')
            print(f'from {self.resume_epoch} epoch training')
