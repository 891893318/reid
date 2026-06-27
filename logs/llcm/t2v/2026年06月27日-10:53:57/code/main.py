import os
import shutil
import argparse
import setproctitle
import warnings

import time
import datasets
import models
from task import train, test
from wsl import CMA
from utils import time_now, makedir, Logger, set_seed, save_checkpoint

warnings.filterwarnings("ignore")

def main(args):
    best_rank1 = 0
    best_mAP = 0
    log_path = os.path.join(args.save_path, "log/")
    model_path = os.path.join(args.save_path, "models/")
    stage1_model_path = os.path.join(model_path, "stage1/")
    stage2_model_path = os.path.join(model_path, "stage2/")
    code_path = os.path.join(args.save_path, "code/")
    makedir(log_path)
    makedir(model_path)
    makedir(stage1_model_path)
    makedir(stage2_model_path)
    makedir(code_path)
    
    # 备份代码
    ignore_dirs = ['logs', 'data', 'docs', 'paper', '.git', '__pycache__', 'save']
    for item in os.listdir('.'):
        if item in ignore_dirs:
            continue
        s_path = item
        d_path = os.path.join(code_path, item)
        if os.path.isdir(s_path):
            shutil.copytree(s_path, d_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            if s_path.endswith('.py') or s_path.endswith('.sh') or s_path.endswith('.md') or s_path.endswith('.txt'):
                shutil.copy2(s_path, d_path)

    logger = Logger(os.path.join(log_path, "log.txt"))
    if not args.resume and args.mode == 'train':
        logger.clear()
    logger(args)
    dataset = datasets.create(args)
    model = models.create(args)

    if args.mode == "train":
        cma = CMA(args)
        if args.resume or not args.model_path == 'default':
            enable_phase1 = False
            if 'wsl' in args.debug and not args.model_path == 'default':
                model.resume_model(args.model_path)
            else:
                model.resume_model()
        elif 'wsl' in args.debug:
            enable_phase1 = True
            model.resume_model()
        else:
            enable_phase1 = False
            model.resume_model()

        if enable_phase1:
            logger('Time: {} | start phase1 from epoch 0'.format(time_now()))
            for current_epoch in range(0, args.stage1_epoch):
                model.scheduler_phase1.step(current_epoch)
                _, result = train(args, model, dataset, current_epoch,cma,logger,enable_phase1)
                cmc, mAP, mINP = test(args, model, dataset,current_epoch) 
                best_rank1 = max(cmc[0], best_rank1)
                best_mAP = max(mAP, best_mAP)
                logger('Time: {} | phase 1 epoch {}; Setting: {}'.format(time_now(), current_epoch+1, args.save_path))
                logger(f'e_lr: {model.scheduler_phase1.get_lr()[0]}')
                logger(result)
                logger('R1:{:.4f};   R10:{:.4f};  R20:{:.4f};  mAP:{:.4f};  mINP:{:.4f}\n\
                       Best_R1: {:.4f};    Best mAP: {:.4f}'\
                       .format(cmc[0], cmc[9], cmc[19],mAP, mINP,best_rank1,best_mAP))
                logger('=================================================')
                if current_epoch == args.stage1_epoch-1:
                    save_checkpoint(args,model,current_epoch+1)
        enable_phase1 = False
        start_epoch = model.resume_epoch
        logger('Time: {} | start phase2 from epoch {}'.format(time_now(), start_epoch))
        for current_epoch in range(start_epoch, args.stage2_epoch):
            model.scheduler_phase2.step(current_epoch)
            _, result = train(args, model, dataset, current_epoch,cma,logger,enable_phase1)

            cmc, mAP, mINP = test(args, model, dataset,current_epoch) 
            is_best_rank = (cmc[0] >= best_rank1)
            is_best_mAP = (mAP >= best_mAP)
            best_rank1 = max(cmc[0], best_rank1)
            best_mAP = max(mAP, best_mAP)
            model.save_model(current_epoch, is_best_rank, is_best_mAP)
            logger('=================================================\nEpoch: {};Time: {};Setting: {}'\
                   .format(current_epoch, time_now(), args.save_path))
            logger(f'e_lr: {model.scheduler_phase2.get_lr()[0]}')
            logger(result)
            logger('R1:{:.4f};   R10:{:.4f};  R20:{:.4f};  mAP:{:.4f};  mINP:{:.4f}\n\
                   Best_R1: {:.4f};    Best mAP: {:.4f}'\
                   .format(cmc[0], cmc[9], cmc[19],mAP, mINP,best_rank1,best_mAP))
            logger('=================================================')
        
    if args.mode == 'test':
        if args.model_path == 'default':
            model.resume_model()
        else:
            model.resume_model(args.model_path)
        cmc, mAP, mINP = test(args, model, dataset)
        logger('Time: {}; Test on Dataset: {}'.format(time_now(), args.dataset))
        logger('R1:{:.4f};   R10:{:.4f};  R20:{:.4f};  mAP:{:.4f};  mINP:{:.4f}\n\
               Best_R1: {:.4f};    Best mAP: {:.4f}'\
               .format(cmc[0], cmc[9], cmc[19],mAP, mINP,best_rank1,best_mAP))
        
def get_parser():
    parser = argparse.ArgumentParser("WSL-ReID")
    parser.add_argument("--dataset", default="regdb", type=str, help="dataset name: sysu, llcm, regdb")
    parser.add_argument("--arch", default="resnet", type=str, help="sysu:clip-resnet, llcm:resnet, regdb:resnet")
    parser.add_argument('--mode', default='train', help='train or test')

    parser.add_argument("--data-path", default="/root/data/", type=str, help="dataset path")
    parser.add_argument("--save-path", default="save/", type=str, help="log and model save path")

    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate 0.0003 for sysu(s) and llcm(l), 0.00045 for regdb(r)')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight deacy')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 70],
                        help='milestones for the learning rate decay, s&l:30, 70, r:50, 70')
    parser.add_argument('--relabel', default=1, type=int, help='relabel train dataset')

    parser.add_argument('--tri-weight',default=0.25, type=float, help='weight of triplet loss')
    parser.add_argument('--img-h',default=288,type=int,help='height of the input image')
    parser.add_argument('--img-w',default=144,type=int,help='width of the input image')
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument('--num-workers', default=32, type=int, help='num workers')
    parser.add_argument('--batch-pidnum', default=8, type=int ,help='training pid in each batch, r:5')
    parser.add_argument('--pid-numsample',default=4, type=int ,help='num sample of each pid in a batch')
    parser.add_argument('--test-batch', default=128, type=int, metavar='tb', help='testing batch size')
    
    parser.add_argument('--sigma', default=0.8, type=float, help='momentum update factor')
    parser.add_argument('-T', '--temperature', default=3, type=float, help='Temperature parameter of softmax')
    # Shared SYSU/LLCM innovation defaults. Dataset-specific overrides stay in bash scripts.
    parser.add_argument('--enable-rgmfd', default=1, type=int, help='enable reliability-guided modality-invariant feature decoupling')
    parser.add_argument('--rgmfd-reduction', default=16, type=int, help='channel reduction ratio in RG-MFD gate')
    parser.add_argument('--rgmfd-gate-scale', default=0.5, type=float, help='residual scale for RG-MFD shared feature gate')
    parser.add_argument('--rgmfd-start-epoch', default=0, type=int, help='phase epoch to start RG-MFD auxiliary losses')
    parser.add_argument('--rgmfd-orth-weight', default=0.05, type=float, help='weight of shared-specific orthogonality for RG-MFD')
    parser.add_argument('--rgmfd-gate-reg-weight', default=0.01, type=float, help='weight of RG-MFD gate balance regularization')
    parser.add_argument('--rgmfd-gate-target', default=0.5, type=float, help='target average shared gate value for RG-MFD')
    parser.add_argument('--phase2-id-weight', default=1.0, type=float, help='weight of non-detached phase2 modality ID loss')
    parser.add_argument('--enable-hcl', default=0, type=int, help='replace CCL with relation-guided hard-negative contrastive learning')
    parser.add_argument('--hcl-weight', default=0.1, type=float, help='weight of hard-negative contrastive loss')
    parser.add_argument('--hcl-temperature', default=0.07, type=float, help='temperature of hard-negative InfoNCE')
    parser.add_argument('--hcl-topk', default=20, type=int, help='number of hardest cross-modal identity negatives')
    parser.add_argument('--hcl-min-coverage', default=0.2, type=float, help='minimum common relation coverage before HCL can start')
    parser.add_argument('--hcl-full-coverage', default=0.7, type=float, help='stable common coverage that gives HCL full relation strength')
    parser.add_argument('--hcl-min-stability', default=0.7, type=float, help='minimum common relation stability before HCL can start')
    parser.add_argument('--hcl-ready-epochs', default=3, type=int, help='consecutive reliable epochs required to activate HCL')
    parser.add_argument('--hcl-pair-streak', default=2, type=int, help='consecutive epochs required before a common pair supervises HCL')
    parser.add_argument('--hcl-warmup-epochs', default=5, type=int, help='epochs used to ramp HCL loss after activation')
    parser.add_argument('--enable-uprt', default=1, type=int, help='replace CCL with uncertainty-guided partial relation transport')
    parser.add_argument('--uprt-weight', default=0.1, type=float, help='weight of posterior-guided cross-modal loss')
    parser.add_argument('--uprt-cls-weight', default=0.05, type=float, help='weight of posterior-guided cross-modal classifier calibration')
    parser.add_argument('--uprt-cls-temperature', default=1.0, type=float, help='temperature of posterior classifier calibration logits')
    parser.add_argument('--uprt-proto-cls-weight', default=0.02, type=float, help='weight of posterior-guided prototype classifier calibration')
    parser.add_argument('--uprt-proto-cls-temperature', default=1.0, type=float, help='temperature of posterior prototype classifier calibration')
    parser.add_argument('--uprt-common-tri-weight', default=1.0, type=float, help='weight multiplier for common-pair cross-modal triplet sharpening')
    parser.add_argument('--uprt-common-tri-start-epoch', default=35, type=int, help='phase2 epoch to start common-pair triplet sharpening')
    parser.add_argument('--uprt-common-tri-warmup-epochs', default=10, type=int, help='epochs used to ramp common-pair triplet sharpening')
    parser.add_argument('--uprt-cmo-weight', default=0.03, type=float, help='weight of common-pair memory prototype output distillation')
    parser.add_argument('--uprt-cmo-start-epoch', default=50, type=int, help='phase2 epoch to start common-pair prototype distillation')
    parser.add_argument('--uprt-cmo-warmup-epochs', default=10, type=int, help='epochs used to ramp common-pair prototype distillation')
    parser.add_argument('--uprt-cmo-temperature', default=2.0, type=float, help='softmax temperature for common-pair prototype distillation')
    parser.add_argument('--uprt-cmo-min-target-prob', default=0.05, type=float, help='minimum prototype target probability for common-pair distillation')
    parser.add_argument('--uprt-specific-ce-weight', default=0.015, type=float, help='weight of CRE specific-pair soft classifier expansion')
    parser.add_argument('--uprt-remain-ce-weight', default=0.0, type=float, help='weight of CRE remain-pair soft classifier expansion')
    parser.add_argument('--uprt-specific-ce-start-epoch', default=35, type=int, help='phase2 epoch to start specific-pair expansion')
    parser.add_argument('--uprt-remain-ce-start-epoch', default=70, type=int, help='phase2 epoch to start remain-pair expansion')
    parser.add_argument('--uprt-relation-ce-warmup-epochs', default=10, type=int, help='epochs used to ramp CRE relation expansion')
    parser.add_argument('--uprt-specific-ce-strength', default=0.35, type=float, help='target blend strength for specific-pair expansion')
    parser.add_argument('--uprt-remain-ce-strength', default=0.15, type=float, help='target blend strength for remain-pair expansion')
    parser.add_argument('--uprt-specific-ce-min-target-prob', default=0.7, type=float, help='minimum posterior probability for specific-pair expansion')
    parser.add_argument('--uprt-remain-ce-min-target-prob', default=0.45, type=float, help='minimum posterior probability for remain-pair expansion')
    parser.add_argument('--uprt-relation-ce-temperature', default=1.0, type=float, help='temperature of CRE relation expansion logits')
    parser.add_argument('--uprt-hard-weight', default=0.03, type=float, help='weight of posterior top-1 hard negative contrastive loss')
    parser.add_argument('--uprt-hard-start-epoch', default=70, type=int, help='phase2 epoch to start posterior top-1 hard supervision')
    parser.add_argument('--uprt-hard-warmup-epochs', default=10, type=int, help='epochs used to ramp posterior top-1 hard supervision')
    parser.add_argument('--uprt-hard-topk', default=20, type=int, help='number of hard negatives for posterior top-1 supervision')
    parser.add_argument('--uprt-hard-temperature', default=0.07, type=float, help='temperature for posterior top-1 hard supervision')
    parser.add_argument('--uprt-hard-min-confidence', default=0.9, type=float, help='minimum posterior top-1 confidence for hard supervision')
    parser.add_argument('--uprt-temperature', default=0.07, type=float, help='temperature of posterior-guided contrastive logits')
    parser.add_argument('--uprt-shared-temperature', default=0.07, type=float, help='temperature for shared-prototype relation evidence')
    parser.add_argument('--uprt-topk', default=10, type=int, help='number of transport candidates per identity')
    parser.add_argument('--uprt-epsilon', default=0.1, type=float, help='entropy regularization for unbalanced transport')
    parser.add_argument('--uprt-tau', default=0.5, type=float, help='marginal relaxation strength for unbalanced transport')
    parser.add_argument('--uprt-iters', default=30, type=int, help='number of unbalanced Sinkhorn iterations')
    parser.add_argument('--uprt-min-mass', default=0.05, type=float, help='minimum transferable identity mass')
    parser.add_argument('--uprt-start-epoch', default=5, type=int, help='phase2 epoch to enable UPRT supervision')
    parser.add_argument('--uprt-warmup-epochs', default=10, type=int, help='epochs used to ramp UPRT supervision')
    parser.add_argument('--uprt-shared-weight', default=1.0, type=float, help='shared-prototype affinity weight')
    parser.add_argument('--uprt-expert-weight', default=0.1, type=float, help='bidirectional expert affinity weight')
    parser.add_argument('--uprt-prior-weight', default=0.05, type=float, help='CRE common prior affinity weight')
    parser.add_argument('--uprt-specific-prior', default=0.5, type=float, help='soft prior strength for one-sided CRE specific pairs')
    parser.add_argument('--uprt-remain-prior', default=0.2, type=float, help='soft prior strength for ambiguous CRE remain pairs')
    parser.add_argument('--uprt-recovery-weight', default=0.08, type=float, help='maximum late-stage posterior recovery mixing weight')
    parser.add_argument('--uprt-recovery-start-epoch', default=40, type=int, help='phase2 epoch to start posterior coverage recovery')
    parser.add_argument('--uprt-recovery-warmup-epochs', default=20, type=int, help='epochs used to ramp posterior coverage recovery')
    parser.add_argument('--uprt-recovery-target-coverage', default=0.95, type=float, help='target common relation coverage for recovery')
    parser.add_argument('--uprt-recovery-min-coverage', default=0.8, type=float, help='coverage below which recovery reaches maximum strength')
    parser.add_argument('--uprt-recovery-min-entropy', default=0.08, type=float, help='minimum normalized posterior entropy before recovery increases')
    parser.add_argument('--uprt-recovery-temperature', default=0.2, type=float, help='temperature for recovery candidate posterior')
    parser.add_argument('--uprt-recovery-topk', default=20, type=int, help='candidate count used by late-stage posterior recovery')
    parser.add_argument('--cre-confidence', default=0, type=int, help='use confidence-aware cross-modal relation establishment')
    parser.add_argument('--cre-sample-rate', default=1.0, type=float, help='candidate sample rate for cross-modal relation mining')
    parser.add_argument('--cre-count-weight', default=0, type=float, help='count-priority weight in confidence-aware CRE')
    parser.add_argument('--cre-prob-weight', default=0, type=float, help='classification probability weight in confidence-aware CRE')
    parser.add_argument('--cre-margin-weight', default=0, type=float, help='top1-top2 margin weight in confidence-aware CRE')
    parser.add_argument('--cre-entropy-weight', default=0, type=float, help='prediction certainty weight in confidence-aware CRE')
    parser.add_argument('--cre-proto-weight', default=0, type=float, help='cross-modal prototype similarity weight in confidence-aware CRE')
    parser.add_argument('--cre-min-margin-start', default=0.0, type=float, help='initial margin filter for confidence-aware CRE')
    parser.add_argument('--cre-min-margin-end', default=0.0, type=float, help='final margin filter for confidence-aware CRE')
    parser.add_argument('--cre-margin-decay-epoch', default=0, type=int, help='epochs used to relax the CRE margin filter')
    parser.add_argument("--device", default=0, type=int, help="gpu")
    parser.add_argument('--stage1-epoch' ,default=20, type=int,help='s:20, l:80, r:50')
    parser.add_argument('--stage2-epoch' ,default=120, type=int,help='stage2 total epoch')
    parser.add_argument('--resume', default= 0, type = int, help='resume or not')
    parser.add_argument('--debug', default='wsl',type=str,help='wsl or sl')
    parser.add_argument('--trial', default=1,type=int,help='trial for regdb')
    parser.add_argument('--search-mode', default='all',type=str,help='all or indoor search gallery')
    parser.add_argument('--gall-mode', default='single',type=str,help='mutil or single shot')
    parser.add_argument('--test-mode', '--test_mode', default='t2v',type=str, dest='test_mode', help='regdb and llcm test_mode')
    parser.add_argument('--model-path', '--model_path', default='default', type=str, dest='model_path', help='load from checkpoint')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    mode=""
    if args.dataset == 'sysu':
        mode = args.search_mode
    elif args.dataset == 'regdb':
        mode = args.test_mode
    elif args.dataset == 'llcm':
        mode = args.test_mode
        # 时间格式：年:月:日-时:分:秒
    current_time = time.strftime("%Y年%m月%d日-%H:%M:%S")

    # 最终路径：logs/dataset/mode/时间
    args.save_path = f'logs/{args.dataset}/{mode}/{current_time}'
    if args.dataset =='sysu':
        args.num_classes = 395
    elif args.dataset =='regdb':
        args.num_classes = 206
        # args.save_path += f'_{args.trial}'
    elif args.dataset == 'llcm':
        args.num_classes = 713
    set_seed(args.seed)
    setproctitle.setproctitle(args.save_path)
    main(args)
