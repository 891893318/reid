import os
import shutil
import argparse
import setproctitle
import torch
import warnings

import time
import datasets
import models
from task import train, test
from wsl import CMA
from utils import time_now, makedir, Logger, MultiItemAverageMeter, set_seed, save_checkpoint

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
            model.save_model(current_epoch, is_best_rank)
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

    parser.add_argument('--weak-weight',default=0.25, type=float, help='weight of weak loss')
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
    parser.add_argument('--num-cams', default=1, type=int, dest='num_cams', help='number of train-time cameras or pseudo-cameras')
    parser.add_argument('--scrc-relation-temp', default=0.07, type=float, dest='scrc_relation_temp', help='temperature for soft cross-modal relation')
    parser.add_argument('--scrc-lambda-instance', default=0.5, type=float, dest='scrc_lambda_instance', help='instance similarity weight in SCRC relation')
    parser.add_argument('--scrc-lambda-cam', default=0.25, type=float, dest='scrc_lambda_cam', help='camera-aware prototype weight in SCRC relation')
    parser.add_argument('--scrc-lambda-global', default=0.25, type=float, dest='scrc_lambda_global', help='global prototype weight in SCRC relation')
    parser.add_argument('--scrc-cm-weight', default=0.2, type=float, dest='scrc_cm_weight', help='weight of SCRC cross-modal alignment loss')
    parser.add_argument('--scrc-bi-weight', default=0.05, type=float, dest='scrc_bi_weight', help='weight of SCRC bidirectional consistency loss')
    parser.add_argument('--scrc-proto-weight', default=0.05, type=float, dest='scrc_proto_weight', help='weight of SCRC prototype regularization')
    parser.add_argument('--enable-rgmfd', default=0, type=int, help='enable reliability-guided modality-invariant feature decoupling')
    parser.add_argument('--rgmfd-reduction', default=16, type=int, help='channel reduction ratio in RG-MFD gate')
    parser.add_argument('--rgmfd-gate-scale', default=0.5, type=float, help='residual scale for RG-MFD shared feature gate')
    parser.add_argument('--rgmfd-start-epoch', default=0, type=int, help='phase epoch to start RG-MFD auxiliary losses')
    parser.add_argument('--rgmfd-rel-weight', default=0.2, type=float, help='weight of reliable common-relation alignment for RG-MFD')
    parser.add_argument('--rgmfd-orth-weight', default=0.05, type=float, help='weight of shared-specific orthogonality for RG-MFD')
    parser.add_argument('--rgmfd-gate-reg-weight', default=0.01, type=float, help='weight of RG-MFD gate balance regularization')
    parser.add_argument('--rgmfd-gate-target', default=0.5, type=float, help='target average shared gate value for RG-MFD')
    parser.add_argument('--disable-structured-indoor', default=1, type=int, help='disable structured local branch for SYSU indoor search')
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
    parser.add_argument('--enable-eot-rr', default=0, type=int, help='replace hard CRE with evidential optimal-transport relation reasoning')
    parser.add_argument('--eot-temp', default=0.07, type=float, help='temperature for EOT-RR Sinkhorn transport')
    parser.add_argument('--eot-iters', default=20, type=int, help='Sinkhorn iterations for EOT-RR')
    parser.add_argument('--eot-topk', default=5, type=int, help='top-k relation candidates kept per identity before Sinkhorn')
    parser.add_argument('--eot-momentum', default=0.8, type=float, help='EMA momentum for EOT-RR temporal relation evidence')
    parser.add_argument('--eot-expert-weight', default=0.45, type=float, help='bidirectional expert evidence weight in EOT-RR')
    parser.add_argument('--eot-proto-weight', default=0.25, type=float, help='prototype similarity evidence weight in EOT-RR')
    parser.add_argument('--eot-temporal-weight', default=0.20, type=float, help='temporal stability evidence weight in EOT-RR')
    parser.add_argument('--eot-certainty-weight', default=0.10, type=float, help='prediction certainty evidence weight in EOT-RR')
    parser.add_argument('--eot-common-threshold', default=0.55, type=float, help='minimum reliability for mutual EOT-RR common relations')
    parser.add_argument('--eot-specific-threshold', default=0.25, type=float, help='minimum reliability for one-way EOT-RR specific relations')
    parser.add_argument('--eot-remain-threshold', default=0.08, type=float, help='minimum reliability for conflicted EOT-RR remain relations')
    parser.add_argument('--eot-common-margin', default=0.03, type=float, help='minimum row/column reliability margin for EOT-RR common relations')
    parser.add_argument('--enable-trrm', default=0, type=int, help='enable temporal reliability relation memory')
    parser.add_argument('--trrm-momentum', default=0.8, type=float, help='momentum for temporal relation reliability memory')
    parser.add_argument('--trrm-specific-start', default=20, type=int, help='phase2 epoch to start using stable specific relations')
    parser.add_argument('--trrm-specific-streak', default=2, type=int, help='minimum consecutive epochs for a specific relation')
    parser.add_argument('--trrm-specific-threshold', default=0.2, type=float, help='minimum memory score for a specific relation')
    parser.add_argument('--trrm-specific-max-ratio', default=1.0, type=float, help='maximum ratio of reliable specific relations kept each epoch, 1.0 disables the cap')
    parser.add_argument('--trrm-specific-max-num', default=0, type=int, help='maximum number of reliable specific relations kept each epoch, 0 disables the cap')
    parser.add_argument('--trrm-specific-ramp', default=0, type=int, help='epochs used to ramp up the specific relation loss weight, 0 disables ramp')
    parser.add_argument('--trrm-specific-weight', default=0.3, type=float, help='loss weight for reliable specific relations')
    parser.add_argument('--trrm-remain-start', default=60, type=int, help='phase2 epoch to start weak supervision for remain relations')
    parser.add_argument('--trrm-remain-weight', default=0.2, type=float, help='extra loss weight for remain weak supervision')
    parser.add_argument('--enable-remain-gate', default=0, type=int, help='enable confidence filtering for remain weak supervision')
    parser.add_argument('--remain-gate-keep-ratio', default=0.7, type=float, help='top ratio of remain samples kept by confidence gate')
    parser.add_argument('--remain-gate-max-num', default=0, type=int, help='maximum number of remain samples kept per batch, 0 disables the cap')
    parser.add_argument('--remain-gate-min-prob', default=0.0, type=float, help='minimum classifier probability on remain candidate labels')
    parser.add_argument('--remain-gate-min-margin', default=0.0, type=float, help='minimum top1-top2 margin for remain weak samples')
    parser.add_argument('--remain-gate-min-certainty', default=0.0, type=float, help='minimum entropy-based certainty for remain weak samples')
    parser.add_argument('--remain-gate-margin-weight', default=0.5, type=float, help='margin weight in remain gate score')
    parser.add_argument('--remain-gate-entropy-weight', default=0.3, type=float, help='entropy certainty weight in remain gate score')
    parser.add_argument('--enable-rdl', default=0, type=int, help='enable relation-guided dynamic loss weighting')
    parser.add_argument('--rdl-warmup', default=5, type=int, help='epochs used to warm up dynamic relation weights')
    parser.add_argument('--rdl-coverage-weight', default=0.5, type=float, help='coverage contribution in relation reliability')
    parser.add_argument('--rdl-stability-weight', default=0.5, type=float, help='stability contribution in relation reliability')
    parser.add_argument('--rdl-common-boost', default=0.05, type=float, help='maximum boost for reliable common relation losses')
    parser.add_argument('--rdl-specific-min', default=0.7, type=float, help='minimum weight for specific relation loss')
    parser.add_argument('--rdl-remain-min', default=0.5, type=float, help='minimum weight for remain weak loss')
    parser.add_argument('--rdl-remain-ratio-weight', default=0.25, type=float, help='penalty for high remain relation ratio')
    parser.add_argument("--device", default=0, type=int, help="gpu")
    parser.add_argument('--stage1-epoch' ,default=20, type=int,help='s:20, l:80, r:50')
    parser.add_argument('--stage2-epoch' ,default=120, type=int,help='CMCL total epoch')
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
    # RegDB has 10 official trials, so keep each split in its own folder.
    if args.dataset == 'regdb':
        args.save_path = f'logs/{args.dataset}/{mode}/trial{args.trial}/{current_time}'
    else:
        args.save_path = f'logs/{args.dataset}/{mode}/{current_time}'
    if args.dataset =='sysu':
        args.num_classes = 395
    elif args.dataset =='regdb':
        args.num_classes = 206
    elif args.dataset == 'llcm':
        args.num_classes = 713
    set_seed(args.seed)
    setproctitle.setproctitle(args.save_path)
    main(args)
