import os
import torch
import warnings
import datasets
import models
import argparse
from hypothesis_verification import run_real_verification

warnings.filterwarnings("ignore")

def test_hypothesis():
    """
    专用测试脚本：加载已训练模型并读取数据集，调用假设验证流。
    """
    print(">>> 正在初始化 SCPS 假设验证测试流 <<<")

    # 1. 模拟命令行参数，构建能够跑起原工程模型和数据集的 args 环境
    from main import get_parser
    parser = get_parser()
    
    # 额外补充针对 SCPS 验证需要的覆盖 / 参数
    parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search for RegDB/LLCM')
    
    args = parser.parse_args()
    
    print(f"1. 初始化数据集 [{args.dataset}] ...")
    
    if args.dataset =='sysu':
        args.num_classes = 395
    elif args.dataset =='regdb':
        args.num_classes = 206
    elif args.dataset == 'llcm':
        args.num_classes = 713
        
    dataset = datasets.create(args)
    
    print(f"2. 初始化模型 [{args.arch}], 分类数={args.num_classes} ...")
    model = models.create(args)
    
    if args.model_path != 'default':
        print(f"   -> 正在装载预训练权重: {args.model_path}")
        model.resume_model(args.model_path)
    else:
        print("   -> [警告] 未提供训练好的权重路径(--model_path), 正在使用未训练的随机权重进行连通性测试。结果将不具有假设验证意义！")
        
    print("\n3. 启动真实 Logit 提取与 SCPS 分析流 ...")
    
    # 设定分解频率阈值，例如前20%作为低频
    k_low = int(args.num_classes * 0.2) 
    if k_low == 0:
        k_low = 10
        
    run_real_verification(model=model, dataset=dataset, args=args, K_LOW_THRESHOLD=k_low)

if __name__ == '__main__':
    test_hypothesis()
