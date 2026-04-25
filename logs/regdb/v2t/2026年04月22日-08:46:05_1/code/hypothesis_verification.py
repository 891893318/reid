import torch
import torch.nn.functional as F
import numpy as np
import math
from scipy.stats import spearmanr

# ==============================================================================
# 专家预测空间的谱分解协作学习 (SCPS) - 假设验证代码骨架
# ==============================================================================
# 该脚本用于离线验证预测分布的频率特性，无需修改目前的训练主干网络。
# 前置准备：需要从预训练模型或者训练中期的模型中，提取特征/Logits输入这里。

def apply_svd_projection(logits_A, logits_B):
    """
    对预测分布 (Logits或Softmax概率) 执行 PCA / SVD 低质-高秩分解
    获取主成分/隐空间的表征
    """
    # 将两模态的数据拼在一起，提取共享主成分
    combined = torch.cat([logits_A, logits_B], dim=0)
    mu = combined.mean(dim=0, keepdim=True)
    centered = combined - mu
    
    # SVD 分解
    U, S, V = torch.svd(centered)
    
    # 将原本的 logit 投影到数据驱动的主特征基底上
    proj_A = (logits_A - mu) @ V
    proj_B = (logits_B - mu) @ V
    
    return proj_A, proj_B

# ------------------------------------------------------------------------------
# 验证 1: 专家预测向量的语义频率分解
# 假设: 同一身份的跨模态样本，低频分量高度一致，高频分量呈不一致/差异性
# ------------------------------------------------------------------------------
def verify_frequency_decomposition(logits_v2v, logits_v2r, K_low):
    """
    :param logits_v2v: VIS专家对VIS图像的预测分布, shape (B, C)
    :param logits_v2r: VIS专家对IR图像的预测分布, shape (B, C)
    :param K_low: 低频带的截断阈值
    """
    print("\n--- 验证 1: SVD 低秩特征协作分解假设 ---")
    dct_v2v, dct_v2r = apply_svd_projection(logits_v2v, logits_v2r)
    
    # 划分高低频
    low_v2v, high_v2v = dct_v2v[:, :K_low], dct_v2v[:, K_low:]
    low_v2r, high_v2r = dct_v2r[:, :K_low], dct_v2r[:, K_low:]
    
    # 计算余弦相似度
    sim_low = F.cosine_similarity(low_v2v, low_v2r, dim=-1).mean().item()
    sim_high = F.cosine_similarity(high_v2v, high_v2r, dim=-1).mean().item()
    
    print(f"低频带平均相似度 (应该趋近于 1): {sim_low:.4f}")
    print(f"高频带平均相似度 (应该显著低于低频): {sim_high:.4f}")
    return sim_low, sim_high

# ------------------------------------------------------------------------------
# 验证 2: 谱感知的跨模态关系矩阵 (SCM)
# 假设: 软截断的关系矩阵分布更加平滑，能捕捉微妙的跨模态依赖
# ------------------------------------------------------------------------------
def verify_scm_smoothness(logits_v2r, logits_r2v, K_low, lambda_penalty=0.1):
    """
    模拟 SCM 矩阵的计算
    """
    print("\n--- 验证 2: 跨模态关系矩阵 (SCM) 平滑性 ---")
    dct_v2r, dct_r2v = apply_svd_projection(logits_v2r, logits_r2v)
    
    B = dct_v2r.shape[0]
    S_matrix = torch.zeros((B, B))
    
    for i in range(B):
        for j in range(B):
            # 1. 低频结构一致性 (使用余弦相似度)
            low_sim = F.cosine_similarity(dct_v2r[i, :K_low].unsqueeze(0), 
                                          dct_r2v[j, :K_low].unsqueeze(0)).item()
            
            # 2. 高频分歧惩罚 (L1距离)
            high_diff = torch.abs(dct_v2r[i, K_low:] - dct_r2v[j, K_low:]).mean().item()
            
            # SCM score
            S_matrix[i, j] = low_sim - lambda_penalty * high_diff

    print(f"SCM 关系矩阵 (前 3x3 个元素示例):\n{S_matrix[:3, :3].numpy()}")
    print(f"矩阵均值: {S_matrix.mean().item():.4f}, 矩阵方差: {S_matrix.std().item():.4f} (预期不应该全部为 0 或 1，应具有过渡态)")
    return S_matrix

# ------------------------------------------------------------------------------
# 验证 3: 异构保真损失 (HPSL) 稳定性测试
# 假设: 强制低频一致、高频负相关。验证目标是其计算梯度的合理性
# ------------------------------------------------------------------------------
def verify_hpsl_loss(logits_v2v, logits_r2v, K_low):
    """
    计算 HPSL 损失值
    """
    print("\n--- 验证 3: 低秩保真-高秩隔离损失 (MoLoRA) ---")
    dct_v2v, dct_r2v = apply_svd_projection(logits_v2v, logits_r2v)
    
    # 允许跟踪梯度以测试稳定性
    
    # 低频 MSE 损失 (结构一致性)
    loss_low = F.mse_loss(dct_v2v[:, :K_low], dct_r2v[:, :K_low])
    
    # 高频负相关奖励 (互相补充性)
    high_v2v = dct_v2v[:, K_low:]
    high_r2v = dct_r2v[:, K_low:]
    
    # 手动计算相关性惩罚
    mean_v2v = high_v2v.mean(dim=-1, keepdim=True)
    mean_r2v = high_r2v.mean(dim=-1, keepdim=True)
    std_v2v = high_v2v.std(dim=-1, keepdim=True) + 1e-6
    std_r2v = high_r2v.std(dim=-1, keepdim=True) + 1e-6
    
    # 高频皮尔逊相关系数
    correlation = ((high_v2v - mean_v2v) * (high_r2v - mean_r2v)).mean(dim=-1) / (std_v2v.squeeze(-1) * std_r2v.squeeze(-1))
    
    # 期望高频不要高度正相关 (即包含独立的特定模态判别信息)
    loss_high = (1.0 + correlation).mean()  # 如果正强相关(1)，则loss大；如果倾向于负相关(-1)，则loss接近0
    
    total_hpsl = loss_low + 0.5 * loss_high
    print(f"Low-Freq MSE Loss: {loss_low.item():.4f}")
    print(f"High-Freq Correlation Penalty: {loss_high.item():.4f}")
    print(f"Total HPSL Loss: {total_hpsl.item():.4f}")
    
    return total_hpsl

# ------------------------------------------------------------------------------
# 验证 4: 跨模态谱排序蒸馏 (CSRD) 排序一致性
# 假设: 预测频率分布在低频带具有极强的秩相关性，高频带呈现弱相关或不相关
# ------------------------------------------------------------------------------
def verify_csrd_ranking(logits_v2v, logits_v2r, K_low):
    """
    计算并打印高低频频带的 Spearman 等级相关系数
    """
    print("\n--- 验证 4: CSRD 排序特性 ---")
    dct_v2v, dct_v2r = apply_svd_projection(logits_v2v, logits_v2r)
    
    low_corr_list = []
    high_corr_list = []
    
    B = dct_v2v.shape[0]
    for i in range(B):
        # 计算每个样本低频分量预测排序的 Spearman 系数
        corr_low, _ = spearmanr(dct_v2v[i, :K_low].detach().numpy(), 
                                dct_v2r[i, :K_low].detach().numpy())
        # 计算高频部分
        corr_high, _ = spearmanr(dct_v2v[i, K_low:].detach().numpy(), 
                                 dct_v2r[i, K_low:].detach().numpy())
        
        low_corr_list.append(corr_low)
        high_corr_list.append(corr_high)
        
    mean_low_corr = np.nanmean(low_corr_list)
    mean_high_corr = np.nanmean(high_corr_list)
    
    print(f"低频带期望排序高度一致: Spearman Corr = {mean_low_corr:.4f}")
    print(f"高频带期望排序差异较大: Spearman Corr = {mean_high_corr:.4f}")

    return mean_low_corr, mean_high_corr

def run_real_verification(model, dataset, args, K_LOW_THRESHOLD=50):
    """
    实际集成在你的 WSL_ReID 工程中运行的入口函数。
    此函数会从测试集（如 query 和 gallery）中提取出可见光和红外的实际模型预测 Logits。
    
    使用方法：可以在 main.py 或 task/test.py 中调用此函数，传入已经加载的 model 和 dataset。
    """
    print("\n===============================================================")
    print(" 开始在真实数据集和模型上验证 SCPS 联合假定 (提取 Logits)...")
    print("===============================================================")
    
    model.set_eval()
    
    # 临时收集 logits 和 labels 的字典
    vis_logits_dict = {} # key: identity_label, value: list of logits
    ir_logits_dict = {}  # key: identity_label, value: list of logits
    
    with torch.no_grad():
        # 以下以当前工程中 dataset 常见的结构进行特征提取适配
        # 假设 dataset 提供了 query_loader 和 gallary_loader
        # 根据数据集不同，query 和 gallery 的模态是不同的 (如 SYSU: query 是 IR, gallery 是 VIS)
        
        # 1. 提取 Query Logits
        for batch_idx, (inputs, labels) in enumerate(dataset.query_loader):
            inputs = inputs.to(model.device)
            # 根据 agw.py 结构获取原始特征
            # 注意: 如果是 SYSU，query 通常是 IR (x2)
            if args.dataset == 'sysu' or args.test_mode == "t2v":
                _, feat = model.model(x2=inputs)
            else:
                _, feat = model.model(x1=inputs)
            
            # 从分类器获取 logits (x_score就是logits分布)
            logits, _ = model.classifier1(feat)
            
            # 将 logits 按照 identity label 存储
            for i, label in enumerate(labels):
                lbl = label.item()
                if args.dataset == 'sysu' or args.test_mode == "t2v":
                    if lbl not in ir_logits_dict:
                        ir_logits_dict[lbl] = []
                    ir_logits_dict[lbl].append(logits[i].cpu())
                else:
                    if lbl not in vis_logits_dict:
                        vis_logits_dict[lbl] = []
                    vis_logits_dict[lbl].append(logits[i].cpu())
                    
        # 2. 提取 Gallery Logits
        for batch_idx, (inputs, labels) in enumerate(dataset.gallery_loader):
            inputs = inputs.to(model.device)
            # SYSU 的 gallery 是 VIS (x1)
            if args.dataset == 'sysu' or args.test_mode == "t2v":
                _, feat = model.model(x1=inputs)
            else:
                _, feat = model.model(x2=inputs)
                
            logits, _ = model.classifier1(feat)
            
            for i, label in enumerate(labels):
                lbl = label.item()
                if args.dataset == 'sysu' or args.test_mode == "t2v":
                    if lbl not in vis_logits_dict:
                        vis_logits_dict[lbl] = []
                    vis_logits_dict[lbl].append(logits[i].cpu())
                else:
                    if lbl not in ir_logits_dict:
                        ir_logits_dict[lbl] = []
                    ir_logits_dict[lbl].append(logits[i].cpu())

    # 3. 构建配对样本的 Tensor
    # 我们只挑选同时存在于 VIS 和 IR 中的 Identity，构造 (v2v, v2r)
    # 对于每个相同 Identity 的 VIS和IR 样本对计算
    paired_vis = []
    paired_ir = []
    
    common_labels = set(vis_logits_dict.keys()).intersection(set(ir_logits_dict.keys()))
    print(f"找到 {len(common_labels)} 个跨模态共有的 ID 类别进行测试。")
    
    for lbl in list(common_labels)[:100]: # 取前100个ID防止内存爆炸
        # 随机取该ID下的一个 VIS Logit 和 一个 IR Logit 作为样本对
        v_logit = vis_logits_dict[lbl][0]
        i_logit = ir_logits_dict[lbl][0]
        paired_vis.append(v_logit)
        paired_ir.append(i_logit)
        
    if len(paired_vis) == 0:
        print("未找到配对样本，无法执行验证！")
        return

    real_logits_v = torch.stack(paired_vis) # shape: (Matched_IDs, Num_Classes)
    real_logits_i = torch.stack(paired_ir)
    real_logits_v_shuffle = real_logits_v[torch.randperm(real_logits_v.size(0))] # 造一个打乱身份的作为错误pair对比
    
    print("\n---> 开始使用真实的配对预测数据进行验证流测试 <---")
    # 代入刚才写好的验证函数
    print("[测试 True Pairs: 同身份不同模态 (v2v vs v2r)]")
    verify_frequency_decomposition(real_logits_v, real_logits_i, K_LOW_THRESHOLD)
    verify_scm_smoothness(real_logits_v, real_logits_i, K_LOW_THRESHOLD)
    verify_hpsl_loss(real_logits_v, real_logits_i, K_LOW_THRESHOLD)
    verify_csrd_ranking(real_logits_v, real_logits_i, K_LOW_THRESHOLD)
    
    print("\n[对比测试 False Pairs: 作为基线对照，若功能正常，假Pair的结构一致性应该很弱]")
    verify_frequency_decomposition(real_logits_v, real_logits_v_shuffle, K_LOW_THRESHOLD)

# ==============================================================================
# 模拟执行测试
# ==============================================================================
if __name__ == '__main__':
    print("生成模拟随机预测数据进行脚本验证流测试...\n")
    BATCH_SIZE = 16
    NUM_CLASSES = 395 # SYSU 数据集分类数示例
    K_LOW_THRESHOLD = 50 # 设定前50个频段为低频（粗粒度语义信息）
    
    # 模拟网络预测的 Logits (均值为0，方差为1)
    # 现实中应使用 Baseline 在验证集前向传播得到的真实 logits
    dummy_logits_v2v = torch.randn(BATCH_SIZE, NUM_CLASSES)
    
    # 制造模拟数据：v2r 的低频部分和 v2v 相似，高频部分随机噪声(不相似)
    dummy_logits_v2r = dummy_logits_v2v.clone()
    # 扰乱高频信息
    dummy_logits_v2r += torch.randn(BATCH_SIZE, NUM_CLASSES) * 0.5 
    
    dummy_logits_r2v = torch.randn(BATCH_SIZE, NUM_CLASSES)
    
    # 执行验证脚本
    verify_frequency_decomposition(dummy_logits_v2v, dummy_logits_v2r, K_LOW_THRESHOLD)
    verify_scm_smoothness(dummy_logits_v2r, dummy_logits_r2v, K_LOW_THRESHOLD)
    verify_hpsl_loss(dummy_logits_v2v, dummy_logits_r2v, K_LOW_THRESHOLD)
    verify_csrd_ranking(dummy_logits_v2v, dummy_logits_v2r, K_LOW_THRESHOLD)
    
    print("\n>>> 请将 Baseline 真实预测传入这些函数提取实际统计结果 <<<")
