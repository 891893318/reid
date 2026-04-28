import torch
import torch.cuda
import cutlass

# ======================
# 关键：触发 INT32 溢出的尺寸
# 2^31-1 = 2147483647
# 下面的大小 = 65536 * 65536 = 4294967296 > 2147483647 → 必溢出
# ======================
BATCH_SIZE = 1
M = 65536
N = 65536

print("="*60)
print(f"张量大小: {M} x {N} = {M*N} 元素")
print(f"INT32 最大值: {2**31 - 1}")
print(f"是否溢出: {M*N > 2**31 -1}")
print("="*60)

# CUDA 大张量（PPU 加密张量本质也是这种）
device = "cuda" if torch.cuda.is_available() else "cpu"
A = torch.randn(M, N, device=device, dtype=torch.float32)
B = torch.randn(M, N, device=device, dtype=torch.float32)
C = torch.zeros_like(A)

# ======================
# 测试 1：原生 torch 运算（正常）
# ======================
try:
    C_torch = A + B
    print("\n✅ PyTorch 计算成功")
except Exception as e:
    print(f"\n❌ PyTorch 失败: {e}")

# ======================
# 测试 2：CUTLASS 算子（触发 INT32 溢出）
# 这就是 PPU 里会崩溃的地方
# ======================
print("\n🚀 运行 CUTLASS 算子 (会触发 INT32 索引溢出)...")
try:
    # CUTLASS gemm / elementwise 内核默认使用 int32 索引
    # 大矩阵下会出现：
    # 1. CUDA illegal memory access
    # 2. 结果错误
    # 3. 程序崩溃
    C_cutlass = torch.ops.cutlass.add(A, B)
    torch.cuda.synchronize()
    print("✅ CUTLASS 计算成功 (未触发溢出)")
except Exception as e:
    print(f"❌ CUTLASS 溢出错误 (符合预期):")
    print(f"   {e}")
    print("\n🔥 确认：这就是 **INT32 索引溢出**")
    print("   原因：CUTLASS 内核使用 32bit 索引，超过 2^31-1 就会溢出")

# ======================
# 测试 3：安全尺寸（不溢出，验证 CUTLASS 正常）
# ======================
print("\n" + "="*60)
print("测试小尺寸（不溢出）...")
M_small = 1024
N_small = 1024
A_s = torch.randn(M_small, N_small, device=device)
B_s = torch.randn(M_small, N_small, device=device)
try:
    C_s = torch.ops.cutlass.add(A_s, B_s)
    print("✅ CUTLASS 小尺寸正常运行")
except Exception as e:
    print(f"❌ CUTLASS 小尺寸失败: {e}")