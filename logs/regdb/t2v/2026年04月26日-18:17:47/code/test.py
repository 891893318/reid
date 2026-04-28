import torch
import cutlass
from cutlass import cute, Float32, Int32, Int64

# 触发条件：M*N > 2^31-1 → 用 M=65536, N=65536 → 4,294,967,296 > 2^31-1
M = 65536
N = 65536

# 构造大张量
A = torch.randn(M, N, device='cuda', dtype=torch.float32)
B = torch.randn(M, N, device='cuda', dtype=torch.float32)
C = torch.zeros_like(A)

# 转为CUTLASS Tensor
a = cutlass.from_dlpack(A)
b = cutlass.from_dlpack(B)
c = cutlass.from_dlpack(C)

# 定义CuTe内核（默认int32索引）
@cute.jit
def add(a: cute.Tensor[Float32], b: cute.Tensor[Float32], c: cute.Tensor[Float32]):
    M, N = a.shape
    for i in range(M):  # i 是 Int32
        for j in range(N):  # j 是 Int32
            c[i, j] = a[i, j] + b[i, j]

# 执行（会溢出报错）
try:
    add(a, b, c)
    print("Success (unlikely for large M/N)")
except Exception as e:
    print(f"溢出错误: {e}")
    # 典型输出：CUDA error: an illegal memory access was encountered