from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.mesh import UniformMesh1d
from fealpy.sparse import csr_matrix

# 创建一维均匀网格
domain=[0,1]
nx=3
hx=(domain[1]-domain[0])/nx
mesh=UniformMesh1d([0,nx],h=hx,origin=domain[0])
N=nx-1

# 初始化 CSR 格式的 data, indices, indptr
data = []
indices = []
indptr = [0]

for i in range(N):
    # 主对角线元素: 2/h²
    data.append(2 / hx**2)
    indices.append(i)
    
    # 左次对角线（非首行）
    if i > 0:
        data.append(-1 / hx**2)
        indices.append(i-1)
    
    # 右次对角线（非末行）
    if i < N-1:
        data.append(-1 / hx**2)
        indices.append(i+1)
    
    indptr.append(len(data))

# 转换为 NumPy 数组
data = bm.array(data, dtype=bm.float64)
indices = bm.array(indices, dtype=bm.int32)
indptr = bm.array(indptr, dtype=bm.int32)

# 构建 CSR 稀疏矩阵
laplace_matrix = csr_matrix((data, indices, indptr), shape=(N, N))

# 验证输出（以 nx=3 为例）
print("一维 Laplace 离散矩阵 (CSR 格式):")
print(laplace_matrix.toarray())