import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.mesh import UniformMesh1d
from fealpy.sparse import csr_matrix
from fealpy.sparse import spdiags
from fealpy.solver import spsolve

# 修改 PDE 数据类以适配新方程
class PDEData:
    def __init__(self):
        pass

    def domain(self):
        return [-1, 1]  # 修改计算域为 [-1, 1]

    def solution(self, x):
        # 真解 u(x) = e^{-x²}(1 - x²)
        return (bm.exp(-x**2) * (1 - x**2)).reshape(-1)  

    def source(self, x):
        # 根据真解推导的源项 f(x) = e^{-x²}(4x⁴ - 16x² + 6)
        return (bm.exp(-x**2) * (4 * x**4 - 16 * x**2 + 6)).reshape(-1)

class PoissonFDMModel:
    def __init__(self, pde):
        self.pde = pde

    def run(self, nx=10, maxit=4):
        e = bm.zeros(maxit, dtype=bm.float64)
        for i in range(maxit):
            self.generate_mesh(nx=nx)
            A, f = self.linear_system()
            self.solve(A, f)
            e[i] = self.error()
            if i < maxit - 1:
                nx *= 2  # 每次迭代加密网格
        print('L2误差:', e)
        print('误差比 (验证 O(h²)):', e[:-1] / e[1:])


    def generate_mesh(self, nx=10):
        domain = self.pde.domain()
        hx = (domain[1] - domain[0]) / nx  # 计算步长
        # 生成一维均匀网格，原点为 domain[0] = -1
        self.mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

    def linear_system(self):
        mesh = self.mesh
        pde = self.pde
        hx = mesh.h
        cx = 1 / hx**2  # 二阶导数系数
        NN = mesh.number_of_nodes()
        K = bm.arange(NN)

        # 构建三对角矩阵 A
        val = bm.zeros(NN, dtype=bm.float64)
        val[:] = 2*cx+2
        I = K
        J = K
        # 对角线元素
        A = csr_matrix((val, (I, I)), shape=(NN, NN))

        # 非对角线元素
        val = -bm.ones(NN-1, dtype=bm.float64)*cx
        I = K[1:] # NN-1
        J = K[:-1] # NN-1
        A += csr_matrix((val, (I, J)), shape=(NN, NN)) # 上对角元素
        A += csr_matrix((val, (J, I)), shape=(NN, NN)) # 下对角元素

        # 处理右端项 f
        node = mesh.entity('node')
        f = pde.source(node) 

        # Dirichlet 边界条件 (固定 u(-1)=0, u(1)=0)
        index = bm.array([0, NN-1], dtype=bm.int32)  # 边界索引
        self.uh = bm.zeros(NN, dtype=bm.float64)
        self.uh[index] = pde.solution(node[index])  # 应用边界值

        # 调整右端项以消除边界条件影响
        f = f - A @ self.uh
        f[index] = self.uh[index]  # 固定边界值

        # 构造修正矩阵确保边界行对角线为 1
        bdIdx = bm.zeros(NN, dtype=bm.int32)
        bdIdx[index] = 1
        D0 = spdiags(1 - bdIdx, 0, NN, NN, format='csr')
        D1 = spdiags(bdIdx, 0, NN, NN, format='csr')
        A = D0 @ A @ D0 + D1  # 修正矩阵

        return A, f

    def solve(self, A, f):
        self.uh[:] = spsolve(A, f, solver='scipy')

    def error(self):
        """计算 L2 范数误差"""
        pde = self.pde
        mesh = self.mesh
        uh = self.uh
        node = mesh.entity('node')
        u = pde.solution(node)
        e = uh - u
        hx = mesh.h  # 步长
        l2_error = bm.sqrt(bm.sum(e**2) * hx)  # L2 误差公式
        return l2_error

    def show(self, u=None):
        fig = plt.figure()
        axes = fig.add_subplot(121)
        self.mesh.add_plot(axes)
        axes = fig.add_subplot(122)
        if u is None:
            node = self.mesh.entity('node')
            u = self.pde.solution(node)
        self.mesh.show_function(axes, u, box=[-1, 1, -0.5, 1.1])  # 调整绘图范围
        plt.show()

# 运行求解器
pde = PDEData()
model = PoissonFDMModel(pde)
model.run(nx=10, maxit=5)  # 计算 10, 20, 40, 80, 160 网格
model.show(u=model.uh)