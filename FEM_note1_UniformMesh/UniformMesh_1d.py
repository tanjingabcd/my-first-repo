from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d # 从fealpy中导入UniformMesh1d

# 在区间[0,1]上创建一维均匀网格
domain=[0,1]
nx=10
hx=(domain[1]-domain[0])/nx

mesh=UniformMesh1d([0,nx],h=hx,origin=domain[0])

# 用entity方法获取node edge 网格实体张量
node=mesh.entity('node')
edge=mesh.entity('edge')

# 测试
print("node.shape:\n",node.shape)
print("edge.shape:\n",edge.shape)
print("node.coordinates:\n",node)
print("edge.tensor:\n",edge)

# 绘制图象
fig=plt.figure()
axes = fig.gca()
axes.set_title('1D Uniform Mesh') 
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=20)
mesh.find_edge(axes, showindex=True, fontsize=20)
plt.show()
