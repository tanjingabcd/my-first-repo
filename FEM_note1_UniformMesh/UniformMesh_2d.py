from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d # 从fealpy中导入UniformMesh1d

# 在区间[0,1]x[0,1]上创建二维均匀网格
domain=[0,1,0,1]
nx=3
ny=3
hx=(domain[1]-domain[0])/nx
hy=(domain[3]-domain[2])/ny

mesh=UniformMesh2d([0,nx,0,ny],h=[hx,hy],origin=[domain[0],domain[2]])

# 用entity方法获取node edge cell网格实体张量
node=mesh.entity('node')
edge=mesh.entity('edge')
cell=mesh.entity('cell')

# 测试
print("node.shape:\n",node.shape)
print("edge.shape:\n",edge.shape)
print("cell.shape:\n",cell.shape)
print("node.coordinates:\n",node)
print("edge.tensor:\n",edge)
print("cell.tensor:\n",cell)

# 绘制图象
fig=plt.figure()
axes = fig.gca()
axes.set_title('2D Uniform Mesh') 
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=10)
mesh.find_edge(axes, showindex=True, fontsize=10)
mesh.find_cell(axes, showindex=True, fontsize=10)
plt.show()
