from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import UniformMesh3d # 从fealpy中导入UniformMesh1d

# 在区间[0,2]x[0,2]x[0,2]上创建二维均匀网格
domain=[0,2,0,2,0,2]
nx=3
ny=3
nz=3
hx=(domain[1]-domain[0])/nx
hy=(domain[3]-domain[2])/ny
hz=(domain[5]-domain[4])/nz

mesh=UniformMesh3d([0,nx,0,ny,0,nz],h=[hx,hy,hz],origin=[domain[0],domain[2],domain[4]])

# 用entity方法获取node edge face cell网格实体张量
node=mesh.entity('node')
edge=mesh.entity('edge')
face=mesh.entity('face')
cell=mesh.entity('cell')

# 测试
print("node.shape:\n",node.shape)
print("edge.shape:\n",edge.shape)
print("face.shape:\n",face.shape)
print("cell.shape:\n",cell.shape)
print("node.coordinates:\n",node)
print("edge.tensor:\n",edge)
print("face.tensor:\n",face)
print("cell.tensor:\n",cell)

# 绘制图象
fig = plt.figure() 
axes = fig.add_subplot(111, projection='3d') 
axes.set_title('3D Uniform Mesh') 
mesh.add_plot(axes, linewidth=3)
mesh.find_node(axes, showindex=True, fontsize=10)
mesh.find_edge(axes, showindex=True, fontsize=10)
mesh.find_cell(axes, showindex=True, fontsize=10)
plt.show()
