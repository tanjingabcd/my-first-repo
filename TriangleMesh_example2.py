import numpy as np
from fealpy.mesh import TriangleMesh #从fealpy导入TriangleMesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #导入3d绘图包

# a.生成单位球面上的三角形网格
mesh = TriangleMesh.from_unit_sphere_surface(refine=3)

# b.用 entity 方法获取 node, edge, face, cell 网格实体张量
node = mesh.entity('node') # (NN, 3) 获取 node 网格实体张量
edge = mesh.entity('edge') # (NE, 2) 获取 edge 网格实体张量
face = mesh.entity('face') # (NF, 2) 获取 face 网格实体张量
cell = mesh.entity('cell') # (NC, 3) 获取 cell 网格实体张量

print("node.shpe:\n", node.shape)
print("edge.shape:\n", edge.shape)
print("face.shape:\n", face.shape)
print("cell.shape:\n", cell.shape)
print("node coordinates:\n", node)
print("edge tensor:\n", edge)
print("face tensor:\n", face)
print("cell tensor:\n", cell)

# 画图
fig = plt.figure() 
axes = fig.add_subplot(111, projection='3d') 
axes.set_title('3D Triangle Mesh') # 为图像命名
mesh.add_plot(axes,linewidth=0.5)

# c.用 matplotlib 画出网格以及各种网格实体的编号
mesh.find_node(axes, showindex=True, fontsize=6)
mesh.find_edge(axes, showindex=True, fontsize=6)
mesh.find_cell(axes, showindex=True, fontsize=6)
plt.show()