"""
  使用FEM求解二维poisson方程
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# 网格剖分

# 节点 node
node=np.array([
    [-1,-1],
    [ 0,-1],
    [ 1,-1],
    [-1, 0],
    [ 0, 0],
    [ 1, 0],
    [-1, 1],
    [ 0, 1],
    [ 1, 1]
])

# 单元 element
elem=np.array([
    [4,1,5],
    [2,5,1],
    [5,2,6],
    [3,6,2],
    [7,4,8],
    [5,8,4],
    [8,5,9],
    [6,9,5]
])-1 # python的索引从0开始

# x = node[:,0]
# y = node[:,1]

# triangle=tri.Triangulation(x,y,elem)
# plt.triplot(triangle)
# plt.scatter(x,y)
# plt.show()

# 网格加密

def refine_mesh(node,elem):
    # 输入：网格加密前存储的节点和单元
    # 输出：网格加密后生成的新的节点和单元
    new_node = node.tolist() # 初始节点列表（保留原始节点）
    new_elem = [] # 初始化为空列表
    edge_midpoint = {} # 空字典，记录边与中点索引

    # 获取边的中点

    def midpoint_index(i,j):
        edge = tuple(sorted((i,j))) # 无向边
        if edge not in edge_midpoint:
            mid = (node[edge[0]]+node[edge[1]])/2
            edge_midpoint[edge]=len(new_node)
            new_node.append(mid.tolist())
        return edge_midpoint[edge]
    
    # 构造 new_node

    # 构造 new_elem，遍历每一个三角形，细化为四个小三角形

    for t in elem:
        i, j, k=t
        m1 = midpoint_index(i,j)
        m2 = midpoint_index(j,k)
        m3 = midpoint_index(k,i)
        new_elem.extend([
            [i, m1, m3],
            [m1, j, m2],
            [m3, m2, k],
            [m1, m2, m3]
        ])
    return np.array(new_node),np.array(new_elem)

# 加密网格

Nref=1
refined_node,refined_elem=node, elem
for i in range(Nref):
    refined_node,refined_elem=refine_mesh(refined_node,refined_elem)
  

# 绘制细化前的网格

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.triplot(node[:,0],node[:,1],elem,color='blue',lw=1)
plt.scatter(node[:,0],node[:,1],color='red',zorder=5,label='Nodes')
plt.gca().set_aspect('equal')
plt.title("Original Mesh")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# 绘制细化后的网格
plt.subplot(1,2,2)
plt.triplot(refined_node[:,0],refined_node[:,1],refined_elem,color='blue',lw=1)
plt.scatter(refined_node[:,0],refined_node[:,1],color='red',zorder=5,label='Nodes')
plt.gca().set_aspect('equal')
plt.title("Refined Mesh")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()