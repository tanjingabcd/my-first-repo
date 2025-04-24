from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import UniformMesh3d

# Create a uniform mesh on the interval [0, 1]
domain = [0, 1, 0, 1, 0, 1]
nx = 2 
ny = 2 
nz = 2
hx = (domain[1]-domain[0]) / nx
hy = (domain[3]-domain[2]) / ny
hz = (domain[5]-domain[4]) / nz
mesh = UniformMesh3d([0, nx, 0, ny, 0, nz], 
                     h=(hx, hy, hz), 
                     origin=(domain[0], domain[2], domain[4]))

node = mesh.entity('node') # (NN, 3) get the node tensor 
edge = mesh.entity('edge') # (NE, 2) get the edge tensor
face = mesh.entity('face') # (NF, 4) get the face tensor
cell = mesh.entity('cell') # (NC, 8) get the cell tensor

print("node.shpe:\n", node.shape)
print("cell.shape:\n", cell.shape)
print("node coordinates:\n", node)
print("cell tensor:\n", cell)

# Plot the mesh
fig = plt.figure() # create a figure
axes = fig.add_subplot(111, projection='3d') # add a subplot
axes.set_title('3D Uniform Mesh') # set the title
mesh.add_plot(axes, linewidth=3)
mesh.find_node(axes, showindex=True, fontsize=30)
mesh.find_edge(axes, showindex=True, fontsize=35)
mesh.find_cell(axes, showindex=True, fontsize=40)
plt.show()

