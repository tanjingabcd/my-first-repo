import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.mesh import UniformMesh1d
from fealpy.sparse import csr_matrix

class SinPDEData:
    def __init__(self):
        pass

    def domain(self):
        return [0, 1]

    def solution(self, x):
        return bm.sin(4 * bm.pi * x)

    def source(self, x):
        return 16 * bm.pi**2 * bm.sin(4 * bm.pi * x)

pde = SinPDEData()
domain = pde.domain()

nx = 100
hx = (domain[1] - domain[0]) / nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

node = mesh.entity('node')
uI = pde.solution(node) # node.shape==(NN, 1)
print("uI.shape=", uI.shape)

# generate stiffness matrix
cx = 1/hx**2
NN = mesh.number_of_nodes() 

k = bm.arange(NN)


fig = plt.figure()
axes = fig.add_subplot(121)
mesh.add_plot(axes)
axes = fig.add_subplot(122)
mesh.show_function(axes, uI, 
                   box=[0, 1, -1, 1])
plt.show()




