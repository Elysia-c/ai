import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# x=np.random.uniform(10,40,300)
# y=np.random.uniform(100,200,300)
x=np.arange(-5,5,0.1)
y=np.arange(-5,5,0.1)
X,Y=np.meshgrid(x,y)
Z=2*X+Y
# Z=np.sin(np.sprt(X**2+Y**2))

fig=plt.figure()
ax3d=Axes3D(fig)

# ax3d.scatter(x,y,z,c='b',marker="*")
# ax3d.plot_surface(X,Y,Z,cmap="rainbow")
# ax3d.plot_wireframe(X,Y,Z,color="m",linewidth=0.5)
ax3d.plot_surface(X,Y,Z,cmap="rainbow")

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z=2X+Y")

plt.show()