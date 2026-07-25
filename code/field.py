import numpy as np
import matplotlib.pyplot as plt

def taylor_green_field(x,y):
    u = np.sin(x)*np.cos(y)
    v = -np.cos(x)*np.sin(y)
    return u, v

X = np.linspace(0, 2*np.pi, 50)
Y = np.linspace(0, 2*np.pi, 50)
x, y = np.meshgrid(X, Y, indexing='xy')

U, V = taylor_green_field(x, y)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q,X=1, Y=1, U=10, label ='Quiver, length =10', labelpos='E')

plt.show()