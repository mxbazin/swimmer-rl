import numpy as np
import matplotlib.pyplot as plt

def taylor_green_field(x,y):
    u = np.sin(x)*np.cos(y)
    v = -np.cos(x)*np.sin(y)
    return u, v

def euler_method(px, py, u, v, dt):
    px_updated = px + dt*u 
    py_updated = py + dt*v
    return px_updated, py_updated

# Initial conditions
x0, y0 = 2, 1

dt = 0.01
N = 2000

list_pos = []

# Taylor-Green Field 
X = np.linspace(0, 2*np.pi, 50)
Y = np.linspace(0, 2*np.pi, 50)
x, y = np.meshgrid(X, Y, indexing='xy')
U, V = taylor_green_field(x, y)

if __name__ == "__main__":
    print("Starting the script..")

    #Initial pos of the swimmer
    px, py = x0, y0

    # Adding the swimming parameters
    theta = np.pi/2
    v_swim = 0.3

    # Compute the position of the swimmer
    for i in range (0, N):
        u, v = taylor_green_field(px, py)

        # Swimming stuff 
        u = u + v_swim*np.cos(theta)
        v = v + v_swim*np.sin(theta)

        px, py = euler_method(px, py, u, v, dt)
        px = np.clip(px, 0, 2*np.pi)
        py = np.clip(py, 0, 2*np.pi)
        list_pos.append((px, py))

    #Plot TG field + trajectory of the swimmer
    traj = np.array(list_pos)
    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V)
    ax.quiverkey(q,X=1, Y=1, U=5, label ='Quiver, length =5', labelpos='E')
    plt.plot(traj[:,0], traj[:,1], 'r--')
    plt.show()
