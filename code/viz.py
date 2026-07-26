 # plot_traj(), make_gif() — importés par les autres


    #Plot TG field + trajectory of the swimmer
    # traj = np.array(list_pos)
    # fig, ax = plt.subplots()
    # q = ax.quiver(X, Y, U, V)
    # ax.quiverkey(q, X=1, Y=1, U=5, label =f'Quiver, length =5', labelpos='E')
    # ax.scatter(traj[0,0], traj[0,1], c='b', marker='o', s=100, label='start')
    # ax.scatter(env.target_x, env.target_y, c='g', marker='*', s=200, label='target')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_aspect('equal')
    # ax.plot(traj[:,0], traj[:,1], 'r--',label='trajectory')
    # ax.set_title(f'v_swim= {env.v_swim}')
    # ax.legend()
    # fig.savefig(f'figs/traj_vswim_{env.v_swim}.png')
    # plt.show()
    # plt.close()

    # #Animation
    # fig2, ax2 = plt.subplots()
    # q2 = ax2.quiver(X, Y, U, V)
    # ax2.quiverkey(q2, X=1, Y=1, U=5, label =f'Quiver, length =5', labelpos='E')
    # ax2.scatter(traj[0,0], traj[0,1], c='b', marker='o', s=100, label='start')
    # ax2.scatter(env.target_x, env.target_y, c='g', marker='*', s=200, label='target')

    # point, = ax2.plot([], [], 'ro')   # nageur, vide au départ
    # trace, = ax2.plot([], [], 'r--')   # trace, vide au départ

    # def update(frame):
    #     point.set_data([traj[frame,0]], [traj[frame,1]])        
    #     trace.set_data(traj[:frame, 0], traj[:frame, 1])        
    #     return point, trace
    
    # anim = FuncAnimation(fig2, update, frames=range(0, len(traj), 5), interval=30)
    # anim.save(f'figs/traj_vswim_{env.v_swim}.gif', writer='pillow')
    # plt.show()