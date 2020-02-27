import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def init_problem_1(N, L, Vp=[0,0,0,0], plot_nodes=False):
    ## Node generation for selected discretization : 
    x = np.linspace(0,L,N)
    y = np.linspace(0,L,N)
    xx, yy = np.meshgrid(x,y)
    
    ## Assigning boundary values : 
    V_A = Vp[0]; V_B = Vp[1]; V_C = Vp[2]; V_D = Vp[3]
    
    V_A_loc = yy == 0
    V_B_loc = xx == L
    V_C_loc = yy == L
    V_D_loc = xx == 0
    
    V = np.zeros_like(xx, dtype=np.float32)
    V[V_A_loc] = V_A
    V[V_B_loc] = V_B
    V[V_C_loc] = V_C
    V[V_D_loc] = V_D
    
    ## Holds information which potentials are unknown 
    I = ~(V_A_loc | V_B_loc | V_C_loc | V_D_loc)

    ## Plot problem nodes : 
    if plot_nodes:
        plot_geometry(xx,yy,Vp,I,[V_A_loc, V_B_loc, V_C_loc, V_D_loc])
    
    return V, I, xx, yy

def init_problem_2(N, Lx, Ly, Vp=[0,0], plot_nodes=False):
    ## Node generation for selected discretization : 
    x = np.linspace(0,Lx,N)
    y = np.linspace(0,Ly,N)
    xx, yy = np.meshgrid(x,y)
    
    ## Assigning boundary values : 
    V_A = Vp[0]; V_B = Vp[1]
    
    indA = yy == Ly
    indB = xx <= 2/3*Lx
    indC = xx >= 1/3*Lx
    V_A_loc = indA & indB & indC 
    V_B_loc = yy == 0

    # Set prescribed potentials :
    V = np.zeros_like(xx, dtype=np.float32)
    V[V_A_loc] = V_A
    V[V_B_loc] = V_B

    ## Holds information which potentials are unknown 
    I = ~(V_A_loc | V_B_loc)

    if plot_nodes:
        plot_geometry(xx,yy,Vp,I,[V_A_loc, V_B_loc])
    
    return V, I, xx, yy
    
def solve(N, V, I, matprop=1):
    # Indices of unknown potentials :
    yi, xi = np.meshgrid(np.arange(N), np.arange(N))
    xi = xi[I]
    yi = yi[I]
    
    # Node number mapping (unknown potentials) : 
    mainmap = np.arange(N**2).reshape(N,N).T
    submap = np.zeros_like(V, dtype=np.int32)
    index = 0
    for x,y in zip(xi, yi):
        submap[x,y] = index
        index += 1
    
    # Stiffness matrix and known potentials vector : 
    n_unknown = len(xi)
    K = np.eye(n_unknown)*4
    Vk = np.zeros((n_unknown, 1))
    
    for ind, xy in enumerate(zip(xi,yi)):
        indices = np.array([[xy[0], xy[1]+1],
                            [xy[0], xy[1]-1],
                            [xy[0]+1, xy[1]],
                            [xy[0]-1, xy[1]]])

        nodeout = np.logical_or(indices >= N, indices < 0)
        nodeout = np.any(nodeout, axis=1)
        indices = indices[~nodeout,:]

        ind_notknown = I[indices[:,0], indices[:,1]]
        nknodes = submap[indices[ind_notknown,0], indices[ind_notknown,1]]
        
        K[ind, nknodes] = -1
        Vk[ind] = np.sum(V[indices[~ind_notknown,0], indices[~ind_notknown,1]])
        
    
    # Solving and placing calculated potentials into potential matrix : 
    Vu = np.linalg.solve(K, Vk)
    Vsolution = V.T.flatten()
    Vsolution[mainmap[I][:,None]] = Vu[submap[I]]
    V = Vsolution.reshape(N,N).T

    # Calculating field components : 
    dVy, dVx = np.gradient(V)
    Ex = -matprop*dVx
    Ey = -matprop*dVy
    
    return V, Ex, Ey

def plot_geometry(xx,yy,Vp,I,VI):
    colors = ["m", "r", "g", "y"]
    names = ["$V_A=$", "$V_B=$", "$V_C=$", "$V_D=$"]
    _, ax = plt.subplots(1,1, figsize=(6,6))
    for ind in range(len(Vp)):
        ax.scatter(xx[VI[ind]],yy[VI[ind]],s=2,c=colors[ind],label=names[ind]+str(Vp[ind]))

    ax.scatter(xx[I],yy[I],s=2,c="b",label="Unknown")
    ax.set_title("Known and unknown node potentials")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()

def plot_potential(V,xx,yy,levels=8,titl=""):
    _, ax = plt.subplots(1,1,figsize=(6,6))
    pc = ax.pcolor(xx,yy,V, cmap='rainbow')
    plt.colorbar(pc)
    ax.contour(xx, yy, V, colors='k', alpha=0.6, levels=levels)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(titl)

    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    pc = ax.plot_surface(xx, yy, V, cmap='rainbow', edgecolor='none')
    plt.colorbar(pc)
    ax.set_title(titl)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel('V')
    plt.show()

def plot_field(Ex,Ey,V,xx,yy,titl=""):

    E = np.sqrt(Ex**2 + Ey**2)

    # Normalize vectors for plot : 
    Ex = Ex/E
    Ey = Ey/E

    _, ax = plt.subplots(1,1,figsize=(6,6))
    ax.quiver(xx,yy,Ex,Ey,color='b',scale=40, label="E-Field")
    ax.contour(xx,yy,V,colors='k',alpha=0.5,levels=8)
    ax.set_title(titl)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    _, ax = plt.subplots(1,1,figsize=(6,6))
    pc = ax.pcolor(xx,yy,E,cmap="rainbow")
    plt.colorbar(pc)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")