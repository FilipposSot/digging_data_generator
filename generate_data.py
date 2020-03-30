import numpy as np 
import copy
import cv2
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime

ELEMENT_SIZE = 0.1
BUCKET_WIDTH = 0.5
EXPERT_LR = 0.1

def cylindrical_to_cartesian(theta,r,z):
    '''
    Convert point in cylindrical coordinates to cartesian coordinates
    '''

    x = -np.multiply(r,np.sin(theta))
    y = np.multiply(r,np.cos(theta))

    return x,y,z

def generate_random_polynomial_surface(width = 19, height = 10, noise_level = 0.1):
    '''
    Generate a random surface based on a 2d polynomial function
    '''

    coeff = np.random.uniform(low = -1.0, high = 1.0, size = 9)
    x_vals = np.linspace(-1.0,1.0,width)
    y_vals = np.linspace(-1.0,1.0,height)
    surface = np.zeros((width,height))

    for i,x in enumerate(x_vals):
        for j,y in enumerate(y_vals):
            surface[i,j] = coeff[0]*x + coeff[1]*y + coeff[2]*x**2 +coeff[3]*y**2 + coeff[4]*x*y \
                     +coeff[5]*x**3 + coeff[6]*y**3 + coeff[7]*y*x**2 + coeff[8]*x*y**3 

    surface = surface + np.random.uniform(low = -noise_level, high = noise_level, size = surface.shape)

    return surface

def dig_surface(surf, X, Y, x_traj, y_traj, z_traj, phi_traj, w_bucket = 0.1):
    '''
    Perform a simulated excavation cycle. It converts the input soil surface
    based on the input trajectory (in cartesian space + bucket yaw).
    '''

    surf_new = copy.copy(surf)

    for i in range(len(x_traj)):

        x_min = x_traj[i] - 0.5*BUCKET_WIDTH*np.cos(phi_traj[i])
        x_max = x_traj[i] + 0.5*BUCKET_WIDTH*np.cos(phi_traj[i])
        y_min = y_traj[i] - 0.5*BUCKET_WIDTH*np.sin(phi_traj[i])
        y_max = y_traj[i] + 0.5*BUCKET_WIDTH*np.sin(phi_traj[i])

        x_idx_min, y_idx_min = pos2index(x_min,y_min,X,Y)
        x_idx_max, y_idx_max = pos2index(x_max,y_max,X,Y)

        for jx in range(x_idx_min,x_idx_max):

            jy = round(y_idx_min + (jx-x_idx_min)*(y_idx_max - y_idx_min)/(x_idx_max - x_idx_min))

            new_depth = z_traj[i]
            surf_new[jy , jx] = new_depth

    return surf_new

def diffuse_soil(surf):
    '''
    Soil is diffused to maintain physicaly meaningful soil repose angles
    '''
    # for N_it in range(20):

    dz_t = 0.2
    
    surf_new = copy.copy(surf)

    for k in range(30):
        for i in range(1,surf.shape[0]-1):
            for j in range(1,surf.shape[1]-1):

                z0 = surf_new[i,j]
                z1 = surf_new[i-1,j]
                z2 = surf_new[i,j+1]
                z3 = surf_new[i+1,j]
                z4 = surf_new[i,j-1]

                dz1 = z0 - z1
                dz2 = z0 - z2
                dz3 = z0 - z3
                dz4 = z0 - z4

                dz_sum = 0.0
                dz_list = []

                if dz1 > dz_t:
                    dz_sum += dz1
                    dz_list.append(dz1)
                if dz2 > dz_t:
                    dz_sum += dz2
                    dz_list.append(dz2)
                if dz3 > dz_t:
                    dz_sum += dz3
                    dz_list.append(dz3)
                if dz4 > dz_t:
                    dz_sum += dz4
                    dz_list.append(dz4)

                if len(dz_list) > 0:
                    dz_min = np.amin(np.array(dz_list))
                    z_removed = 0.1*dz_min

                    if dz1 > dz_t:
                        surf_new[i-1,j] += z_removed*dz1/dz_sum
                    if dz2 > dz_t:
                        surf_new[i,j+1] += z_removed*dz2/dz_sum
                    if dz3 > dz_t:
                        surf_new[i+1,j] += z_removed*dz3/dz_sum
                    if dz4 > dz_t:
                        surf_new[i,j-1] += z_removed*dz4/dz_sum

                    surf_new[i,j] += -z_removed

    return surf_new

def relu(x):
    return max(0,x)

def pos2index(x,y,X,Y):
    '''
    For a given grid of X and Y coordinates it converts a specific 
    coordinate (x,y) to its closest index value
    '''

    dx = X[1,1] - X[0,0] 
    dy = Y[1,1] - Y[0,0]

    x0 = X[0,0]
    y0 = Y[0,0]

    idx = round((x-x0)/dx)
    idy = round((y-y0)/dy)

    return int(idx),int(idy)

def generate_expert_trajectory(x,y,z,target_area = 1.0):
    ''' 
    function that generates simulated "expert" trajectory based on surface profile.
    Based on heuristic rules for where to excavate.
    '''

    k = (np.sum(z[x>0]+np.amin(z)) - np.sum(z[x<0]+np.amin(z)))/np.abs(np.sum(z[x>0]+np.amin(z)) + np.sum(z[x<0]+np.amin(z)))
    
    if k<-0.10:
        theta = np.pi/12
    elif k>0.10:
        theta = -np.pi/12
    else:
        theta = 0

    r_init = 1.50
    r_final = 0.0

    t_waypoints = np.array([0,0.3,0.5,0.7,1.0])
    t_array = np.linspace(0,1,100)

    f_z = interpolate.interp2d(x, y, z, kind='cubic')
    f_r = interpolate.interp1d(np.array([0,1]),np.array([r_init,r_final]))

    z_init = f_z(-r_init*np.sin(theta), r_init*np.cos(theta))[0]
    z_final = f_z(-r_final*np.sin(theta), r_final*np.cos(theta))[0]

    r_1 = f_r(0.2)
    r_2 = f_r(0.5)
    r_3 = f_r(0.8)

    depth = 0.75

    for N_it in range(50):
        
        N_it += 1

        z_1 = f_z(-r_1*np.sin(theta), r_1*np.cos(theta))[0] - depth
        z_2 = f_z(-r_2*np.sin(theta), r_2*np.cos(theta))[0] - depth
        z_3 = f_z(-r_3*np.sin(theta), r_3*np.cos(theta))[0] - depth

        tck_z = interpolate.splrep(t_waypoints, np.array([z_init,z_1,z_2,z_3,z_final]), s=0)
        tck_r = interpolate.splrep(t_waypoints, np.array([r_init,r_1,r_2,r_3,r_final]), s=0)

        z_array = interpolate.splev(t_array, tck_z, der=0)
        r_array = interpolate.splev(t_array, tck_r, der=0)

        z_soil = f_z(-r_array*np.sin(theta),  r_array*np.cos(theta))[0]

        Dz = z_soil - z_array
        Dr = np.abs(r_array[1:] - r_array[:-1])

        A_trapz = 0.5*np.multiply((Dz[1:] + Dz[:-1]),Dr)
        A_soil = np.sum(A_trapz)

        epsilon = target_area - A_soil

        if np.abs(epsilon) < 0.05:
            break

        depth = depth + EXPERT_LR*epsilon

    print(A_soil,depth)

    theta_array = np.zeros(r_array.shape) + theta

    return theta_array, r_array, z_array, np.array([0.0,0.0,0.0,0.0,0.0]) + theta, np.array([r_init,r_1,r_2,r_3,r_final]), np.array([z_init,z_1,z_2,z_3,z_final])


def create_dataset(N = 1000, w = 20, h = 20, prefix = "dataset_"):
    
    file_name = prefix + now.strftime("%d_%m_%Y__%H_%M_%S")
    surfaces = []
    hashes = []

    X_surf = np.linspace(-2, 2, w)
    Y_surf = np.linspace(-2, 2, h)
    X_surf, Y_surf = np.meshgrid(X_surf, Y_surf)


    for i in range(N):
        surf = generate_random_surface()
        surfaces.append(surf)

if __name__ == '__main__':
    N = 5000
    w = 50
    h = 50
    surfaces = []
    hashes = []

    X_surf = np.linspace(-2, 2, w)
    Y_surf = np.linspace(-2, 2, h)
    X_surf, Y_surf = np.meshgrid(X_surf, Y_surf)

    surf = generate_random_polynomial_surface(width = w, height = h)
    theta_traj, r_traj, z_traj, theta_wp, r_wp, z_wp = generate_expert_trajectory(X_surf, Y_surf, surf)
    x_traj, y_traj, z_traj = cylindrical_to_cartesian(theta_traj,r_traj,z_traj)
    x_wp, y_wp, z_wp = cylindrical_to_cartesian(theta_wp, r_wp, z_wp)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(X_surf, Y_surf, surf, rstride=1, cstride=1)
    ax.plot3D(x_traj, y_traj, z_traj,'r')
    ax.plot3D(x_wp, y_wp, z_wp,'.k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # plt.imshow(surf)
    plt.show()

    surf_dug = dig_surface(surf, X_surf, Y_surf, x_traj, y_traj, z_traj, theta_traj, w_bucket = 0.2)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(X_surf, Y_surf, surf_dug, rstride=1, cstride=1)
    ax.plot3D(x_traj, y_traj, z_traj,'r')
    ax.plot3D(x_wp, y_wp, z_wp,'.k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.imshow(surf_dug)
    plt.show()

    surf_new = diffuse_soil(surf_dug)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(X_surf, Y_surf, surf_new, rstride=1, cstride=1)
    ax.plot3D(x_traj, y_traj, z_traj,'r')
    ax.plot3D(x_wp, y_wp, z_wp,'.k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.imshow(surf_dug)
    plt.show()

    # for i in range(N):
    #     surf = generate_random_surface()
    #     surfaces.append(surf)
    #     surf_hash = dhash(surf)
    #     hash_vec = [int(x) for x in surf_hash]
    #     hashes.append(hash_vec)

    # X = np.array(hashes)
    # nbrs = NearestNeighbors(n_neighbors=5, metric = 'hamming').fit(X)
    
    # surf_new = generate_random_surface()
    # X_new = [int(x) for x in dhash(surf_new)]
    # distances, indices = nbrs.kneighbors(np.array(X_new).reshape(1, -1))

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_wireframe(X_surf, Y_surf, surf_new, rstride=1, cstride=1)
    # plt.show()

    # for idx in indices[0]:
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.plot_wireframe(X_surf, Y_surf, surfaces[idx], rstride=1, cstride=1)
    #     plt.show()