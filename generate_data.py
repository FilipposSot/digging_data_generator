import numpy as np 
import copy
import cv2
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import interpolate
import datetime
import tqdm

def cylindrical_to_cartesian(theta,r,z):
    '''
    Convert point in cylindrical coordinates to cartesian coordinates
    '''
    x = np.multiply(r,np.sin(theta))
    y = np.multiply(r,np.cos(theta))

    return x,y,z

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

def cot(x):
    return 1.0/np.tan(x)

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

class DiggingSimulator():
    def __init__(self, width = 20, height = 20, element_size = 0.05,  bucket_width = 0.75):
        
        self.T = 100
        self.w = width
        self.h = height
        self.bucket_w = bucket_width
        self.element_size = element_size

        self.x = np.linspace(-self.element_size*(self.w-1), self.element_size*(self.w-1), self.w)
        self.y = np.linspace(-self.element_size*(self.h-1), self.element_size*(self.h-1), self.h) + 0.75*self.element_size*(self.h-1)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def generate_surface(self, noise_level = 0.05, surface_type = 'bench', random_heaps = 0):
        '''
        Generate a random surface based on a 2d polynomial function
        '''

        if surface_type == 'polynomial':
            coeff = np.random.uniform(low = -1.0, high = 1.0, size = 9)
            x_vals = np.linspace(-1.0,1.0,self.w)
            y_vals = np.linspace(-1.0,1.0,self.h)
            surface = np.zeros((self.w,self.h))

            for i,x in enumerate(x_vals):
                for j,y in enumerate(y_vals):
                    surface[i,j] = coeff[0]*x + coeff[1]*y + coeff[2]*x**2 +coeff[3]*y**2 + coeff[4]*x*y \
                             +coeff[5]*x**3 + coeff[6]*y**3 + coeff[7]*y*x**2 + coeff[8]*x*y**3 

        elif surface_type == 'bench':

            surface = np.zeros((self.w,self.h))
            g = np.random.uniform(low = 0.6, high = 0.9, size = 1)
            surface = -g*self.Y - np.amin(self.Y)

            # add/remove some random heaps
            for i in range(random_heaps):
                heap_height = np.random.uniform(0.3,0.8,1)
                heap_sigma = np.random.uniform(0.5,1.0,1)
                x_c = np.random.uniform(low = np.amin(self.X), high = np.amax(self.X), size = 1)
                y_c = np.random.uniform(low = 1.0, high = np.amax(self.Y), size = 1)
                # x_c = 1.0 +  np.random.uniform(low = -0.5, high = 0.5, size =1)
                # y_c = 2.5 +  np.random.uniform(low = -0.5, high = 0.5, size =1)
                # print(x_c,y_c)
                surf_heap_i = heap_height*np.exp(-(np.square(self.X-x_c) + np.square(self.Y-y_c))/heap_sigma**2)
                surface = surface + np.sign(np.random.uniform(-1,2,1))*surf_heap_i

            surface[:self.h//4,:] = np.amax(surface[self.h//4,:])
        # print('#########################')
        surface = surface + np.random.uniform(low = -noise_level, high = noise_level, size = surface.shape)

        return surface

    def dig_surface(self, surf, x_traj, y_traj, z_traj, phi_traj):
        '''
        Perform a simulated excavation cycle. It converts the input soil surface
        based on the input trajectory (in cartesian space + bucket yaw).
        '''

        surf_new = copy.copy(surf)

        for i in range(len(x_traj)):

            x_min = x_traj[i] - 0.5*self.bucket_w*np.cos(phi_traj[i])
            x_max = x_traj[i] + 0.5*self.bucket_w*np.cos(phi_traj[i])
            y_min = y_traj[i] - 0.5*self.bucket_w*np.sin(phi_traj[i])
            y_max = y_traj[i] + 0.5*self.bucket_w*np.sin(phi_traj[i])

            x_idx_min, y_idx_min = pos2index(x_min, y_min, self.X, self.Y)
            x_idx_max, y_idx_max = pos2index(x_max, y_max, self.X, self.Y)

            for jx in range(x_idx_min,x_idx_max):

                jy = round(y_idx_min + (jx-x_idx_min)*(y_idx_max - y_idx_min)/(x_idx_max - x_idx_min))

                new_depth = z_traj[i]
                surf_new[jy, jx] = new_depth

        return surf_new

    def diffuse_soil(self, surf):
        '''
        Soil is diffused to maintain physicaly meaningful soil repose angles
        '''

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


    def generate_expert_trajectory(self, surf, target_area = 3.0, expert_lr = 0.02, return_wp = True):
        ''' 
        function that generates simulated "expert" trajectory based on surface profile.
        Based on heuristic rules for where to excavate.
        '''
        f_z = interpolate.interp2d(self.x, self.y, surf)#, kind='cubic')

        theta_candidates = np.linspace(-np.pi/8,np.pi/8,50)
        x_candidates = np.multiply(2.5,np.sin(theta_candidates))
        y_candidates = 2.5*np.ones(theta_candidates.shape)
        z_candidates = f_z(x_candidates,y_candidates)[0]   

        # self.plot_surf_and_traj(surf, x_candidates, y_candidates, z_candidates)
        
        # plt.figure()
        # plt.plot(x_candidates,z_candidates)
        # plt.show()

        theta = theta_candidates[np.argmax(z_candidates)]

        r_init =  3.5
        r_final = 1.25

        t_waypoints = np.array([0,0.3,0.5,0.7,1.0])
        t_array = np.linspace(0,1,self.T+np.random.randint(5))

        f_r = interpolate.interp1d(np.array([0,1]),np.array([r_init,r_final]))

        z_init = f_z(r_init*np.sin(theta), r_init*np.cos(theta))[0]
        z_final = f_z(r_final*np.sin(theta), r_final*np.cos(theta))[0]

        r_1 = f_r(0.3)
        r_2 = f_r(0.5)
        r_3 = f_r(0.8)

        depth = 0.75

        for N_it in range(10):
            
            N_it += 1

            # z_1 = f_z(r_1*np.sin(theta), r_1*np.cos(theta))[0] - depth
            # z_2 = f_z(r_2*np.sin(theta), r_2*np.cos(theta))[0] - depth
            # z_3 = f_z(r_3*np.sin(theta), r_3*np.cos(theta))[0] - depth
            
            z_1 = (0.7*z_init + 0.3*z_final) - depth
            z_2 = (0.5*z_init + 0.5*z_final) - depth
            z_3 = (0.2*z_init + 0.8*z_final) - depth
            
            # tck_z = interpolate.splrep(t_waypoints, np.array([z_init,z_1,z_2,z_3,z_final]), s=0)
            # tck_r = interpolate.splrep(t_waypoints, np.array([r_init,r_1,r_2,r_3,r_final]), s=0)
            # z_array = interpolate.splev(t_array, tck_z, der=0)
            # r_array = interpolate.splev(t_array, tck_r, der=0)

            f_z_traj = interpolate.interp1d(t_waypoints,np.array([z_init,z_1,z_2,z_3,z_final]),kind='cubic')
            f_r_traj = interpolate.interp1d(t_waypoints,np.array([r_init,r_1,r_2,r_3,r_final]),kind='cubic')
            z_array = f_z_traj(t_array)
            r_array = f_r_traj(t_array)

            z_soil = f_z(r_array*np.sin(theta),  r_array*np.cos(theta))[0]

            Dz = z_soil - z_array
            Dr = np.abs(r_array[1:] - r_array[:-1])

            A_trapz = 0.5*np.multiply((Dz[1:] + Dz[:-1]),Dr)
            A_soil = np.sum(A_trapz)

            epsilon = target_area - A_soil

            if np.abs(epsilon) < 0.05:
                break

            depth = depth + expert_lr*epsilon
        
        # print(depth,N_it,A_soil)

        theta_array = np.zeros(r_array.shape) + theta

        return t_array, np.vstack((theta_array, r_array, z_array, np.gradient(theta_array,1/self.T), np.gradient(r_array,1/self.T), np.gradient(z_array,1/self.T)))

    def generate_soil_force_trajectory(self, surf, r_traj, z_traj, theta_traj, r_dot_traj, z_dot_traj, model='FEE+'):
        '''
        from a trajectory and soil surface generate mock excavation force data. 
        For information on force models see:

        "Review of Resistive Force Models for Earthmoving Processes" S. Blouin et al.
        Journal of Aerospace Engineering, Volume 14 Issue 3 - July 2001
        '''

        ### Explanation of Terms ###

        # rho is the rake angle: keep constant (10-30 degrees)
        # beta is the soil shear angle: (find from soil data tables)
        # phi is the internal friction angle: (find from soil data tables)
        # delta is the external friction angle: (find from table)
        # gamma is the soil density: (find from material data)
        # c is cohesion (find from table)
        # d is the instantaneous depth find from depth-bucket location
        # q  is the integration of soil along the path
        # p_c

        rho = np.pi/6
        beta = np.pi/6
        phi = np.pi/6
        delta = np.pi/6
        gamma = 100.0
        c = 1.0
        g = 9.8
        
        f_z = interpolate.interp2d(self.X, self.Y, surf, kind='cubic')
        z_soil = np.flip(f_z(-np.multiply(r_traj,np.sin(theta_traj)),  np.multiply(r_traj,np.cos(theta_traj)))[:,0])
        
        d_traj = z_soil - z_traj
        q_traj = np.zeros(d_traj.shape)
        for i in range(len(q_traj)):
            q_traj[i] = -np.trapz(d_traj[:i+1],r_traj[:i+1]) # very inefficient

        F = np.empty(d_traj.shape)
        F_r = np.empty(d_traj.shape)
        F_z = np.empty(d_traj.shape)

        for i in range(len(d_traj)):
            d = d_traj[i]
            q = q_traj[i]
            N_g = (cot(rho) + cot(beta))/(2*(np.cos(rho+beta) + np.sin(rho+beta)*cot(beta+phi)))
            N_c = (1+cot(beta)*cot(beta+phi))/(np.cos(rho+delta) + np.sin(rho+delta)*cot(beta+phi))
            N_q = (cot(rho) + cot(beta))/(np.cos(rho+beta) + np.sin(rho+beta)*cot(beta+phi))

            F[i] = (N_g*gamma*g*d**2  +  0*N_c*c*d  +  0*N_q*q*g*d)*w
            F_r[i] = F[i]*np.sin(rho + delta)
            F_z[i] = F[i]*np.cos(rho + delta)

        return F_r, F_z

    def plot_surf_and_traj(self, surf, x_traj=None, y_traj=None, z_traj=None, colour = 'r'):
        '''
        plot a soil surface an optionally a trajectory too
        '''

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(self.X, self.Y, surf, rstride=1, cstride=1, linewidth=0.5)
        if x_traj is not None:
            ax.plot3D(x_traj, y_traj, z_traj,colour)
        ax.view_init(30, 50)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        axisEqual3D(ax)
        # plt.show()

    def create_dataset(self, N = 10, filename = None):
        
        if filename is None:
            filename = prefix + datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        # surfaces_pre = np.empty((0,self.w,self.h))
        # surfaces_post = np.empty((0,self.w,self.h))
        # trajectories = np.empty((0,6,self.T))

        surfaces = []
        T = []
        Q = []
        Qdot = []

        for i in tqdm.tqdm(range(N)):
            surf = self.diffuse_soil(self.generate_surface(random_heaps = 5))
            t, traj_cylindrical = self.generate_expert_trajectory(surf)
            # surfaces_pre = np.append(surfaces_pre, np.expand_dims(surf, axis=0), axis = 0)
            # trajectories = np.append(trajectories, np.expand_dims(traj_cylindrical, axis=0), axis = 0)
            T.append(t)
            Q.append(traj_cylindrical[:3,:].tolist())
            Qdot.append(traj_cylindrical[3:,:].tolist())
            surfaces.append(surf.tolist())

        np.savez(filename,
            T = np.array(T),
            Q = np.array(Q),
            Qdot = np.array(Qdot),
            surfaces = np.array(surfaces))


if __name__ == '__main__':

    w = 50
    h = 50
    element_size = 0.05
    surfaces = []
    hashes = []

    DS = DiggingSimulator(width = w, height = h, element_size = element_size)
    # DS.create_dataset()
    
    surf = DS.diffuse_soil(DS.generate_surface(random_heaps = 4))
    # # DS.plot_surf_and_traj(surf)

    t, traj = DS.generate_expert_trajectory(surf)
    # # exit()

    theta_traj, r_traj, z_traj, theta_dot_traj, r_dot_traj, z_dot_traj = traj[0,:],traj[1,:],traj[2,:],traj[3,:],traj[4,:],traj[5,:]
    # F_r, F_z = DS.generate_soil_force_trajectory(surf, r_traj, z_traj, theta_traj, r_dot_traj, z_dot_traj)
    x_traj, y_traj, z_traj = cylindrical_to_cartesian(theta_traj,r_traj,z_traj)

    surf_dug = DS.dig_surface(surf, x_traj, y_traj, z_traj, theta_traj)
    surf_new = DS.diffuse_soil(surf_dug)

    DS.plot_surf_and_traj(surf, x_traj, y_traj, z_traj)
    DS.plot_surf_and_traj(surf_dug, x_traj, y_traj, z_traj)
    DS.plot_surf_and_traj(surf_new, x_traj, y_traj, z_traj)
    plt.show()

    # fig = plt.figure()
    # ax = fig.gca()
    # ax.plot(t,F_r/np.amax(F_r))
    # ax.plot(t,F_z/np.amax(F_r))
    # ax.set_xlabel('Time (normalized)')
    # ax.set_ylabel('Force (normalized)')
    # ax.legend(['F_horizontal','F_vertical'])
    # plt.show()

    #####################################################################################
    # theta_traj, r_traj, z_traj, theta_wp, r_wp, z_wp = generate_expert_trajectory(X_surf, Y_surf, surf)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_wireframe(X_surf, Y_surf, surf, rstride=1, cstride=1)
    # ax.plot3D(x_traj, y_traj, z_traj,'r')
    # ax.plot3D(x_wp, y_wp, z_wp,'.k')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # # plt.imshow(surf)
    # plt.show()



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