#!/usr/bin/python

# Authors:            Ryan Sandzimier (rsandz@mit.edu)

import numpy as np
import cv2

import generate_data as gd
import matplotlib.pyplot as plt


# self.excavator_position = (250,250)
# self.dump_truck_position = (100,100)
class Simulator:
    def __init__(self):

        N = 5000
        w = 50
        h = 50
        surfaces = []
        hashes = []

        self.img_size = (500,500)

        self.x_bounds = (-2,2)
        self.y_bounds = (-2,2)

        self.X_surf = np.linspace(self.x_bounds[0], self.x_bounds[1], w)
        self.Y_surf = np.linspace(self.y_bounds[0], self.y_bounds[1], h)
        self.X_surf, self.Y_surf = np.meshgrid(self.X_surf, self.Y_surf)
        self.r_max = 1.5
        self.r_min = 0.4

        # self.elevation = gd.generate_random_polynomial_surface(width = w, height = h, noise_level=0.03)
        self.elevation = gd.generate_bench_surface(width = w, height = h, noise_level = 0.0*0.03, theta=4*np.pi/4, z_max=1*0, z_min=0, offset1=3, offset2=12)
        self.excavator_position = (0.0,0.0)
        self.truck_pos_polar = (1.05, 0.0)
        self.truck_pos = (self.truck_pos_polar[0]*np.cos(self.truck_pos_polar[1]), self.truck_pos_polar[0]*np.sin(self.truck_pos_polar[1]))

        self.truck_size = (0.9,0.6)

        self.mouse_pos = (0,0)

        self.updating = False

        self.fig = plt.figure(0)
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_xlim3d(-2,2)
        self.ax.set_ylim3d(-2,2)
        self.ax.set_zlim3d(-3,3)

        plt.show(block=False)

        # self.dump_truck_position = (1.0,1.0)

        cv2.imshow('img', np.zeros((self.elevation.shape[0], self.elevation.shape[1], 3), np.uint8))
        cv2.setMouseCallback('img', self.onMouse)
        cv2.waitKey(10)

    def Simulate(self):
        # img = np.zeros((500,500,3),np.uint8)
        for _ in range(100000):
            # img = np.zeros((500,800,3),np.uint8)
            # img[:,:,0] = self.elevation
            # if self.center is not None:
            # cv2.circle(img, self.excavator_position, 10, (0,255,255), -1)
            # cv2.circle(img, self.dump_truck_position, 10, (0,0,255), -1)
            img = self.get_display_image(self.elevation, z_min=-3, z_max=3)
            self.draw()
            cv2.imshow('img', img)
            cv2.waitKey(10)

    def draw(self):
        # ax.plot3D(x_traj, y_traj, z_traj,'r')
        # ax.plot3D(x_wp, y_wp, z_wp,'.k')
        plt.cla()
        # self.ax.plot_wireframe(self.X_surf, self.Y_surf, self.elevation, rstride=1, cstride=1)

        self.ax.plot_surface(self.X_surf, self.Y_surf, self.elevation, rstride=1, cstride=1, cmap='ocean', alpha=0.2)
        a = np.array([self.truck_pos[0] - 0.5*self.truck_size[0] - 0.001, self.truck_pos[0] - 0.5*self.truck_size[0], self.truck_pos[0] + 0.5*self.truck_size[0], self.truck_pos[0] + 0.5*self.truck_size[0] + 0.001])
        b = np.array([self.truck_pos[1] - 0.5*self.truck_size[1] - 0.001, self.truck_pos[1] - 0.5*self.truck_size[1], self.truck_pos[1] + 0.5*self.truck_size[1], self.truck_pos[1] + 0.5*self.truck_size[1] + 0.001])
        x,y = np.meshgrid(a, b)
        z = 1.0*np.ones((4,4))
        z[0,:] = 0
        z[-1,:] = 0
        z[:,0] = 0
        z[:,-1] = 0

        # self.ax.plot_surface(x,y,z, rstride=1, cstride=1, color='r')

        r = int(self.X_surf.shape[0]/2.)
        c = int(self.X_surf.shape[1]/2.)
        # self.ax.scatter(self.X_surf[r,c], self.Y_surf[r,c], self.elevation[r,c]+0.1, s=100, c='y')
        r = self.X_surf.shape[0] - int(self.mouse_pos[1]*self.elevation.shape[1]/self.img_size[1]) - 1
        c = int(self.mouse_pos[0]*self.elevation.shape[0]/self.img_size[0])
        # self.ax.scatter(self.X_surf[r,c], self.Y_surf[r,c], self.elevation[r,c]+0.1, s=100, c='g')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_xlim3d(-2,2)
        self.ax.set_ylim3d(-2,2)
        self.ax.set_zlim3d(-2,2)
        # plt.imshow(surf)
        plt.draw()

        plt.pause(0.001)

    def get_display_image(self, surf, z_min=None, z_max=None):
        img = np.zeros((surf.shape[0], surf.shape[1], 1), np.uint8)
        if z_min is None:
            z_min = np.amin(surf)
        if z_max is None:
            z_max = np.amax(surf)
        surf[surf < z_min] = z_min
        surf[surf > z_max] = z_max

        img[:,:,0] = (255*(surf - z_min)/(z_max - z_min)).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

        img_resized = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
        xx = 0.5*self.truck_size[0]*np.cos(self.truck_pos_polar[1])
        xy = 0.5*self.truck_size[0]*np.sin(self.truck_pos_polar[1])
        yx = -0.5*self.truck_size[1]*np.sin(self.truck_pos_polar[1])
        yy = 0.5*self.truck_size[1]*np.cos(self.truck_pos_polar[1])
        c1 = (self.truck_pos[0] + xx + yx, self.truck_pos[1] + xy + yy)
        c2 = (self.truck_pos[0] - xx + yx, self.truck_pos[1] - xy + yy)
        c3 = (self.truck_pos[0] + xx - yx, self.truck_pos[1] + xy - yy)
        c4 = (self.truck_pos[0] - xx - yx, self.truck_pos[1] - xy - yy)

        corners = np.array([c1,c2,c3,c4])*np.array(self.img_size)/np.array([self.x_bounds[1]-self.x_bounds[0],self.y_bounds[1]-self.y_bounds[0]]) + np.array(self.img_size)/2
        corners = corners.astype(np.int64)
        rect = cv2.minAreaRect(corners)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_resized,[box],-1,(0,0,255),-1)

        cv2.circle(img_resized, (int(self.img_size[1]/2.), int(self.img_size[0]/2.)), 10, (0,255,255),-1)
        r_max_pix = int(self.r_max*np.sqrt(self.img_size[0]**2+self.img_size[1]**2)/np.sqrt((self.X_surf[0,-1]-self.X_surf[0,0])**2 + (self.Y_surf[-1,0]-self.Y_surf[0,0])**2))
        cv2.circle(img_resized, (int(self.img_size[1]/2.), int(self.img_size[0]/2.)), r_max_pix, (0,255,0))
        img_resized = cv2.flip(img_resized, 0)
        return img_resized
    def onMouse(self, event, x, y, flags, param):
        self.mouse_pos = (x,y)

        if event == cv2.EVENT_LBUTTONDOWN and not self.updating:
            self.updating = True
            x = x*(self.X_surf[0,-1]-self.X_surf[0,0])/(self.img_size[1]-1) + self.X_surf[0,0]
            y = (self.img_size[0]-1-y)*(self.Y_surf[-1,0]-self.Y_surf[0,0])/(self.img_size[0]-1) + self.Y_surf[0,0]
            r_init = np.sqrt(x**2+y**2)
            r_final = self.r_min
            theta = np.arctan2(-y,x)
            if r_init > self.r_max or r_init < r_final:
                self.updating = False
                return
            theta_traj, r_traj, z_traj, theta_wp, r_wp, z_wp = gd.generate_expert_trajectory(self.X_surf, self.Y_surf, self.elevation, target_area = 3*0.15, f_z=self.surface_interpolator, theta=theta, r_init=r_init, r_final=r_final)

            x_traj, y_traj, z_traj = gd.cylindrical_to_cartesian(theta_traj,r_traj,z_traj)

            x_wp, y_wp, z_wp = gd.cylindrical_to_cartesian(theta_wp, r_wp, z_wp)
            surf_dug = gd.dig_surface(self.elevation, self.X_surf, self.Y_surf, x_traj, y_traj, z_traj, theta_traj, w_bucket = 0.2)
            surf_new = gd.diffuse_soil(surf_dug)
            # surf_new = surf_dug
            self.elevation = surf_new
            self.updating = False

    def surface_interpolator(self, y_list, x_list):
        if not isinstance(x_list, np.ndarray):
            x_list = np.array([x_list])
        if not isinstance(y_list, np.ndarray):
            y_list = np.array([y_list])
        z_list = []
        for x,y in zip(x_list,y_list):
            ind_x = np.searchsorted(self.X_surf[0,:], x)
            ind_y = np.searchsorted(self.Y_surf[:,0], y)

            x1 = self.X_surf[0,ind_x-1] if ind_x-1 > 0 else self.X_surf[0,0] - 1.0
            y1 = self.Y_surf[ind_y-1,0] if ind_y-1 > 0 else self.Y_surf[0,0] - 1.0

            x2 = self.X_surf[0,ind_x] if ind_x < self.X_surf.shape[1] else self.X_surf[0,-1] + 1.0
            y2 = self.Y_surf[ind_y,0] if ind_y < self.Y_surf.shape[0] else self.Y_surf[-1,0] + 1.0

            z11 = self.elevation[max(ind_y-1,0),max(ind_x-1,0)]
            z12 = self.elevation[min(ind_y,self.Y_surf.shape[0]-1),max(ind_x-1,0)]
            z21 = self.elevation[max(ind_y-1,0),min(ind_x,self.X_surf.shape[1]-1)]
            z22 = self.elevation[min(ind_y,self.Y_surf.shape[0]-1),min(ind_x,self.X_surf.shape[1]-1)]

            Z = np.array([[z11, z12],[z21, z22]])

            z_list.append(np.dot([x2-x,x-x1],np.dot(Z,[y2-y,y-y1]))/(x2-x1)/(y2-y1))
        return z_list

s = Simulator()
s.Simulate()
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


# class Simulation:
#     def __init__(self):
#         self.elevation = np.zeros((500,800),np.uint8)
#         self.elevation[0:200,:] = 255
#         self.elevation[200:300,:] = np.linspace(255, 0, 100).astype(np.uint8).reshape(100,1)

#         self.excavator_position = (250,250)
#         self.dump_truck_position = (100,100)


#         cv2.imshow('img', self.elevation)
#         cv2.setMouseCallback('img', self.onMouse)
#         cv2.waitKey(10)
#         pass
#     def Simulate(self):
#         # img = np.zeros((500,500,3),np.uint8)
#         for _ in range(1000):
#             img = np.zeros((500,800,3),np.uint8)
#             img[:,:,0] = self.elevation
#             # if self.center is not None:
#             cv2.circle(img, self.excavator_position, 10, (0,255,255), -1)
#             cv2.circle(img, self.dump_truck_position, 10, (0,0,255), -1)

#             cv2.imshow('img', img)
#             cv2.waitKey(10)
#     def onMouse(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             self.center = (x,y)

# s = Simulation()
# s.Simulate()




# import rospy
# from sensor_msgs.msg import JointState
# from ur_rtde.msg import ToolState
# import tf
# import math

# class DynamicFrames:
#     def __init__(self):
#         rospy.init_node('dynamic_frames', anonymous=True)
#         rospy.Subscriber('joint_states', JointState, self.cb_joint_states)
#         self.br = tf.TransformBroadcaster()

#     def cb_joint_states(self, msg): # Broadcast transform from robot_base to bucket
#         self.br.sendTransform((0.0,0.0,0.0993),tf.transformations.quaternion_from_euler(0.0,0.0,msg.position[0],axes="rzxz"),msg.header.stamp,"robot_link1","robot_base")
#         self.br.sendTransform((0.0,-0.0946,0.0814),tf.transformations.quaternion_from_euler(0.0,math.pi/2,msg.position[1],axes="rzxz"),msg.header.stamp,"robot_link2","robot_link1")
#         self.br.sendTransform((-0.6127,0.0,0.018),tf.transformations.quaternion_from_euler(0.0,0.0,msg.position[2],axes="rzxz"),msg.header.stamp,"robot_link3","robot_link2")
#         self.br.sendTransform((-0.57155,0.0,-0.0169),tf.transformations.quaternion_from_euler(0.0,0.0,msg.position[3],axes="rzxz"),msg.header.stamp,"robot_link4","robot_link3")
#         self.br.sendTransform((0.0,-0.0569,0.07845),tf.transformations.quaternion_from_euler(0.0,math.pi/2,msg.position[4],axes="rzxz"),msg.header.stamp,"robot_link5","robot_link4")
#         self.br.sendTransform((0.0,0.0569,0.06295),tf.transformations.quaternion_from_euler(0.0,-math.pi/2,msg.position[5],axes="rzxz"),msg.header.stamp,"robot_link6","robot_link5")
#         self.br.sendTransform((0.0,0.0,0.05965),tf.transformations.quaternion_from_euler(0.0,0.0,0.0),msg.header.stamp,"bucket","robot_link6")

# if __name__ == '__main__':
#     df = DynamicFrames()
#     try:              
#         rospy.spin()

#     except rospy.ROSInterruptException:
#         pass