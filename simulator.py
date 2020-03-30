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

        self.X_surf = np.linspace(-2, 2, w)
        self.Y_surf = np.linspace(-2, 2, h)
        self.X_surf, self.Y_surf = np.meshgrid(self.X_surf, self.Y_surf)

        self.elevation = gd.generate_random_polynomial_surface(width = w, height = h)

        self.excavator_position = (0.0,0.0)

        self.updating = False
        # self.dump_truck_position = (1.0,1.0)

        cv2.imshow('img', np.zeros((self.elevation.shape[0], self.elevation.shape[1], 3), np.uint8))
        cv2.setMouseCallback('img', self.onMouse)
        cv2.waitKey(10)

    def Simulate(self):
        # img = np.zeros((500,500,3),np.uint8)
        for _ in range(1000):
            # img = np.zeros((500,800,3),np.uint8)
            # img[:,:,0] = self.elevation
            # if self.center is not None:
            # cv2.circle(img, self.excavator_position, 10, (0,255,255), -1)
            # cv2.circle(img, self.dump_truck_position, 10, (0,0,255), -1)
            img = self.get_display_image(self.elevation)

            cv2.imshow('img', img)
            cv2.waitKey(10)

    def get_display_image(self, surf, z_min=None, z_max=None, img_size=(500,500)):
        img = np.zeros((surf.shape[0], surf.shape[1], 3), np.uint8)
        if z_min is None:
            z_min = np.amin(surf)
        if z_max is None:
            z_max = np.amax(surf)

        surf[surf < z_min] = z_min
        surf[surf > z_max] = z_max

        img[:,:,0] = (255*(surf - z_min)/(z_max - z_min)).astype(np.uint8)
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)

        return img_resized

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.updating:
            self.updating = True
            self.center = (x,y)
            theta_traj, r_traj, z_traj, theta_wp, r_wp, z_wp = gd.generate_expert_trajectory(self.X_surf, self.Y_surf, self.elevation)
            x_traj, y_traj, z_traj = gd.cylindrical_to_cartesian(theta_traj,r_traj,z_traj)
            x_wp, y_wp, z_wp = gd.cylindrical_to_cartesian(theta_wp, r_wp, z_wp)
            surf_dug = gd.dig_surface(self.elevation, self.X_surf, self.Y_surf, x_traj, y_traj, z_traj, theta_traj, w_bucket = 0.2)
            surf_new = gd.diffuse_soil(surf_dug)
            self.elevation = surf_new
            print self.elevation.shape
            self.updating = False

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