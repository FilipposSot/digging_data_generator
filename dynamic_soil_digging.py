import numpy as np 
from scipy.integrate import ode
import matplotlib.pyplot as plt

def cot(x):
    return 1.0/np.tan(x)


class DiggingSimulator2D():
    
    def __init__(self):


        print("Simulator instatiated")

        return
        
    def set_s_func(self):
        '''
         This function sets the soil surface shape used by the dynamics simulator

         For now it isnt doing anything
                 '''
        print("Using flat soil surface")

        return 0

    def calc_s(self,x):
        '''
        Returns the soil surface variables at a particular x value
        '''

        # for now just a flat surface at z = 0 
        s = 0
        s_dash = 0
        s_dash_dash = 0

        return s, s_dash, s_dash_dash

    @staticmethod
    def calc_F_soil(D, x, z, v_x, v_z, q):
        '''
        ### Explanation of Terms 

        rho is the rake angle: keep constant (10-30 degrees)
        beta is the soil shear angle: (find from soil data tables)
        phi is the internal friction angle: (find from soil data tables)
        delta is the external friction angle: (find from table)
        gamma is the soil density: (find from material data)
        c is cohesion (find from table)
        d is the instantaneous depth find from depth-bucket location
        q  is the integration of soil along the path
        '''

        # rho = np.pi/6
        # beta = np.pi/6
        # phi = np.pi/6
        # delta = np.pi/6
        # gamma = 100.0
        # c = 1.0
        # g = 9.8
                
        # N_g = (cot(rho) + cot(beta))/(2*(np.cos(rho+beta) + np.sin(rho+beta)*cot(beta+phi)))
        # N_c = (1+cot(beta)*cot(beta+phi))/(np.cos(rho+delta) + np.sin(rho+delta)*cot(beta+phi))
        # N_q = (cot(rho) + cot(beta))/(np.cos(rho+beta) + np.sin(rho+beta)*cot(beta+phi))
        # N_a = (np.tan(rho) + cot(rho+phi))/(np.cos(rho+beta) + np.sin(rho+beta)*cot(beta+phi))

        F = (-D**2 + -10*D*v_x)*np.array([1.0,0.0])
        F = ( D**2 + -10*D*v_z)*np.array([0.0,1.0]) + F


        # K_g = 1.0
        # K_c = 1.0
        # K_q = 1.0
        # K_a = 1.0

        # F_x = -D**2 +  K_c*D  +  K_q*q*D 

        # F_x = -F - K_a*D*v_x**2
        # print(F_x)
        # F_z =  F - K_a*D*v_z**2*np.sign(v_z)

        return F

    @staticmethod
    def calc_I(gamma,D):
        '''
        Calculates variable system inertia. 
        gamma is the bucket filling 
        D is the depth

        NOTE: Currently this is just a hypothetical 
        '''
        return (1.0 + D*gamma)*np.diag([1,1])

    def f(self, t, xi, u):
        '''
        Continuous time state dynamics 
        x, z  bucket position
        v_x, v_z bucket velocity
        gamma bucket fill
        '''

        # unpack the state variables
        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        
        # Get the soil shape at the current bucket location
        s, _ , _ = self.calc_s(x)
        
        # Calculate the depth of the bucket
        D = s-z

        # Calculate the inertia matrix (this is based on bukcet filling and depth)
        I = self.calc_I(gamma,D)

        # Calculate the force from the soil
        F_soil = self.calc_F_soil(D, x, z, v_x, v_z,gamma)

        # Add some noise to represent soil stochasticity
        F_noise =  np.random.normal(np.array([0.0,0.0]),np.array([0.1,0.1]))

        # Calculate the time derivatives
        v_dot = np.linalg.inv(I).dot(F_soil + u) 
        gamma_dot = v_x*D
        x_dot = v_x
        z_dot = v_z

        xi_dot = np.array([x_dot, z_dot, v_dot[0], v_dot[1], gamma_dot]) 

        return xi_dot

    def g(self, t, xi, u):
        '''
        Output observation

        For now its just a measurement of the states
        '''
        return xi
   
    def simulate_system(self, x_0, u_0, dt=0.01):
        '''
        Simulate a system in continuous time
        Arguments:
        x_0: initial state
        '''

        # initial time and input
        t = 0.0

        # create numerical integration object and initial values
        r = ode(self.f).set_integrator('vode', method = 'bdf', max_step = 0.001)
        
        # set initial state, time and inputs
        r.set_initial_value(x_0,t)
        r.set_f_params(u_0).set_jac_params(u_0)

        # time step forward
        x_t = r.integrate(r.t + dt)

        # calculate the observed output (may not be the state)
        y_t = self.g(t, x_t, u_0)


        return x_t, y_t

if __name__ == '__main__':

    DS = DiggingSimulator2D()

    DS.set_s_func()
    
    x_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # bucket force pushing it forward and downwards
    u = np.array([10, -1.5])

    x_array = []
    x_array.append(x_t)

    for i in range(500):

        x_t, _ = DS.simulate_system(x_t, u)
        x_array.append(x_t)

    x_array = np.array(x_array)

    #####################################################
    
    fig, axs = plt.subplots(5, 1)   
    
    axs[0].plot(x_array[:,0],'k')
    axs[0].set_ylabel('x position')
    axs[0].grid(True)

    axs[1].plot(x_array[:,1],'k')
    axs[1].set_ylabel('z position')
    axs[1].grid(True)

    axs[2].plot(x_array[:,2],'k')
    axs[2].set_ylabel('x speed')
    axs[2].grid(True)

    axs[3].plot(x_array[:,3],'k')
    axs[3].set_ylabel('z speed')
    axs[3].grid(True)

    axs[4].plot(x_array[:,4],'k')
    axs[4].set_ylabel('bucket filling')
    axs[4].grid(True)

    fig.tight_layout()
    plt.show()