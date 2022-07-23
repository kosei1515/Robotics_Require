"""
Particle Filter localization
author: Kosei Tanada (@kosei1515)
"""

from asyncio.windows_events import INFINITE
from cmath import cos, sqrt
from json.encoder import INFINITY
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from scipy import stats

class ParticleFilter:
    def __init__(self,x_init,y_init,theta_init):
        '''
        Initialization
            input:
                x_init:  x init [m]
                y_init:  y init [m]
                theta_init:     theta init [m]
            parameters:
                __particle_num: the number of particle
                __w_val: the valiance of input u error
                __v_val: the valiance of observation v error        
        '''
        
        '''
        Parameters
        '''
        self.__particle_num=500    # The number of particles
        self.__dT_s=0.01           # Discrete time[s]
        
        
        '''
        State initialization
        '''
        self.__refx=np.array([[x_init,
                              y_init,
                              theta_init]])                      # Set init state into x[0]
        self.__px=np.full((self.__particle_num, 3), self.__refx[0, :])     # Set all init particles to x[0]
        self.__init_pw=np.full(self.__particle_num, 1/self.__particle_num)
        
        '''
        State transition matrix
        '''
        self.__A=np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        '''
        Control matrix
        '''
        self.__B=np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        
        # self.__sensor_range=3.0
        
        '''
        Parameters of the state space model
        '''
        self.__radius__m=5                  # The radius of circular motion [m]
        self.__omega_radps=np.deg2rad(5)    # The angular velocity [rad/s]
        self.__v__mps=self.__radius__m*self.__omega_radps   # The velocity of the robot [m/s]
        
        
        '''
        The landmark list
        '''
        self.__land_mark_list=np.array([[1.0,2.0], 
                                       [6.0,9.0], 
                                       [5.0,5.0]])
        
        '''
        Parameters of the input error matrix
        '''
        wx_std_dev=0.003      # The standard deviation x[m] of the input error
        wy_std_dev=0.003      # The standard deviation y[m] of the input error
        wyaw_std_dev=0.01    # The standard deviation yaw[rad] of the input error
        self.__Q=np.diag([wx_std_dev, wy_std_dev, wyaw_std_dev])**2 # The covariance matrix of the input error
        '''
        Parameters of the obsevation error matrix
        '''
        vx_std_dev=0.01      # The standard deviation x[m] of the observation error
        vy_std_dev=0.01      # The standard deviation y[m] of the observation error
        self.__R=np.diag([vx_std_dev, vy_std_dev])**2 # The covariance matrix of the observation error
        
        
        self.__mu=np.array([0.0,0.0])
        self.__cov=np.array([[self.__R[0][0],0.0],[0.0,self.__R[1][1]]])
        
    def __state_equation(self,x_k):
        '''
        1step equation of state
        calculate x(k+1)
        Input:
            x_k: state x(k)
        Output:
            x_kplus1: state x(k+1)
        '''
        yaw=x_k[:,2]
        
        u0 = self.__radius__m * self.__dT_s * self.__v__mps * np.cos(yaw)
        u1 = self.__radius__m * self.__dT_s * self.__v__mps * np.sin(yaw)
        u2= np.full_like(u0, self.__dT_s * self.__v__mps) 
        
        u=np.array([u0,
                    u1,
                    u2])
        
        x_kplus1=(self.__A @ x_k.T) + (self.__B @ u)
        return x_kplus1.T
        
    
    def __prediction(self, px):
        '''
        Prediction based on the state equation
            Input: 
                px: The states of all particles
            Output
                estimted_px: The estimted states of all particles
        '''
        # input error
        w = np.random.multivariate_normal([0.0, 0.0, 0.0], self.__Q, self.__particle_num)
        # calculate particles based on state equation
        estimated_px = self.__state_equation(px)+w
        
        return estimated_px
        
    def __observation(self):
        '''
        Observation
        Detect direction between robot to the nearest landmark
            Input: None
        '''
        # observation error
        v = np.random.multivariate_normal([0.0, 0.0], self.__R)
        
        observed_z=np.zeros(2)
        min=INFINITY
        
        target_landmark=np.zeros(2)
        
        # Search the nearest landmark
        for i in self.__land_mark_list:
            diff_x=i+v-self.__refx.T[:,0:1]
            if np.sqrt((diff_x[0,0])**2+(diff_x[0,1])**2)<min:
                # Set observed distance
                min=np.sqrt((diff_x[0,0])**2+(diff_x[0,1])**2)
                observed_z=diff_x
                target_landmark=i
        
        return observed_z, target_landmark
        
    def __likelihoodCalculation(self, particles_t, z_observed, target):
        '''
        Calculate likelihoods of all particles
        Input: 
            particles_t: The states of all particles
            z_observerd: The observation data
            target: The corrdinates of targetted landmark
        Output:
            rho: The likelihoods of all particles
        '''
        rho = np.zeros(self.__particle_num)
        pn = np.zeros(self.__particle_num)
        
        for i in range(self.__particle_num):
            diff_x=target-particles_t[i, 0:1]
            error=z_observed-diff_x
            error=np.array([error[0,0],error[0,1]])
            
            pn[i] = stats.multivariate_normal.pdf(error, self.__mu, self.__cov)
        rho=self.__init_pw*pn
        
        # Normalization
        self.__normalization(rho)
        
        return rho
    
    def __resampling(self, px, likelihood):

        px_new=np.zeros((self.__particle_num, 3))
        argsorted_likelihood=np.argsort(likelihood)[::-1]
        
        j=int(0)
        sum=likelihood[argsorted_likelihood[j]]
        for i in range(self.__particle_num):
            px_new[i,:]=px[argsorted_likelihood[j],:]
            
            if i>=sum:
                j=j+1
                sum=sum+likelihood[argsorted_likelihood[j]]
        
        return px_new 
        
    def __normalization(self, likelihood):
        sum = np.sum(likelihood)
        nor_likelihood = likelihood / sum
        nor_likelihood[np.isnan(nor_likelihood)] = 0
        
    def particlefilter_main(self):
        '''
        main process
        Input: none
        Output:
            self.__refx:    The reference state
            x_estimated:    The estimated state
            self.__px:      The states of all particles
            self.__land_mark_list: The positions of all landmarks
            target_landmark:    The position of the nearst landmark
        '''
        # Renew state
        self.__refx = self.__state_equation(self.__refx)
        
        # --------------Particle Filter----------------
        # Prediction
        self.__px=self.__prediction(self.__px)
        
        # Observation
        observed_z, target_landmark = self.__observation()
        
        # Calcurate likelihoods 
        likelihoods = self.__likelihoodCalculation(self.__px, observed_z, target_landmark)
        
        max_val = np.max(likelihoods)
        max_idx = np.argmax(likelihoods)
        x_estimated = np.array([self.__px[max_idx, :]])
        
        # Resampling
        self.__px = self.__resampling(self.__px, likelihoods)
        
        return self.__refx, x_estimated, self.__px, self.__land_mark_list, target_landmark
        
x_list = []
y_list=[]
est_x_list=[]
est_y_list=[]
x_est_trajectory = []

def anim(i, pf):
    plt.cla()
    
    ref_x, x_est, px, landmarks, target_landmark = pf.particlefilter_main()
    
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    # Plot landmarks
    h = 0.25
    w = 0.25
    for i in range(landmarks.shape[0]):
        r = patches.Rectangle(xy=(landmarks[i, 0] - w/2, landmarks[i, 1]- h/2), width = w, height = h, fc='r',ec='r')
        ax1.add_patch(r)
        r = patches.Rectangle(xy=(landmarks[i, 0] - w/2, landmarks[i, 1]- h/2), width=0.25, height=0.25, fc='r',ec='r')
        ax2.add_patch(r)
        
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], s=100, c="yellow", marker="*", alpha=0.5, edgecolors="orange")
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], s=100, c="yellow", marker="*", alpha=0.5, edgecolors="orange")
    
    # Plot line
    est_t = x_est[0, 0:2]
    x = np.array([est_t[0], target_landmark[0]])
    y = np.array([est_t[1], target_landmark[1]])
    ax1.plot(x, y, '--', c='green')
    ax2.plot(x, y, '--', c='green')
    
    # Plot x of all particles
    ax1.scatter(px[:,0], px[:,1], c='cyan', marker='o', alpha=0.5)
    ax2.scatter(px[:,0], px[:,1], c='cyan', marker='o', alpha=0.5)
    ax2.quiver(px[:,0], px[:,1], np.cos(px[:,2]), np.sin(px[:,2]), color='cyan', units='inches', scale=6.0, width=0.01,
               headwidth=0.0, headlength=0.0, headaxislength=0.0)
    
    # Plot x trajectory
    x_list.append(ref_x[0,0])
    y_list.append(ref_x[0,1])
    
    c = patches.Circle(xy=(ref_x[0,0], ref_x[0,1]), radius=0.5, ec='#000000', fill=False)
    r1 = patches.Rectangle(xy=(ref_x[0,0] + 0.3*np.sin(ref_x[0,2]), ref_x[0,1] - 0.3*np.cos(ref_x[0,2])), width=0.25, height=0.1, angle = math.degrees(ref_x[0,2]), ec='#000000', fill=False)
    r2 = patches.Rectangle(xy=(ref_x[0,0] - 0.2*np.sin(ref_x[0,2]), ref_x[0,1] + 0.2*np.cos(ref_x[0,2])), width=0.25, height=0.1, angle = math.degrees(ref_x[0,2]), ec='#000000', fill=False)
    ax1.plot(x_list, y_list, c = 'red', linewidth=1.0, linestyle='-', label='Ground Truth')
    ax1.scatter(ref_x[:,0], ref_x[:,1], c='red', marker='o', alpha=0.5)
    ax1.add_patch(c)
    ax1.add_patch(r1)
    ax1.add_patch(r2)
    
    c = patches.Circle(xy=(ref_x[0,0], ref_x[0,1]), radius=0.5, ec='#000000', fill=False)
    r1 = patches.Rectangle(xy=(ref_x[0,0] + 0.3*np.sin(ref_x[0,2]), ref_x[0,1] - 0.3*np.cos(ref_x[0,2])), width=0.25, height=0.1, angle = math.degrees(ref_x[0,2]), ec='#000000', fill=False)
    r2 = patches.Rectangle(xy=(ref_x[0,0] - 0.2*np.sin(ref_x[0,2]), ref_x[0,1] + 0.2*np.cos(ref_x[0,2])), width=0.25, height=0.1, angle = math.degrees(ref_x[0,2]), ec='#000000', fill=False)
    ax2.plot(x_list, y_list, c = 'red', linewidth=1.0, linestyle='-', label='Ground Truth')
    ax2.scatter(ref_x[:,0], ref_x[:,1], c='red', marker='o', alpha=0.5)
    ax2.add_patch(c)
    ax2.add_patch(r1)
    ax2.add_patch(r2)
    ax2.quiver(ref_x[:,0], ref_x[:,1], np.cos(ref_x[:,2]), np.sin(ref_x[:,2]), color='red', units='inches', scale=6.0,
               width=0.01, headwidth=0.0, headlength=0.0, headaxislength=0.0)
    
    # Plot estimated x trajectory
    est_x_list.append(x_est[0,0])
    est_y_list.append(x_est[0,1])
    
    c = patches.Circle(xy=(x_est[0,0], x_est[0,1]), radius=0.5, ec='#000000', fill=False)
    r1 = patches.Rectangle(xy=(x_est[0,0] + 0.3*np.sin(x_est[0,2]), x_est[0,1] - 0.3*np.cos(x_est[0,2])), width=0.25, height=0.1, angle = math.degrees(x_est[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    r2 = patches.Rectangle(xy=(x_est[0,0] - 0.2*np.sin(x_est[0,2]), x_est[0,1] + 0.2*np.cos(x_est[0,2])), width=0.25, height=0.1, angle = math.degrees(x_est[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    ax1.plot(est_x_list, est_y_list, c = 'blue', linewidth=1.0, linestyle='-', label='Ground Truth')
    ax1.scatter(x_est[:,0], x_est[:,1], c='blue', marker='o', alpha=0.5)
    ax1.add_patch(c)
    ax1.add_patch(r1)
    ax1.add_patch(r2)
    
    c = patches.Circle(xy=(x_est[0,0], x_est[0,1]), radius=0.5, ec='#000000', fill=False)
    r1 = patches.Rectangle(xy=(x_est[0,0] + 0.3*np.sin(x_est[0,2]), x_est[0,1] - 0.3*np.cos(x_est[0,2])), width=0.25, height=0.1, angle = math.degrees(x_est[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    r2 = patches.Rectangle(xy=(x_est[0,0] - 0.2*np.sin(x_est[0,2]), x_est[0,1] + 0.2*np.cos(x_est[0,2])), width=0.25, height=0.1, angle = math.degrees(x_est[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    ax2.plot(est_x_list, est_y_list, c = 'blue', linewidth=1.0, linestyle='-', label='Ground Truth')
    ax2.scatter(x_est[:,0], x_est[:,1], c='blue', marker='o', alpha=0.5)
    ax2.quiver(x_est[:,0], x_est[:,1], np.cos(x_est[:,2]), np.sin(x_est[:,2]), color='red', units='inches', scale=6.0,
               width=0.01, headwidth=0.0, headlength=0.0, headaxislength=0.0)
    ax2.add_patch(c)
    ax2.add_patch(r1)
    ax2.add_patch(r2)
    
    # Plot label
    # txt = 'Maximuim Likelihood Estimate:\n[Index]:{0}\n[Weight]:{1:.3f}'.format(w_idx, w_val)
    # ax2.annotate(txt, xy=(x_est[0, 0], x_est[1, 0]), xycoords='data',
    #             xytext=(0.55, 0.9), textcoords='axes fraction',
    #             bbox=dict(boxstyle='round,pad=0.5', fc=(1.0, 0.7, 0.7)),
    #             arrowprops=dict(arrowstyle="->", color='black',
    #                             connectionstyle='arc3,rad=0'),
    #             )

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Localization by PF')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid()
    ax1.legend(fontsize=10)

    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Zoom')
    ax2.set_xlim(ref_x[0][0] - 1.0, ref_x[0][0] + 1.0)
    ax2.set_ylim(ref_x[0][1] - 1.0, ref_x[0][1] + 1.0)
    ax2.grid()
    ax2.legend(fontsize=10)    

if __name__=='__main__':   
    p_filter2=ParticleFilter(1.0, 0.0, 0.0)
    fig = plt.figure(figsize=(18, 9))
    frame = int(360)
    
    video = animation.FuncAnimation(fig, anim, frames=frame, fargs=(p_filter2,), blit=False, interval=10, repeat=False)
    
    plt.show()
    
    
    

