"""
Extended Kalman Filter localization
author: Kosei Tanada (@kosei1515)
"""
from cmath import pi
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import patches
from json.encoder import INFINITY
import math

class ExtendedKalmanFilter(object):
    def __init__(self):
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
        self.__dT_s=0.01           # Discrete time[s]
        
        
        '''
        State initialization
        '''
        self.__refx=np.array([[0,
                              0,
                              0]]) # Set init state into x[0]
        self.__xhat=np.array([[0,
                              0,
                              0]])       
        
        '''
        Covariance matrix initilization
        '''
        self.__cov_matrix = np.diag([0.01, 0.01, np.deg2rad(30.0)]) ** 2
        
        
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
        
        '''
        observation matrix
        '''
        self.__C=np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
        
        # self.__sensor_range=3.0
        
        '''
        Parameters of the state space model
        '''
        self.__radius__m=3                  # The radius of circular motion [m]
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
        vx_std_dev=0.05      # The standard deviation x[m] of the observation error
        vy_std_dev=0.05      # The standard deviation y[m] of the observation error
        self.__R=np.diag([vx_std_dev, vy_std_dev])**2 # The covariance matrix of the observation error
        
        
        # self.__mu=np.array([0.0,0.0])
        # self.__cov=np.array([[self.__R[0][0],0.0],[0.0,self.__R[1][1]]])
        
    def __state_update(self,x_k):
        '''
        1step equation of state
        calculate x(k+1)
        Input:
            x_k: state x(k)
        Output:
            x_kplus1: state x(k+1)
        '''
        # print(x_k)
        
        yaw=x_k[:,2]
        
        u0 = self.__radius__m * self.__dT_s * self.__v__mps * np.cos(yaw)
        u1 = self.__radius__m * self.__dT_s * self.__v__mps * np.sin(yaw)
        u2= np.full_like(u0, self.__dT_s * self.__v__mps) 
        
        u=np.array([u0,
                    u1,
                    u2])
        # print(u0)
        
        x_kplus1=(self.__A @ x_k.T) + (self.__B @ u)
        return x_kplus1.T

    def __observation(self, state_x):
        '''
        Observation
        Detect direction between robot to the nearest landmark
            Input: None
        '''
        # observation error
        v = np.random.multivariate_normal([0.0, 0.0], self.__R, 1).T
        
        observed_z = (self.__C @ state_x.T) + v
        
        return observed_z.T

    def __state_prediction(self, px):
        '''
        Prediction based on the state equation
            Input: 
                px: The states of all particles
            Output
                estimted_px: The estimted states of all particles
        '''
        # input error
        w = np.random.multivariate_normal([0.0, 0.0, 0.0], self.__Q, 1)
        # calculate particles based on state equation
        estimated_px = self.__state_update(px) + w
        
        return estimated_px
    
    def __corvariance_matrix_prediction(self, pcov, jF):
        '''
        Prediction based on the state equation
            Input: 
                px: The states of all particles
            Output
                estimted_px: The estimted states of all particles
        '''
        # calculate particles based on state equation
        estimated_pcov = jF @ pcov @ (jF.T) + self.__Q
        
        return estimated_pcov
        
    def __update(self, prior_x, observed_z, prior_cov):
        # Update corvariance matrix Sigmma t|t
        posteriror_cov = np.linalg.inv((self.__C.T @ np.linalg.inv(self.__R)) @ self.__C + np.linalg.inv(prior_cov))
        
        # Update estimated hat{x t|t}
        jH = self.__jacobian_H()
        e_t = observed_z.T - jH @ prior_x.T

        # S = self.__C @ prior_cov @ self.__C.T + self.__R
        K_t = posteriror_cov @ self.__C.T @ np.linalg.inv(self.__R)
        
        updated_px = prior_x.T + K_t @ e_t

        return updated_px.T, posteriror_cov
    
    def __jacobian_F(self, xhat):
        '''Calculate Jacobian
        Input:
            xHat: Estimated state x
            
        Output:
            jF: The jacobian of the motion model jF(t)
        '''
        yaw = xhat[: ,2]
        a = -self.__dT_s * self.__v__mps * np.sin(yaw)
        b = self.__dT_s * self.__v__mps * np.cos(yaw)
        jF = np.array([[1.0, 0.0, a[0]  ],
                       [0.0, 1.0, b[0]  ],
                       [0.0, 0.0, 1.0]])
        return jF
        
    
    def __jacobian_H(self):
        '''Calcurate jacobian matrix jF(t)
        Input: 
            None
        Output:
            jH: The jacobian matrix jH(t) of the motion model
        '''
        jH = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
        return jH
    
    def main_ekf(self):
        '''
        main process
        Input: none
        Output:
            self.__refx:    The reference state x(t)
            x_estimated:    The estimated state xhat(t)
            self.__land_mark_list: The positions of all landmarks
            target_landmark:    The position of the nearst landmark
        '''
        
        # Update reference state
        self.__refx = self.__state_update(self.__refx)
        
        # Observation
        observed_z = self.__observation(self.__refx)
        # --------------Extended Kalman Filter----------------
        # Estimate prior state
        prior_xhat = self.__state_prediction(self.__xhat)
        
        # Estimate prior covariance matrix
        jF = self.__jacobian_F(self.__xhat)
        prior_cov_matrix = self.__corvariance_matrix_prediction(self.__cov_matrix, jF)
        
        # Calcurate the posterior probability
        # print(self.__xhat)
        
        self.__xhat, self.__cov_matrix = self.__update(prior_xhat, observed_z, prior_cov_matrix)      
         
        return self.__refx, prior_xhat, self.__xhat, observed_z, self.__cov_matrix, self.__land_mark_list
        

x_list = []
y_list=[]
est_x_list=[]
est_y_list=[]
observed_x_list=[]
observed_y_list=[]

def anim(i, ekf):
    plt.cla()
    
    ref_x, prior_x, x_hat, observed_x, cov, landmarks = ekf.main_ekf()
    
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
    # est_t = x_hat[0, 0:2]
    # x = np.array([est_t[0], target_landmark[0]])
    # y = np.array([est_t[1], target_landmark[1]])
    # ax1.plot(x, y, '--', c='green')
    # ax2.plot(x, y, '--', c='green')
    
    # Plot x of all particles
    # ax1.scatter(px[:,0], px[:,1], c='cyan', marker='o', alpha=0.5)
    # ax2.scatter(px[:,0], px[:,1], c='cyan', marker='o', alpha=0.5)
    # ax2.quiver(px[:,0], px[:,1], np.cos(px[:,2]), np.sin(px[:,2]), color='cyan', units='inches', scale=6.0, width=0.01,
    #            headwidth=0.0, headlength=0.0, headaxislength=0.0)
    
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
    est_x_list.append(x_hat[0,0])
    est_y_list.append(x_hat[0,1])
    
    c = patches.Circle(xy=(x_hat[0,0], x_hat[0,1]), radius=0.5, ec='#000000', fill=False)
    r1 = patches.Rectangle(xy=(x_hat[0,0] + 0.3*np.sin(x_hat[0,2]), x_hat[0,1] - 0.3*np.cos(x_hat[0,2])), width=0.25, height=0.1, angle = math.degrees(x_hat[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    r2 = patches.Rectangle(xy=(x_hat[0,0] - 0.2*np.sin(x_hat[0,2]), x_hat[0,1] + 0.2*np.cos(x_hat[0,2])), width=0.25, height=0.1, angle = math.degrees(x_hat[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    ax1.plot(est_x_list, est_y_list, c = 'blue', linewidth=1.0, linestyle='-', label='EKF')
    ax1.scatter(x_hat[:,0], x_hat[:,1], c='blue', marker='o', alpha=0.5)
    ax1.add_patch(c)
    ax1.add_patch(r1)
    ax1.add_patch(r2)
    
    c = patches.Circle(xy=(x_hat[0,0], x_hat[0,1]), radius=0.5, ec='#000000', fill=False)
    r1 = patches.Rectangle(xy=(x_hat[0,0] + 0.3*np.sin(x_hat[0,2]), x_hat[0,1] - 0.3*np.cos(x_hat[0,2])), width=0.25, height=0.1, angle = math.degrees(x_hat[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    r2 = patches.Rectangle(xy=(x_hat[0,0] - 0.2*np.sin(x_hat[0,2]), x_hat[0,1] + 0.2*np.cos(x_hat[0,2])), width=0.25, height=0.1, angle = math.degrees(x_hat[0,2]), linestyle="--", ec='#CCCCCC', fill=False)
    ax2.plot(est_x_list, est_y_list, c = 'blue', linewidth=1.0, linestyle='-', label='EKF')
    ax2.scatter(x_hat[:,0], x_hat[:,1], c='blue', marker='o', alpha=0.5)
    ax2.quiver(x_hat[:,0], x_hat[:,1], np.cos(x_hat[:,2]), np.sin(x_hat[:,2]), color='red', units='inches', scale=6.0,
               width=0.01, headwidth=0.0, headlength=0.0, headaxislength=0.0)
    ax2.add_patch(c)
    ax2.add_patch(r1)
    ax2.add_patch(r2)
    
    # Plot observed x trajectory
    observed_x_list.append(observed_x[0,0])
    observed_y_list.append(observed_x[0,1])
    
    ax1.plot(observed_x_list, observed_y_list, c = 'green', linewidth=1.0, linestyle='-', label='Observed')
    ax1.scatter(observed_x[:,0], observed_x[:,1], c='green', marker='o', alpha=0.5)    
    
    ax2.plot(observed_x_list, observed_y_list, c = 'green', linewidth=1.0, linestyle='-', label='Observed')
    ax2.scatter(observed_x[:,0], observed_x[:,1], c='green', marker='o', alpha=0.5)
    # ax2.quiver(observed_x[:,0], observed_x[:,1], np.cos(observed_x[:,2]), np.sin(observed_x[:,2]), color='red', units='inches', scale=6.0,
    #            width=0.01, headwidth=0.0, headlength=0.0, headaxislength=0.0)
    
    # Plot label
    # txt = 'Maximuim Likelihood Estimate:\n[Index]:{0}\n[Weight]:{1:.3f}'.format(w_idx, w_val)
    # ax2.annotate(txt, xy=(x_hat[0, 0], x_hat[1, 0]), xycoords='data',
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
    ekf_filter = ExtendedKalmanFilter()
    fig = plt.figure(figsize=(18, 9))
    frame = int(3600)
        
    
    video = animation.FuncAnimation(fig, anim, frames=frame, fargs=(ekf_filter,), blit=False, interval=10, repeat=False)
    
    plt.show()
    
    
    
    