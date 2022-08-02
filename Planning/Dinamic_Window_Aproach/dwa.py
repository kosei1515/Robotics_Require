from json.encoder import INFINITY
from random import sample
from tokenize import Triple
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import math


class DinamicWindowAproach:
    def __init__(self):
        '''
        Set the discrete time
        '''
        self.__dT = 0.1
        self.__goal_cost_weight = 2.0
        self.__velocity_cost_weight = 3.0
        self.__obstacle_cost_weight = 3.0
        self.__obstacle_margin = 0.2
        
        '''
        Set start and goal
        '''
        self.__start = np.array([0.0,0.0])
        self.__goal = np.array([10.0, 10.0])
        
        '''
        Set obstacles
        '''
        self.__set_obstacles()
        
        '''
        Set parameters of the robot
        '''
        self.__robot_radius = 0.5
        self.__detection_range = 5.0
        '''
        Set parameters of the motion model 
        '''
        self.__max_velocity = 0.5           # [m/s]
        self.__min_velocity = 0.0           # [m/s]
        self.__max_angular_velocity = math.pi      # [rad/s]
        self.__min_angular_velocity = -math.pi     # [rad/s] 

        self.__delta_velocity = 0.1    # [m/s]
        self.__delta_angular_velocity = 0.1   #[rad/s]
        self.__predict_step = 30
        self.__predict_time = 3.0           # [s]
        
        '''
        Set the motion model of the robot
        '''
        '''
        State transition matrix A
        '''
        self.__A = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        
        '''
        Control matrix B
        '''
        self.__B = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        
        
        
    class PathPoint:
        def __init__(self, x, y, theta, vel, angular_vel):
            self.x = x
            self.y = y
            self.theta = theta
            self.vel= vel
            self.angular_vel = angular_vel
    
    def __set_obstacles(self):
        '''
        Set obstacles
        Input: None
        Output: None
        '''
        self.__obstacle_list = np.array([[1.0, 2.0, 0.1],
                                    [5.0, 6.0, 0.1],
                                    [1.0, 0.0, 0.1],
                                    [9.0, 9.0, 0.5]]) # [x, y, radius]
        self.__detected_objects = []
        
    def __set_detected_obstacles(self, current_state):
        '''
        Set detected object. (All objects in the detection range are considered as detected.)
        Input:
            current_state: The current state of the robot
        Output: 
            None
        '''
        self.__detected_objects = []
        for i in self.__obstacle_list:
            distance = np.sqrt((current_state[0,0] - i[0]) ** 2 + (current_state[0,1] - i[1]) ** 2)
            if distance <= self.__detection_range - i[2]:
                self.__detected_objects.append(i)   
        
    
    def __is_collision_with_obstacles(self, state):
        '''
        Check the state is in collision
        Input: 
            state: the predicted_state
        Output:
            True or False: If in collision, return True.
        '''
        for i in self.__detected_objects:
            distance = np.sqrt((state[0, 0] - i[0]) ** 2 + (state[0, 1] - i[1]) ** 2)
            if distance < i[2] + self.__robot_radius + self.__obstacle_margin:
                return True
        
        return False 
        
            
    def __predict_robot_state(self, state_t, sample_vel, sample_angular_vel):
        '''
        Predict the next state using the Kinematic model of the robot.
        Input: 
            state_t: the previous state of the robot
            u_t: The input to the system
        Output: 
            hat_state: the predicted state
        '''
        hat_x_list = []
        hat_y_list = []
        hat_theta_list = []
        
        # Evaluate a veloity of the robot
        evaluation_cost = self.__evaluate_robot_velocity(sample_vel)
        evaluation_cost = 0
        for i in range(self.__predict_step):
            u_t = np.array([[sample_vel * self.__dT * np.cos(state_t[0, 2]), sample_vel * self.__dT * np.sin(state_t[0, 2]), sample_angular_vel * self.__dT]])
            hat_state = self.__update_state(state_t, u_t)
            # print(hat_state)
            state_t = hat_state
            hat_x_list.append(state_t[0,0])
            hat_y_list.append(state_t[0,1])
            hat_theta_list.append(state_t[0,2])
            
            # Check if the predicted state is not in collision
            if self.__is_collision_with_obstacles(state_t):
                # If in collision, return None
                return [], [], [], -1   

        # Evaluate the distance to the obstacles
        evaluation_cost += self.__evaluate_distance_to_obstacles(state_t)
        # Evaluate the distance to the goal
        evaluation_cost += self.__evaluate_distance_to_goal(hat_x_list, hat_y_list, hat_theta_list)
        return hat_x_list, hat_y_list, hat_theta_list, evaluation_cost
        
    def __calc_velocity_range(self, current_velocuity):
        '''
        Calcurate the range of the velocity
        Input: 
            current_state: the current state of the robot
        Output:
            min_v: the minimum velocity
            max_v: the maximum velocity
        '''
        velocity_range = self.__predict_time * self.__max_velocity
        min_v =  current_velocuity - velocity_range
        max_v =  current_velocuity + velocity_range
        
        if min_v < self.__min_velocity:
            min_v = self.__min_velocity
        if max_v > self.__max_velocity:
            max_v = self.__max_velocity
        
        return min_v, max_v
        
    def __calc_angular_range(self, current_angular_velocity):
        '''
        Calcurate the range of the angular velocity
        Input: 
            current_state: the current state of the robot
        Output:
            min_w: the minimum angular velocity
            max_w: the maximum angular velocity
        '''
        angular_velocity_range = self.__predict_time * self.__max_angular_velocity
        min_w = current_angular_velocity - angular_velocity_range
        max_w = current_angular_velocity + angular_velocity_range
        
        if min_w < self.__min_angular_velocity:
            min_w = self.__min_angular_velocity
        if max_w > self.__max_angular_velocity:
            max_w = self.__max_angular_velocity
        
        return min_w, max_w
    
    def __generate_and_evaluate_paths(self, state, current_velocity, current_angular_velocity):
        '''
        Generate 1 step path points 
        Input:
            state: the previous predicted state of the robot
            path_points_trajectory: the all path trajectory from the current states to the previous predicted states
        '''
        # Get the range of the (angular) velocity
        min_v, max_v = self.__calc_velocity_range(current_velocity)
        min_w, max_w = self.__calc_angular_range(current_angular_velocity)
        
        next_paths = []
        
        angular_sample_list = np.arange(min_w, max_w, self.__delta_angular_velocity)
        velocity_sample_list = np.arange(min_v, max_v, self.__delta_velocity)
        
        minimum_path = []
        minimum_cost = INFINITY
        
        # Generate path points using all (angular) velocity samples 
        for sample_angular_vel in angular_sample_list:
            for sample_vel in velocity_sample_list:
                # Generate a path and evaluate it
                hat_x_list, hat_y_list, hat_theta_list, eval_value = self.__predict_robot_state(state, sample_vel, sample_angular_vel)
                
                # print(hat_theta_list)
                
                # If the predicted state is in collision, the states does not be added in the list
                if len(hat_x_list) != 0:
                    sample_path = self.PathPoint(hat_x_list, hat_y_list, hat_theta_list, sample_vel, sample_angular_vel)
                    next_paths.append(sample_path)
                    
                    if minimum_cost > eval_value:
                        minimum_cost = eval_value
                        minimum_path = sample_path
                # else:
                #     # print("Collision")
        
        if len(next_paths) == 0:
            print("Can't find any ways")
        
        return next_paths, minimum_path
        
    def __evaluate_distance_to_goal(self, x_list, y_list, theta_list):
        '''
        Evaluate the distance to the goal. The better the evaluation is, the smaller the value becomes.
        Input:
            state: The velocity of the robot
        Output: 
            v_cost: The evaluation value.
        '''
        distance = np.sqrt((x_list[-1] - self.__goal[0]) ** 2 + (y_list[-1] - self.__goal[1]) ** 2)
        g_cost = distance
        
        # last_x = x_list[-1]
        # last_y = y_list[-1]
        # last_theta = theta_list[-1]

        # angle_to_goal = math.atan2(self.__goal[1] - last_y, self.__goal[0] - last_x)

        # score_angle = angle_to_goal - last_theta
        
        # g_cost += abs(self.__angle_range_corrector(score_angle))
        return self.__goal_cost_weight * g_cost
    
    def __angle_range_corrector(self, angle):
    
        if angle > math.pi:
            while angle > math.pi:
                angle -=  2 * math.pi
        elif angle < -math.pi:
            while angle < -math.pi:
                angle += 2 * math.pi

        return angle

        
    def __evaluate_robot_velocity(self, velocity):
        '''
        Evaluate the velocity of the robot. The better the evaluation is, the smaller the value becomes.
        Input:
            state: The velocity of the robot
        Output: 
            v_cost: The evaluation value.
        '''
        if velocity < 0.001:
            velocity = 0.001
        v_cost = 1/abs(velocity)
        return self.__velocity_cost_weight * v_cost
        
        
    def __evaluate_distance_to_obstacles(self, state):
        '''
        Evaluate distance to the nearest obstacle. The better the evaluation is, the smaller the value becomes.
        Input:
            state: The predicted state
        Output: 
            c_cost: The evaluation value.
        '''
        if len(self.__detected_objects) != 0:
            for i in self.__detected_objects:
                distance = np.sqrt((state[0, 0] - i[0]) ** 2 + (state[0, 1] - i[1]) ** 2)
                min_distance = distance
            c_cost = 1/min_distance
        else:
            c_cost = 0
        
        return self.__obstacle_cost_weight * c_cost
    
    def __update_state(self, previous_state, u_t):
        '''
        Calculate the state euation of the sysytem
        Input:
            previous_state: The previous state
        Output:
            next_state: The next state
        '''
        next_state = self.__A @ previous_state.T + self.__B @ u_t.T
        
        return next_state.T
    
    def __is_goal(self, state):
        distance = np.sqrt((state[0,0] - self.__goal[0]) ** 2 + (state[0,1] - self.__goal[1]) ** 2)
        if distance <=0.1:
            return True
        else:
            return False
    
    def main(self):
        '''
        Main function of DWA.
        Input: None
        Output: None
        '''
        ax = plt.subplot()
        
        # Set Initial state
        init_state = np.array([[self.__start[0], self.__start[1], 0.0]])
        state_t = init_state
        u_t = np.array([[0.0, 0.0, 0.0]])
        selected_path = self.PathPoint(self.__start[0], self.__start[1], 0.0, 1.0, 0.0)
        
        # For visuaization
        count = 0
        state_trajectory_x = []
        state_trajectory_y = []
        state_trajectory_x.append(state_t[0, 0])
        state_trajectory_y.append(state_t[0, 1])
        
        # Start planning
        while 1:
            # Detect objects
            self.__set_detected_obstacles(state_t)
            
            # Plot obstacles
            for i in self.__obstacle_list:
                c = patches.Circle(xy=(i[0], i[1]), radius=i[2], ec='#000000', fill=False)
                ax.add_patch(c)
            
            # Plot detected obstacles
            for i in self.__detected_objects:
                c = patches.Circle(xy=(i[0], i[1]), radius=i[2], ec='blue', fill=False)
                ax.add_patch(c)
            
            # Plot the robot
            c = patches.Circle(xy=(state_t[0,0], state_t[0,1]), radius = self.__robot_radius, ec='#000000', fill = True, alpha = 0.5, fc = "gray")
            r1 = patches.Rectangle(xy=(state_t[0,0] + 0.4* self.__robot_radius * np.sin(state_t[0,2]), state_t[0,1] - 0.4 * self.__robot_radius * np.cos(state_t[0,2])), width=0.25, height=0.1, angle = math.degrees(state_t[0,2]), linestyle="-", ec='#000000', fill=False)
            r2 = patches.Rectangle(xy=(state_t[0,0] - 0.3 * self.__robot_radius * np.sin(state_t[0,2]), state_t[0,1] + 0.3 * self.__robot_radius * np.cos(state_t[0,2])), width=0.25, height=0.1, angle = math.degrees(state_t[0,2]), linestyle="-", ec='#000000', fill=False)
            ax.add_patch(c)
            ax.add_patch(r1)
            ax.add_patch(r2)
            
            # Plot detection range of the robot
            c = patches.Circle(xy=(state_t[0,0], state_t[0,1]), radius = self.__detection_range, ec='lawngreen', fill = False, alpha = 0.5, ls = "--")
            ax.add_patch(c)
            
            # Predict the trajectories based on DWA
            next_paths, min_path = self.__generate_and_evaluate_paths(state_t, selected_path.vel, selected_path.angular_vel)
         
            selected_path = min_path
            
            # Update the state of the robot
            u_t = np.array([[selected_path.vel * self.__dT * np.cos(state_t[0, 2]), selected_path.vel * self.__dT * np.sin(state_t[0, 2]), selected_path.angular_vel * self.__dT]])
            state_t = self.__update_state(state_t, u_t)
            
            
            # Plot minimun costing path
            ax.plot(min_path.x, min_path.y, "-", c = "cyan")
            # Plot trajectory of the robot
            state_trajectory_x.append(state_t[0, 0])
            state_trajectory_y.append(state_t[0, 1])
            ax.plot(state_trajectory_x, state_trajectory_y, c = "red")
            
            ax.grid(True)
            ax.axis("equal")
            if count % 1 ==0:
                plt.pause(0.01)
            count += 1
            if self.__is_goal(state_t):
                break
            
            ax.cla()
        
        print("Goal!!!")
            

if __name__ == '__main__':
    dwa_planning = DinamicWindowAproach()
    
    dwa_planning.main()

    
    plt.show() 
                
            
        
    
    
        
        
           
                
        