"""
Potential Field based path planner
author: Kosei Tanada (@kosei1515)
Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf
"""

import enum
from weakref import ref
import numpy as np
from matplotlib import pyplot as plt 

class PotentalFieldPlanning:
    def __init__(self, ):
        '''
        Initialization
        '''
        '''
        Set parameters
        '''
        self.__repulsive_gain= 200
        self.__atrractive_gain= 1.0
        
        '''
        Set robot model
        '''
        self.__robot_radius = 2.0
        
        '''
        Set the limitation of the map
        '''
        self.__minx = 0.0
        self.__miny = 0.0
        self.__maxx = 11.0
        self.__maxy = 11.0
        
        '''
        Set start, goal and grid size. (should be in the limitation)
        '''
        self.__start = np.array([1.0, 1.0])
        self.__goal = np.array([10.0, 10.0])
        self.__grid_resolution = 0.1
        
        
        '''
        Set obstacles
        '''
        self.__obstacle_list = np. array([[3.0, 2.0, 2.0],
                                          [7.0, 9.0, 2.0]])
        
        '''
        Get motion model.
        '''
        self.__motion_model = self.__get_motion_model()
        

    def __generate_potential_grid_map(self):
        '''
        Set a potential grid map
        Input: None
        Output:
        '''
        width = self.__get_grid_index(self.__maxx, self.__minx)       
        height = self.__get_grid_index(self.__maxy, self.__miny)
        self.__gridmap = np.zeros((width, height))
        
        for i in range(width):
            for j in range(height):
                potential = self.__calculate_atrractive_potential(i, j)
                potential += self.__calculate_repulsive_potential(i, j)
                self.__gridmap[i, j] = potential                
        
    def __calculate_atrractive_potential(self, grid_x, grid_y):
        '''
        Calcurate atrractive potential.
        Input: 
            grid_x: The index of the grid x.
            grid_y: The index of the grid y.
        Output: 
            potential: The calcurated atrractive potential
        '''
        x = self.__get_real_coordinate(grid_x, self.__minx)
        y = self.__get_real_coordinate(grid_y, self.__miny)
        
        potential = self.__atrractive_gain * np.sqrt((x - self.__goal[0]) ** 2 + (y - self.__goal[1]) ** 2)
        return potential
        
    def __calculate_repulsive_potential(self, grid_x, grid_y):
        '''
        Calculate repulsize potential.
        Input: 
            grid_x: The raw grid index.
            grid_y: The col grid index. 
        Output: 
            potential: The repulsive potential of the grid.
        '''
        x = self.__get_real_coordinate(grid_x, self.__minx)
        y = self.__get_real_coordinate(grid_y, self.__miny)
        
        # minid = -1
        min_distance = float("inf")
        
        for i in self.__obstacle_list:
            distance = np.sqrt((x - i[0]) ** 2 + (y - i[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
        
        if min_distance <= self.__robot_radius:
            if min_distance < 0.1:
                min_distance = 0.1
        
            # Calcurate repulsive cost
            potential = 0.5 * self.__repulsive_gain * (1.0 / min_distance - 1.0 / self.__robot_radius) **2
            return potential
        else:
            return 0.0
    
    def __get_grid_index(self, point, min_point):
        '''
        Get x (or y) index of the grid map
        Input: 
            point: The real coordinate
            min_point: The minimum point of the grid map.
        Output: 
            index: The grid index.
        '''
        index = int(abs(min_point - point) / self.__grid_resolution)  
        return index
    
    def __get_real_coordinate(self, grid_index, min_point):
        '''
        Get x (or y) coordinate from the grid index.
        Input: 
            grid_index: The index of the grid map
            min_point: The min x (or y) of the map
        Output:
            real_point: The coordinate at the grid
        '''
        real_point = grid_index * self.__grid_resolution + min_point
        return real_point
    
    def __get_motion_model(self):
        '''
        Get motion model of the robot
        Input: None
        Output: 
            motion: The motion model of the robot
        '''
        # dx, dy
        motion = [[1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1]]

        return motion
    
    def __is_goal(self, grid_x, grid_y):
        '''
        Check if the path is near the goal
        Input:
            grid_x: The raw index of the grid map
            grid_y: The col index of the grid map
        Output:
            True (if near goal)
            False (if not)
        '''
        goal_grid_index_x = self.__get_grid_index(self.__goal[0], self.__minx)
        goal_grid_index_y = self.__get_grid_index(self.__goal[1], self.__miny)
        
        if np.sqrt((grid_x- goal_grid_index_x) ** 2 + (grid_y- goal_grid_index_y) ** 2) < self.__grid_resolution:
            return True
        else:
            return False
    
    def potentilal_map_planning(self):
        '''
        Plan the path for the robot.
        Input: None
        Output: None
        '''
        # Plot start and goal
        start_x = self.__get_grid_index(self.__start[0], self.__minx)
        start_y = self.__get_grid_index(self.__start[1], self.__miny)
        plt.plot(start_x, start_y, "*k")
        goal_x = self.__get_grid_index(self.__goal[0], self.__minx)
        goal_y = self.__get_grid_index(self.__goal[1], self.__miny)
        plt.plot(goal_x, goal_y, "*m")
        # Get potential map
        self.__generate_potential_grid_map()
        
        # Initialize state
        raw_index_t = self.__get_grid_index(self.__start[0], self.__minx)
        col_index_t = self.__get_grid_index(self.__start[1], self.__miny)
        
        ref_x, ref_y = [raw_index_t], [col_index_t]
        
        while not self.__is_goal(raw_index_t, col_index_t):
            min_potential = float("inf")
            min_raw_index = -1
            min_col_index = -1
            
            for i, _ in enumerate(self.__motion_model):
                new_raw_index = int(raw_index_t + self.__motion_model[i][0])
                new_col_index = int(col_index_t + self.__motion_model[i][1])
                
                if new_raw_index > self.__gridmap.shape[1] or new_col_index > self.__gridmap.shape[0] or new_raw_index < 0 or new_col_index < 0:
                    potential = float("inf")
                else:
                    potential = self.__gridmap[new_raw_index][new_col_index]
                    
                if min_potential > potential:
                    min_potential = potential
                    min_raw_index = new_raw_index
                    min_col_index = new_col_index
                
            # print("check")  
            # print(min_raw_index)
            # print(min_col_index)
                
            path_x = self.__get_real_coordinate(min_raw_index, self.__minx)
            path_y = self.__get_real_coordinate(min_col_index, self.__miny)
            
            ref_x.append(path_x)
            ref_y.append(path_y)
            
            plt.plot(min_raw_index, min_col_index, ".r")
            plt.pause(0.01)
            
            raw_index_t = min_raw_index
            col_index_t = min_col_index
        
        
        print("Goal!!!!!")
        
        return ref_x, ref_y
        
if __name__ == '__main__':
    plt.grid(True)
    plt.axis("equal")
    
    pfp = PotentalFieldPlanning()
    ref_x, ref_y = pfp.potentilal_map_planning()   
    
    # plt.plot(ref_x, ref_y, c = "red")
    plt.show()     
            
                
                
                
        