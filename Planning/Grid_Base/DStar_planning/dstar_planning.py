from cmath import isnan
from json.encoder import INFINITY
from math import nan
from matplotlib import pyplot as plt
import time
import numpy as np
import math

class DStar():
    def __init__(self, start_x, start_y, goal_x, goal_y):
        '''
        Initialize Parameters.
        '''
        self.__weight_h = 1.0           # The weight of the heuristic cost
        self.__grid_resolution = 1.0    # The grid resolusion [m]
        self.__width = 50.0      #[m]
        self.__height = 50.0     #[m]
        self.__minx = 0          #[m]
        self.__miny = 0          #[m]
        
        '''
        Set the motion model of the robot
        '''
        self.__motion = self.__get_motion_model()
        '''
        Set start and goal
        '''
        self.__start_node = self.Node(self.__calc_raw_or_col_index(start_x, self.__minx), self.__calc_raw_or_col_index(start_y, self.__miny), 0.0, -1)
        self.__goal_node = self.Node(self.__calc_raw_or_col_index(goal_x, self.__minx), self.__calc_raw_or_col_index(goal_y, self.__miny), 0.0, -1)
        
        '''
        Set obstacles
        '''
        raw_grid_num = round((self.__width - self.__minx) / self.__grid_resolution)
        col_grid_num = round((self.__height - self.__miny) / self.__grid_resolution)
        max_index = round(col_grid_num * raw_grid_num)
        self.__cost_map = np.zeros(max_index)
        for i in range(col_grid_num):
            index = i * raw_grid_num
            self.__cost_map[index]=INFINITY
            index2 = (i + 1) * raw_grid_num - 1
            self.__cost_map[index2]=INFINITY
            
        for i in range(raw_grid_num):
            index = i
            self.__cost_map[index]=INFINITY
            index2 = round((col_grid_num - 1) * raw_grid_num + i)
            self.__cost_map[index2]=INFINITY
            
        for i in range(round(2 * raw_grid_num / 3)):
            index = i + (round(col_grid_num / 4) - 1) * raw_grid_num
            index2 = raw_grid_num - 1 - i + int(col_grid_num / 2 - 1) * raw_grid_num
            index3 = int(i + int(col_grid_num / 4 * 3 - 1) * raw_grid_num)
            self.__cost_map[index] = INFINITY
            self.__cost_map[index2] = INFINITY
            self.__cost_map[index3] = INFINITY
        
        '''
        Initialize the open list and the close list
        '''
        self.__open_list = dict()
        self.__close_list = dict()
        
        # Put the goal node into the open list
        self.__open_list[self.__calc_grid_index(self.__goal_node)] = self.__goal_node  
        
        #---------------------------
        self.children_callback_flag = False
        self.__node_list = []               # The node list of the last path    
        
    class Node:
        def __init__(self, x, y, cost, child_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.child_index = child_index
         
    def __calc_raw_or_col_index(self, position, min_pos):
        '''
        Calculate the raw or col index
        Input:
            position: x or y coorinate
            min_pos: min x or y coordinate
        Output:
            index
        '''
        return round((position - min_pos) / self.__grid_resolution)
    
    def __calc_grid_index(self, node):
        '''
        Calcurate the grid index
        Input:
            node: Node info(class Node)
        Output:
            index
        '''
        return (node.y - self.__miny) * self.__width + (node.x - self.__minx)
    
    def __calc_real_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.__grid_resolution + min_position
        return pos
    
    def __is_start(self, select_node_index):
        '''
        Check whether the selected node is the goal node or not. 
        Input: 
            select_node_index: the node index selected from the open list
        '''
        if select_node_index == self.__calc_grid_index(self.__start_node):
            return True
        else:
            return False
        
    def __callback_children(self, selected_node):
        self.__node_list.append(selected_node)
        
        # Get the child node's index of the goal node
        index = selected_node.child_index
        goal_index = self.__calc_grid_index(self.__goal_node)
        
        while index != goal_index: 
            # Add node
            node = self.__close_list[index]
            self.__node_list.append(node)
            # Get the next child index
            index = node.child_index
        
        # Add the last child node (start node)
        node = self.__close_list[index]   
        self.__node_list.append(node)
        self.children_callback_flag = True

    @staticmethod
    def __get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion
    
    def __DStar_plannig(self):        
        # If the open list is empty, the seach is failed.
        if len(self.__open_list) == 0:
            print("A-Star search failed....")
            exit()
                    
        # Get the minimum cost node from the Open List
        min_cost_node_index = min(self.__open_list, key = lambda x : self.__open_list[x].cost)
        selected_node = self.__open_list[min_cost_node_index]

        # Check if the selected node's index is same with the goal node 
        if self.__is_start(min_cost_node_index):
            # Call back all children until the start node
            self.__callback_children(selected_node)
            return selected_node, self.__node_list

        # Put the selected node into the closed list
        self.__close_list[min_cost_node_index] = selected_node
        del self.__open_list[min_cost_node_index]

        for i, _ in enumerate(self.__motion):
            searched_node = self.Node(selected_node.x + self.__motion[i][0],
                                 selected_node.y + self.__motion[i][1],
                                 selected_node.cost + self.__motion[i][2], min_cost_node_index)
            searched_node_index = round(self.__calc_grid_index(searched_node))
            # Check if the searching node is searchable.
            if self.__cost_map[searched_node_index] == INFINITY:
                continue
            
            # Check if the searching node is in the close list.
            if searched_node_index in self.__close_list:
                continue
            
            # Check if the searching node is in the open list.
            if not searched_node_index in self.__open_list:
                self.__open_list[searched_node_index] = searched_node
            else:
                if searched_node.cost < self.__open_list[searched_node_index].cost:
                    self.__open_list[searched_node_index].cost = searched_node.cost
                    self.__open_list[searched_node_index].child_index = min_cost_node_index
        
        return selected_node, self.__node_list, 

    def main(self):
        plt.grid(True)
        plt.axis("equal")
        ax1 = plt.subplot()
        # Plot the start node
        ax1.scatter(self.__start_node.x, self.__start_node.y, s=100, c="red", marker="o", alpha=0.5)    
        # Plot the goal node
        ax1.scatter(self.__goal_node.x, self.__goal_node.y, s=100, c="blue", marker="o", alpha=0.5)    
        
        # Plot the obstacles
        grid_x=[]
        grid_y=[]
        for i in range(self.__cost_map.shape[0]):
            if self.__cost_map[i]==INFINITY:
                x = self.__minx + i  % round(self.__width / self.__grid_resolution) * self.__grid_resolution
                y = self.__miny + int(i  / (self.__width / self.__grid_resolution)) * self.__grid_resolution
                grid_x.append(x)
                grid_y.append(y)
    
        ax1.scatter(grid_x, grid_y, s=100, c="black", marker="s", alpha=0.5) 
        
        node_list = []
        
        while 1: 
            # 1 step planning
            selected_node, node_list = self.__DStar_plannig()
            
            # Plot the selected_node
            ax1.plot(self.__calc_real_position(selected_node.x, self.__minx),
                         self.__calc_real_position(selected_node.y, self.__miny), "xc")
            
            if len(self.__close_list.keys()) % 10 == 0:
                plt.pause(0.001)
            
            if self.children_callback_flag:
                #If children callback is called, terminate planning
                break
        
        path_x = []
        path_y = []
        for i in node_list:
            path_x.append(i.x)
            path_y.append(i.y)
        
        ax1.plot(path_x, path_y, c = 'red', linewidth=1.0, linestyle='-', label='Ground Truth')
        
        ax1.scatter(selected_node.x, selected_node.y, s=100, c="cyan", marker="x", alpha=0.5)
        
        
        
        
                
if __name__ == '__main__':
    start_x = 1.0
    start_y = 1.0
    goal_x = 1.0
    goal_y = 48.0
    dstar_planning = DStar(start_x, start_y, goal_x, goal_y)
    
    dstar_planning.main()

    
    plt.show()
        
         
                    
                
         
        
        
        
        