from cmath import isnan
from json import detect_encoding
from json.encoder import INFINITY
from math import nan
from turtle import distance
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import math

class AStar():
    def __init__(self):
        '''
        Initialize Parameters.
        '''
        self.__weight_h = 3.0
        
        '''
        Set cost map
        '''
        # self.grid_size = 0.1 # [m]
        self.width = int(20)
        self.height = int(20)
        
        self.__cost_map = np.zeros(self.width * self.height)
        
        '''
        Set obstacles
        '''
        for i in range(self.height):
            index = i * self.width
            self.__cost_map[index]=INFINITY
            index2 = (i + 1) * self.width - 1
            self.__cost_map[index2]=INFINITY
            
        for i in range(self.width):
            index = i
            self.__cost_map[index]=INFINITY
            index2 = (self.height - 1) * self.width + i
            self.__cost_map[index2]=INFINITY
            
        for i in range(int(2 * self.width / 3)):
            index = i + (int(self.height / 4) - 1) * self.width
            index2 = self.width - 1 - i + int(self.height / 2 - 1) * self.width
            index3 = int(i + int(self.height / 4 * 3 - 1) * self.width)
            self.__cost_map[index] = INFINITY
            self.__cost_map[index2] = INFINITY
            self.__cost_map[index3] = INFINITY
        
        '''
        Set start and goal
        '''
        self.__start_node = np.array([1, 1])
        self.__goal_node = np.array([18, 18])
        
        '''
        Initialize the open list and the close list
        '''
        # Calculate a heauristic cost of the start node
        init_cost = self.__calculate_heauristic_cost(self.__start_node)
        # The index of point(i,j) is calculated as j*width + i 
        init_index = (self.__start_node[1]) * self.width + self.__start_node[0]
        self.__open_list = []
        self.__open_list.append(np.array([init_index, init_index, self.__start_node, init_cost], dtype=object)) # [index, parent_index, Coordinates, cost] list of the opended nodes
        self.__close_list = np.full((self.height * self.width, 3), [math.nan, math.nan, math.nan])     # The coordinates list of the closed nodes [x, y, parent_index]
        
        
        
        #---------------------------
        self.parent_callback_flag = False
        self.__node_list = []
        # self.__tree_lank_count = 1      
    
    def __calculate_heauristic_cost(self, selected_node):
        '''
        Calcurate heauristic cost h(n)
        Input:
        Output: h_cost    The heauristic cost of the selected node
        '''
        h_cost = self.__weight_h * np.sqrt((selected_node[0]-self.__goal_node[0])**2 + (selected_node[1]-self.__goal_node[1])**2)
        
        return h_cost
    
    def __is_goal(self, selected_node_position):
        # point = selected_node[2]
        diff = np.sqrt((selected_node_position[0] - self.__goal_node[0])**2 + (selected_node_position[1] - self.__goal_node[1])**2)
        if diff <= 0.5:
            return True
        else:
            return False
        
    def __callback_parents(self, selected_node):
        node = selected_node[2] 
        index = int(selected_node[1])
        # node_list = [node]
        self.__node_list.append(node)
        
        start_index = self.__start_node[1]*self.width + self.__start_node[0] 
        # 2 heap method
        while index != start_index: 
            index = self.__close_list[int(index), 2]
            node = self.__close_list[int(index), 0:2]
            self.__node_list.append(node)
            
        self.parent_callback_flag = True
            
    def __is_in_openlist(self, selected_node_index):
        for i in range(len(self.__open_list)):
            node = self.__open_list[i]
            if node[0] == selected_node_index:
                return True, i
        
        return False, nan
    
    def __update_cost(self, selected_node, openlist_index, distance):
        f_cost = distance + selected_node[3]
        h_cost = self.__calculate_heauristic_cost(selected_node)
        
        found_node = self.__open_list[openlist_index]
        
        if f_cost+h_cost < found_node[3]:
            found_node[3] = f_cost + h_cost
            found_node[1] = selected_node[0]
            self.__open_list[openlist_index] = found_node 
    
    def AStar_main(self):
        # If already end, end
        if self.parent_callback_flag:
            return  self.__cost_map, self.__close_list, self.__start_node, self.__goal_node, self.__node_list
        # If the open list is empty, the seach is failed.
        if len(self.__open_list) == 0:
            print("A-Star search failed....")
            exit()
                    
        # Get minimum cost node from the Open List
        
        min_index = self.__open_list.index(min(self.__open_list, key = lambda x:x[3]))
        selected_node = self.__open_list.pop(min_index)
        
        selected_node_position = selected_node[2]
        selected_node_index = int(selected_node[0])
        
        # Check if the selected node is close to the goal
        if self.__is_goal(selected_node_position):
            self.__callback_parents(selected_node)
            return self.__cost_map, self.__close_list, self.__start_node, self.__goal_node, self.__node_list

        # Put the selected node into the closed list
        self.__close_list[selected_node_index] = np.array([selected_node_position[0], selected_node_position[1], selected_node[1]])
        
        
        # Search all nearest nodes
        for i in range(-1,2):
            for j in range(-1,2):
                # If the node is itself, continue
                if i==0 and j==0:
                    continue
    
                index = selected_node_index + i + j * self.width
                
                # Check if the searching node is searchable.
                if self.__cost_map[index] == INFINITY:
                    continue
                
                # Check if the searching node is in the close list.
                if not np.isnan(self.__close_list[index][0]):
                    distance = np.sqrt(i**2 + j**2)
                    f_cost = distance + selected_node[3]
                    h_cost = self.__calculate_heauristic_cost(selected_node_position)
                    
                    continue
                                
                # Check if the searced node is in the open list.
                is_in_openlist_flag, openlist_index = self.__is_in_openlist(selected_node_index)
                if is_in_openlist_flag:
                    # Update the cost of the node.
                    distance = np.sqrt(i**2 + j**2)
                    self.__update_cost(selected_node, openlist_index, distance)
                    continue
                
                # Calculate the cost of the searching node.
                distance = np.sqrt(i**2 + j**2)
                f_cost = distance + selected_node[3]
                h_cost = self.__calculate_heauristic_cost(selected_node_position)
                
                # Put the searching node into the open list
                new_point = selected_node_position + np.array([i, j])
                
                self.__open_list.append(np.array([index, selected_node_index, new_point, f_cost + h_cost], dtype=object))
        
        return self.__cost_map, self.__close_list, self.__start_node, self.__goal_node, self.__node_list

def anim(i, astar_plan):
    plt.cla()
    
    cost_map, close_list, start_node, goal_node, node_list = astar_plan.AStar_main()
    ax1 = plt.subplot()
    
    grid_x=[]
    grid_y=[]
    for i in range(cost_map.shape[0]):
        if cost_map[i]==INFINITY:
            x = int(i % astar_plan.width)
            y = int(i / astar_plan.width)
            grid_x.append(x)
            grid_y.append(y)
    
    ax1.scatter(grid_x, grid_y, s=100, c="black", marker="s", alpha=0.5)    
    ax1.scatter(close_list[:,0], close_list[:,1], s=100, c="green", marker="x", alpha=0.5)
    
    path_x = []
    path_y = []
    for i in node_list:
        path_x.append(i[0])
        path_y.append(i[1])
    
    ax1.plot(path_x, path_y, c = 'red', linewidth=1.0, linestyle='-', label='Ground Truth')
        
        
                
if __name__ == '__main__':
    astar_planning = AStar()
    
    # while True:
    #     astar_planning.AStar_main()
    
    fig = plt.figure(figsize=(14, 7))
    frame = int(3600)
    
    video = animation.FuncAnimation(fig, anim, frames=frame, fargs=(astar_planning,), blit=False, interval=1, repeat=False)
    
    plt.show()
        
         
                    
                
         
        
        
        
        