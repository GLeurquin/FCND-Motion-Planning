from rtree import index
import time
from sklearn.neighbors import KDTree
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt

from queue import PriorityQueue
from bresenham import bresenham
from abc import ABC, abstractmethod

class SearchSpace(ABC):
    """Generic search space class implemeting A*"""
    
    @abstractmethod
    def explore_from_node(self, current_node):
        """Returns the other nodes we can reach from this node."""
        pass
    
    @abstractmethod
    def heuristic(self, current_node, next_node):
        pass
    
    @abstractmethod
    def is_node_in_search_space(self, node):
        pass
    
    @abstractmethod
    def move_cost(self, current_node, next_node):
        pass
    
    def find_node_in_search_space(self, node):
        return node
    
    def a_star(self, start, goal):
        path = []
        path_cost = 0
        queue = PriorityQueue()
        queue.put((0, start))
        visited = set(start)

        branch = {}
        found = False
        while not queue.empty():
            item = queue.get()
            current_node = item[1]
            if current_node == start:
                current_cost = 0.0
            else:              
                current_cost = branch[current_node][0]

            if current_node == goal:        
                print('Found a path.')
                found = True
                break

            # Explore the branches
            for next_node in self.explore_from_node(current_node):
                # get the tuple representation
                if next_node not in visited:
                    visited.add(next_node)
                    branch_cost = self.move_cost(current_node, next_node)
                    branch[next_node] = (branch_cost, current_node, next_node)
                    queue_cost = branch_cost + self.heuristic(next_node, goal)
                    queue.put((queue_cost, next_node))

        if found:
            # retrace steps
            n = goal
            path_cost = branch[n][0]
            path.append(goal)
            while branch[n][1] != start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1])
        else:
            print('**********************')
            print('Failed to find a path!')
            print('**********************')
            return None, None
        path = path[::-1]
        return path, path_cost
    
    def prune_path(self, path):
        return path
    
    def search_path(self, start, goal):
        """Returns a path and its cost from start to goal."""
        start_g = self.find_node_in_search_space(start)
        goal_g = self.find_node_in_search_space(goal)
        if start_g == goal_g:
            print('Start equals goal. No path to search')
            return [start, goal], 0
        path, path_cost = self.a_star(start_g, goal_g)
        if path is None:
            return None, None
        print(f'Path before pruning has length {len(path)+2}')
        path = self.prune_path(path)
        path = [start] + path + [goal]
        path_cost += self.move_cost(start, path[0]) + self.move_cost(goal, path[-1])
        print(f'Path after pruning has length {len(path)}')
        return path, path_cost


class GraphSearchSpace(SearchSpace):
    """
    Search space for a networkx graph. Supports 2D and 3D.
    """
    
    def __init__(self, graph, offset=None):
        self.graph = graph
        self.graph_dim = len(list(self.graph.nodes())[0])
        self.offset = offset if offset is not None else np.array([0]*self.graph_dim)
        
    def is_node_in_search_space(self, node):
        node = self.to_local_node(node)
        all_nodes = np.array(list(self.graph.nodes))
        min_values = np.amin(all_nodes, axis=0)
        max_values = np.amax(all_nodes, axis=0)
        node = np.array(node)
        return all(min_values <= node) and all(node <= max_values)
        
    def explore_from_node(self, current_node):
        return list(self.graph[current_node].keys())
        
    def heuristic(self, n1, n2):
        return LA.norm(np.array(n1)-np.array(n2))
    
    def move_cost(self, current_node, next_node):
        try:
            return self.graph.edges[current_node, next_node]['weight']
        except KeyError:
            return LA.norm(np.array(current_node)-np.array(next_node))        
    
    def find_node_in_search_space(self, node):
        nodes = np.array(list(self.graph.nodes()))
        node_idx = np.argmin(np.linalg.norm(nodes - np.array(node), axis=1))
        new_node = tuple(nodes[node_idx])
        print(f"Closest graph node to {node}: {new_node}")
        return new_node
    
    def to_local_node(self, node):
        node = node[:self.graph_dim]
        node = (np.array(node) - self.offset).astype(float)
        return tuple(node)
    
    def search_path(self, start, goal):
        print(f"Searching path from {start} to {goal} (no offset represented)")
        start = self.to_local_node(start)
        goal = self.to_local_node(goal)
        print(f"Searching path from {start} to {goal} (offset {self.offset})")
        path, path_cost = super().search_path(start, goal)
        if path is None:
            return None, None
        path = [tuple(np.array(p) + self.offset) for p in path]
        return path, path_cost

class GridGridSearchSpace():
    """Search on one grid, prune using another"""
    def __init__(self, search_grid, prune_grid, offset=None):
        grid_dim = len(search_grid.shape)
        default_offset = np.array([0]*len(search_grid.shape))
        self.search_grid_search_space = GridSearchSpace(search_grid, offset=default_offset, max_height=None)
        self.prune_grid_search_space = GridSearchSpace(prune_grid, offset=default_offset, max_height=None)
        self.offset = offset if offset is not None else default_offset
        
    def to_local_node(self, node):
        node = (np.array(node) - self.offset).astype(float)
        return tuple(node)
        
    def search_path(self, start, goal):
        if self.search_grid_search_space.is_2D:
            start, goal = start[:2], goal[:2]
        start = self.to_local_node(start)
        goal = self.to_local_node(goal)
        path, path_cost = self.search_grid_search_space.search_path(start, goal)
        if path is None:
            return None, None
        path = self.prune_grid_search_space.prune_path(path)
        path = [tuple(np.array(p) + self.offset) for p in path]

        return path, path_cost
    
class GraphGridSearchSpace():
    """Search in the graph, prune with the grid"""
    def __init__(self, graph, grid, offset=None):
        graph_dim = len(list(graph.nodes())[0])
        grid_dim = len(grid.shape)
        self.grid_search_space = GridSearchSpace(grid, offset=np.array([0]*grid_dim), max_height=None)
        self.graph_search_space = GraphSearchSpace(graph, offset=np.array([0]*graph_dim))
        self.offset = offset if offset is not None else default_offset
        
    def to_local_node(self, node):
        node = (np.array(node) - self.offset).astype(float)
        return tuple(node)
        
    def search_path(self, start, goal):
        start = self.to_local_node(start)
        goal = self.to_local_node(goal)
        path, path_cost = self.grid_search_space.search_path(start, goal)
        if path is None:
            return None, None
        path = self.grid_search_space.prune_path(path)
        path = [tuple(np.array(p) + self.offset) for p in path]
        return path, path_cost
    
class GridSearchSpace(SearchSpace):
    """
    Search space for a grid. Supports 2D, 2.5D and 3D grids.
    """
    
    def __init__(self, grid, offset=None, max_height=None):
        self.grid = grid
        self.offset = np.array(offset)
        self.is_2D = len(self.grid.shape) == 2 and len(self.offset) == 2
        self.is_3D = len(self.grid.shape) == 3
        self.is_25D = len(self.grid.shape) == 2 and len(self.offset) == 3
        self.max_height = int(max_height or (grid.shape[2]-1 if len(grid.shape) == 3 else np.max(grid))) # Not used for 2D or 3D maps
    
    def is_node_in_search_space(self, node):
        return self.node_in_grid(node)
    
    def node_in_grid(self, node):
        """Returns True if the node is within the grid bounds."""
        if self.is_25D and not 0 <= node[2] <= self.max_height:
            # 2.5D map, check height
            return False
        # Make sure coords are in grid
        shape = self.grid.shape
        grid_dim = len(shape)
        np_node = np.array(node[:grid_dim])
        return all(0 <= np_node) and all(np_node < shape)
    
    def node_in_obstacle(self, node):
        """Returns true if the node is in an obstacle"""
        grid_dim = len(self.grid.shape)
        obstacle = self.grid[node[:grid_dim]]
        
        if self.is_25D:
            # 2.5D map
            return node[2] <= obstacle and obstacle > 0
        # 2D or 3D map
        return bool(obstacle)
    
    def heuristic(self, n1, n2):
        return LA.norm(np.array(n1)-np.array(n2))
    
    def move_cost(self, n1, n2):
        return LA.norm(np.array(n1)-np.array(n2))
    
    def find_closest_non_obstacle(self, node):
        grid_dim = len(self.grid.shape)
        height = node[2] if self.is_25D else 1
        # Get points for which the grid value is <= height
        non_obstacle_points = np.transpose(np.nonzero(np.where(self.grid < height, 1, 0)))
        closest_node_idx = np.argmin(np.linalg.norm(non_obstacle_points - node[:grid_dim], axis=1))
        closest_node = non_obstacle_points[closest_node_idx]
        assert self.grid[tuple(closest_node)] < height, f"{self.grid[tuple(closest_node)]} >= {height}"
        if self.is_25D:
            return tuple([*closest_node, node[2]])
        return tuple(closest_node)
    
    def find_node_in_search_space(self, node):
        new_node = np.round(np.array(node)).astype(int)
        max_values = np.array([self.grid.shape[0]-1, self.grid.shape[1]-1, self.max_height])
        new_node = np.clip(new_node, 0, max_values[:len(node)])
        new_node = tuple(new_node)
        in_grid = self.node_in_grid(new_node)
        if not in_grid:
            print(f'[WARNING] Node is not in the grid: {new_node}')
        
        in_obstacle = self.node_in_obstacle(new_node)
        if in_obstacle:
            print(f'[WARNING] Node is in an obstacle {new_node}')
        
        if not in_grid or in_obstacle:
            non_obstacle_node = self.find_closest_non_obstacle(new_node)
            print(f'Finding closest non obstacle node in grid to {new_node}: {non_obstacle_node}')
            new_node = non_obstacle_node
            
        return new_node
        
    def explore_from_node(self, current_node):
        """Yields valid moves with their associated cost from this node."""
        shape = np.array(self.grid.shape)
        grid_dim = len(shape)
        n_dim = len(current_node)
        # Allow a move of 1 in any direction
        
        # All possible actions that allow
        # - to move 1 in each direction
        # - diagonally in 2 directions
        # - (for 3D) diagonally in 3 directions
        actions = np.array(np.meshgrid(*([[1, -1, 0]]*n_dim))).T.reshape(-1,n_dim)[:-1]
        for action in actions:
            next_node = tuple(np.array(current_node).astype(int) + action)
            action_valid = self.node_in_grid(next_node) and not self.node_in_obstacle(next_node)
            if not action_valid:
                continue
            yield next_node
    
    def to_local_node(self, node):
        node = (np.array(node) - self.offset).astype(int)
        return tuple(node)
    
    def search_path(self, start, goal):
        """If start is 3D, then path will be searched in 3D. Otherwise in 2D."""
        assert len(start) == len(goal), f"{start} and {goal} must have the same shape"
        if self.is_2D:
            start = start[:2]
            goal = goal[:2]
        print(f"Searching path from {start} to {goal} (no offset represented)")
        start = self.to_local_node(start)
        goal = self.to_local_node(goal)
        
        height_info = f"height {self.max_height}" if self.is_2D else ""
        print(f"Searching path from {start} to {goal} (offset {self.offset}) {height_info}")
        
        path, path_cost = super().search_path(start, goal)
        if path is None:
            return None, None
        path = [tuple(np.array(p) + self.offset) for p in path]
        return path, path_cost
    
    def bres_check(self, p1, p2):
        """Returns true if the two points can be linked without going through obstacles"""
        assert len(p1) == len(p2)
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        alt = min(p1[2], p2[2]) if len(p1) == 3 else 0
        for cell in cells:
            node = (cell[0], cell[1], alt)
            if self.node_in_obstacle(node):
                return False
        return True
    
    def prune_path(self, path):
        """Works on the path without offset."""
        if not path:
            return path

        pruned_path = [path[0]]
        for i in range(1, len(path)-1):
            p1 = pruned_path[-1]
            p2 = path[i]
            p3 = path[i+1]
            if self.bres_check(p1, p3):
                # The two points can be simplified
                continue
            pruned_path.append(p2)
        pruned_path.append(path[-1])
        return pruned_path
    
class RecedingHorizonSearchSpace():
    def __init__(self, get_local_search_space, get_rough_path):
        self.get_local_search_space = get_local_search_space
        self.get_rough_path = get_rough_path
        
    def search_path(self, start, goal):
        rough_path = self.get_rough_path(start, goal)
        return self.search_path_from_rough_path(rough_path)        
    
    def search_path_from_rough_path(self, rough_path):
        """Path: some rough 3D path to go from start to finish."""
        new_path = [rough_path[0]]
        cost = 0
        for w_idx, waypoint in enumerate(rough_path[1:]):
            already_at_waypoint = lambda: np.array_equal(np.array(new_path[-1]).astype(float), np.array(waypoint).astype(float))
            waypoint_in_local_map = already_at_waypoint()
            while not waypoint_in_local_map and not already_at_waypoint():
                current_node = new_path[-1] # last node of path
                print('Getting local search space')
                search_space = self.get_local_search_space(current_node)
                waypoint_in_local_map = search_space.is_node_in_search_space(waypoint)
                # Search local path, convert waypoints into local coordinates
                print('Searching path..')
                sub_path, sub_cost = search_space.search_path(
                    current_node,
                    waypoint
                )
                if len(sub_path) == 2:
                    print("DETECTED END OF PATH")
                    # Start == Goal, and there is nothing more we can do
                    waypoint_in_local_map = True
                
                if sub_path is None:
                    print(f"Partial path found: {new_path}. Got stuck at node {current_node} and next waypoint {next_waypoint}")
                    return None, None
                
                the_path = sub_path[1:] # Remove the first node since it's already in the global path
                if not waypoint_in_local_map:
                    # Remove the last node, as that node will be the waypoint, which is not in this map.
                    the_path = the_path[:-1]
                
                the_path = [p if len(p) == 3 else (*p, current_node[2]) for p in the_path]
                cost += sub_cost
                new_path += the_path
        
        return new_path, cost