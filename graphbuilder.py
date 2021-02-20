from shapely.geometry import LineString
from scipy.spatial import Voronoi
from bresenham import bresenham
from rtree import index
import numpy as np
from sklearn.neighbors import KDTree
import numpy.linalg as LA
from shapely.geometry import LineString
from skimage.morphology import medial_axis
import networkx as nx


class CollisionTree():
    def __init__(self, polygons):
        self.polygons = polygons
        self.tree = self.get_collision_tree()
    
    def get_collision_tree(self):
        """Returns a rectangle tree from the polygons for fast lookup."""
        tree = index.Index()
        for idx, polygon in enumerate(self.polygons):
            minx, miny, maxx, maxy = polygon.bounds
            left = minx
            bottom = miny
            right = maxx
            top = maxy
            bounds = (left, bottom, right, top)
            tree.insert(idx, bounds)

        return tree
    
    def collides_with_point(self, point):
        """Returns True if the point collides with a polygon.
        Uses a rectangle tree for fast lookup."""
        x, y = point[:2]
        # Using generators for lazy search makes it even faster
        intersections = self.tree.intersection((x, y, x, y))
        return any(self.polygons[idx].collides_with_point(point) for idx in intersections)
    
    def collides_with_line(self, p1, p2):
        """Returns True if the line joining the two points collides with a polyheight."""
        line = LineString([p1[:2], p2[:2]])
        return any(
            self.polygons[idx].collides_with_line((p1, p2))
            for idx in self.tree.intersection(line.bounds)
        )

def select_input(x_rand, x_near):
    dx, dy = np.array(x_rand[:2])-np.array(x_near[:2])
    return np.arctan2(dy, dx)

def new_state(x_near, u, speed, dt):
    dx = np.cos(u) * dt * speed
    dy = np.sin(u) * dt * speed
    return (x_near[0]+dx, x_near[1]+dy)

def get_nearby_state_and_simulate(idx, nodes, speed, dt, target_coords):
    x_near_idx = next(idx.nearest(target_coords, objects=False))
    x_near = nodes[x_near_idx]
    dx, dy = np.array(target_coords[:2])-np.array(x_near[:2])
    orientation = np.arctan2(dy, dx)
    x_new = new_state(x_near, orientation, speed, dt)
    return x_near, x_new, orientation

class GraphBuilder():
    
    def __init__(self, min_values, max_values, polygons, collision_tree=None):
        """Optional collision tree to avoid having to recreate it each time."""
        self.min_values = min_values
        self.max_values = max_values
        self.polygons = polygons
        self.collision_tree = CollisionTree(self.polygons) if collision_tree is None else collision_tree
        
    def create_sample_graph(self, n_samples=100, n_d=3, k=3):
        #samples = np.random.uniform(low=self.min_values[:n_d], high=self.max_values[:n_d], size=(n_samples, n_d))
        nodes = set()
        while len(nodes) < n_samples:
            random = np.random.uniform(low=self.min_values[:n_d], high=self.max_values[:n_d], size=(1, n_d))[0]
            random = tuple(random.astype(int))
            if random not in nodes and not self.collision_tree.collides_with_point(random):
                nodes.add(random)
        nodes = list(nodes)
        #nodes = [s for s in samples if not self.collision_tree.collides_with_point(s)]
        edges = self.get_edges_from_neighbors(nodes, k=k)
        graph = nx.Graph()
        graph.add_weighted_edges_from(edges)
        graph.add_nodes_from(tuple(map(tuple, nodes)))
        print(f"There are {len(nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def create_rrt_graph(self, x_init, num_vertices=300, dt=1, speed=1):
        """State is x, y, z, orientation"""
        assert len(x_init) == 3
        assert not self.collision_tree.collides_with_point((x_init)), "RRT root collides with an obstacle"
        height = x_init[2]
        x_init = x_init[:2]
        tree = nx.DiGraph()
        tree.add_node(x_init)
        
        nodes = []
        idx = index.Index()
        
        idx.insert(len(nodes), x_init)
        nodes.append(x_init)
        fail = 0
        MAX_TRIES=500
        while len(nodes) != num_vertices:
            if fail >= MAX_TRIES:
                raise ValueError('Something is wrong when creating the RRT, new nodes keep colliding')
            x_rand = np.random.uniform(self.min_values[:2], self.max_values[:2], (1, 2))[0]
            # sample states until a free state is found
            tries = 0
            while self.collision_tree.collides_with_point((*x_rand, height)) and tries < 15:
                tries += 1
                x_rand = np.random.uniform(self.min_values[:2], self.max_values[:2], (1, 2))[0]
            if tries >= MAX_TRIES:
                raise ValueError('Something is wrong when creating the RRT, random nodes keep colliding')
            x_rand = tuple(x_rand)
            x_near, x_new, orientation = get_nearby_state_and_simulate(idx, nodes, speed, dt, x_rand)           
            if self.collision_tree.collides_with_point((*x_new, height)):
                fail += 1
                continue
            fail = 0
            # Add the node
            idx.insert(len(nodes), x_new)
            nodes.append(x_new)
            # Add the edge
            cost = np.linalg.norm(np.array(x_near)-np.array(x_new))
            tree.add_weighted_edges_from([(tuple(x_near), tuple(x_new), cost)], orientation=orientation)
        return tree
    
    def create_voronoi_graph(self, drone_altitude, grid=None):
        """This method has the advantage of not using much floating point arithmetics."""
        obstacles = [tuple(np.array(poly.centroid.coords)[0].astype(int)) for poly in self.polygons if poly.height >= drone_altitude]
        # create a voronoi graph based on
        # location of obstacle centres
        vgraph = Voronoi(obstacles)
        
        # check each edge from graph.ridge_vertices for collision
        edges = []
        for v in vgraph.ridge_vertices:
            p1 = vgraph.vertices[v[0]]
            p2 = vgraph.vertices[v[1]]
            if any(np.array(p1) < self.min_values[:2]) or any(np.array(p1) >= self.max_values[:2]):
                continue
            if any(np.array(p2) < self.min_values[:2]) or any(np.array(p2) >= self.max_values[:2]):
                continue
            
            can_connect = self.can_connect_grid(grid, drone_altitude, p1, p2) if grid is not None else self.can_connect(p1, p2)
            if can_connect:
                # array to tuple for future graph creation step
                weight = LA.norm(np.array(p1) - np.array(p2))
                edges.append((tuple(p1), tuple(p2), weight))
        graph = nx.Graph()
        graph.add_weighted_edges_from(edges)
        print(f"There are {len(vgraph.vertices)} nodes and {len(edges)} edges")
        return graph
    
    def can_connect_grid(self, grid, drone_altitude, p1, p2):
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                return False
            # Next check if we're in collision
            obstacle = grid[c[0], c[1]]
            if obstacle >= drone_altitude:
                return False
        return True
    
    def can_connect(self, p1, p2):
        """Returns True if the line that p1 and p2
        creates does not intersect any polygon."""
        return not self.collision_tree.collides_with_line(p1, p2)

    def get_edges_from_neighbors(self, points, k=4):
        kdtree = KDTree(points)
        ii = kdtree.query(points, k=k+1, return_distance=False)
        # Remove first column of ii since the closest point will always be itself
        edges = set()
        for idx_a, closest_indices in enumerate(ii[:, 1:]):
            for idx_b in closest_indices:
                a = points[idx_a]
                b = points[idx_b]
                weight = LA.norm(np.array(a) - np.array(b))
                # Check if the edge already exists.
                # Edges are non-directional, so check both ways
                edge = (tuple(a), tuple(b), weight)
                edge_b = (tuple(b), tuple(a), weight)
                exists = edge in edges or edge_b in edges
                if not exists and self.can_connect(a, b):
                    edges.add(edge)
        return list(edges)