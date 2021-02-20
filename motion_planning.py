import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

from gridbuilder import GridBuilder
from graphbuilder import GraphBuilder, CollisionTree
from search_space import GridSearchSpace, GraphSearchSpace, GraphGridSearchSpace, RecedingHorizonSearchSpace, GridGridSearchSpace
from polyheight import extract_polyheights

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class SearchSpaceType(str, Enum):
    GRID_2D = 'GRID_2D'
    GRID_25D = 'GRID_25D'
    SKELETON = 'SKELETON'
    VORONOI = 'VORONOI'
    VORONOI_WITH_PRUNING = 'VORONOI_WITH_PRUNING'
    RECEDING_HORIZON_RRT = 'RECEDING_HORIZON_RRT'
    RANDOM_SAMPLING_2D = 'RANDOM_SAMPLING_2D'
    RANDOM_SAMPLING_3D = 'RANDOM_SAMPLING_3D'
    
def create_search_space(search_space_type: SearchSpaceType, search_altitude: float, safety_distance=np.array([5, 5, 0])):
    """search_altitude: only for 2D search space"""
    print('Creating search space')
    # Read in obstacle map
    data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
    # read lat0, lon0 from colliders into floating point values
    with open('colliders.csv') as f:
        origin_pos_data = f.readline().split(',')
    lat0 = float(origin_pos_data[0].strip().split(' ')[1])
    lon0 = float(origin_pos_data[1].strip().split(' ')[1])
    home = (lon0, lat0, 0)
    # Define a grid for a particular altitude and safety margin around obstacles
    polygons, min_bounds = extract_polyheights(data, safety_distance=safety_distance)
    
    north_offset, east_offset = min_bounds[:2].astype(int)
    print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
    offset = np.array([north_offset, east_offset, 0])
    offset_2D = offset[:2] # 2D map
    
    if search_space_type == SearchSpaceType.SKELETON:
        grid_builder = GridBuilder(polygons)
        prune_grid = grid_builder.create_grid_2D(altitude=search_altitude)
        skel_grid = grid_builder.create_grid_skeleton(altitude=search_altitude)
        search_space = GridGridSearchSpace(skel_grid, prune_grid, offset=offset_2D)
        #search_space = GridSearchSpace(grid, offset=offset_2D)
    elif search_space_type == SearchSpaceType.GRID_25D:
        grid = GridBuilder(polygons).create_grid_25D()
        search_space = GridSearchSpace(grid, offset=offset)
        
    elif search_space_type == SearchSpaceType.GRID_2D:
        grid = GridBuilder(polygons).create_grid_2D(altitude=search_altitude)
        search_space = GridSearchSpace(grid, offset=offset_2D)
        
    elif search_space_type == SearchSpaceType.RECEDING_HORIZON_RRT:
        # Very very naive rough path that just goes in a straight line
        get_rough_path = lambda s, g: [s, g]
        collision_tree = CollisionTree(polygons)
    
        def get_local_rrt_search_space(current_position, horizon=300):
            current_position = tuple((np.array(current_position) - offset[:len(current_position)]).astype(float))
            print("Creating local ss", current_position)

            gb = GraphBuilder(np.array(current_position[:2])-horizon, np.array(current_position[:2])+horizon, polygons, collision_tree=collision_tree)
            print('Creating RRT graph')
            graph = gb.create_rrt_graph(current_position, num_vertices=300, dt=5)
            print('Created RRT graph')
            #all_nodes = np.array(list(graph.nodes))
            #min_values = np.amin(all_nodes, axis=0)
            #max_values = np.amax(all_nodes, axis=0)

            search_space = GraphSearchSpace(graph, offset=offset_2D)
            print('Created search space')
            return search_space
        
        search_space = RecedingHorizonSearchSpace(get_local_rrt_search_space, get_rough_path)
    elif search_space_type == SearchSpaceType.VORONOI:
        graph_builder = GraphBuilder(np.array([0, 0, 0]), np.array([926, 926, 220]), polygons)
        graph = graph_builder.create_voronoi_graph(drone_altitude=search_altitude)
        search_space = GraphSearchSpace(graph, offset=offset_2D)
        
    elif search_space_type == SearchSpaceType.VORONOI_WITH_PRUNING:
        graph_builder = GraphBuilder(np.array([0, 0, 0]), np.array([926, 926, 220]), polygons)
        graph = graph_builder.create_voronoi_graph(drone_altitude=search_altitude)
        grid = GridBuilder(polygons).create_grid_25D()
        search_space = GraphGridSearchSpace(graph, grid, offset=offset)
        
    elif search_space_type == SearchSpaceType.RANDOM_SAMPLING_2D:
        graph_builder = GraphBuilder(np.array([0, 0, 0]), np.array([926, 926, 220]), polygons)
        graph = graph_builder.create_sample_graph(n_samples=500, n_d=2, k=10)
        search_space = GraphSearchSpace(graph, offset=offset_2D)
        
    elif search_space_type == SearchSpaceType.RANDOM_SAMPLING_3D:
        graph_builder = GraphBuilder(np.array([0, 0, 0]), np.array([926, 926, 220]), polygons)
        graph = graph_builder.create_sample_graph(n_samples=1000, n_d=3, k=10)
        search_space = GraphSearchSpace(graph, offset=offset)
        
    else:
        raise ValueError(f"Unknown search space type {search_space_type}")

    print('Search space created')
    return search_space, home

class MotionPlanning(Drone):

    def __init__(self, connection, search_space, home):
        super().__init__(connection)
        self.flight_altitude = 5 # Relative to initial position
        self.search_space = search_space
        self.home = home
        
        # north, east, altitude, heading
        self.target_position = np.array([0.0, 0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL
        
        self.grid_goal = (0, 0)
        
        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)
        
        self.landing_count = 0
        
    
        
    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                if self.waypoints:
                    self.waypoint_transition()
                else:
                    self.landing_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[:3] - self.local_position[:3] * np.array([1, 1, -1])) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                elif np.linalg.norm(self.local_velocity[:3]) < 1.0:
                    self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            # Instead of checking if the local position is 0,
            # Check if the vertical velocity has been 0 for a while.
            # This allows landing on buildings.
            if abs(self.local_velocity[2]) < 0.05:
                self.landing_count += 1
                if self.landing_count > 5:
                    self.disarming_transition()
                    self.landing_count = 0
            else:
                self.landing_count = 0
            
            # Old code
            #if self.global_position[2] - self.global_home[2] < 0.1:
            #    if abs(self.local_position[2]) < 0.01:
            #        self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()
    
    def arming_transition(self):
        print("arming transition")
        self.take_control()
        self.arm()
        self.flight_state = States.ARMING
        
        
    def takeoff_transition(self):
        target_altitude = self.target_position[2]
        print(f"takeoff transition to {target_altitude}")
        self.takeoff(target_altitude)
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        if not self.waypoints:
            print("No waypoints set")
            return
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(
            self.target_position[0], # north
            self.target_position[1], # east
            self.target_position[2], # altitude
            self.target_position[3] # heading
        )

    def landing_transition(self):
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        print("disarm transition")
        self.disarm()
        self.release_control()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        print("manual transition")
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def global_to_local(self, global_position):
        lon_p, lat_p, alt_p = global_position
        lon_h, lat_h, alt_h = self.global_home
        # Get easting and northing of global_position
        (easting_p, northing_p, zone_number_p, zone_letter_p) =  utm.from_latlon(lat_p, lon_p)
        # Get easting and northing of global_home
        (easting_h, northing_h, zone_number_h, zone_letter_h) =  utm.from_latlon(lat_h, lon_h)
        # Create local_position from global and home positions                                     
        local_position = numpy.array([
            northing_p-northing_h,
            easting_p-easting_h,
            -(alt_p-alt_h)
        ])

        return local_position
    
    def set_goal(self, lon, lat, alt=0):
        self.global_goal = (lon, lat, alt)
    
    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        
        self.set_home_position(*self.home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Load the map and build the search space for path planning
        print("Loading data")       
        
        # Starting point on the grid is the current local position
        # Local position is the map center. Needs to be offset
        print("Local position: ", self.local_position)
        
        
        
        grid_start = self.local_position.astype(int)[:3]
        grid_start[2] *= -1
        grid_start[2] += self.flight_altitude
        self.target_position[2] = grid_start[2]
        
        
        self.grid_goal = global_to_local(self.global_goal, self.global_home).astype(int)
        self.grid_goal[2] *= -1

        # Run A* to find a path from start to goal in the given search space
        start, goal = grid_start, self.grid_goal
        print('Local Start and Goal: ', start, goal)
        
        print('Searching for a path')
        path, _ = self.search_space.search_path(start, goal)
        if not path:
            print('Could not find a path')
            self.flight_state = States.LANDING
            return
        print('Found path')
        # Convert path to waypoints
        waypoints = [[float(p[0]), float(p[1]), float(p[2] if len(p) == 3 else grid_start[2]), 0.0] for p in path]
        # Set self.waypoints
        
        self.waypoints = waypoints
        print(self.waypoints)
        #print('Sending waypoints to simulator')
        # I disabled this because the simulator just hangs forever when this is enabled for some unknown reason...
        #self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        super().start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--goal', type=str, default='-122.395975,37.795270,5', help="The goal global coordinates: lon,lat,alt")
    parser.add_argument('--search-space-type', type=SearchSpaceType, dest="search_space_type", choices=list(SearchSpaceType), default=SearchSpaceType.GRID_25D)
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=180, threaded=False, PX4=False)
    search_flight_altitude = 5
    search_space, home = create_search_space(args.search_space_type, search_flight_altitude)
    drone = MotionPlanning(conn, search_space, home)
    drone.set_goal(*[float(c) for c in args.goal.split(',')])
    
    # Some example goal positions
    #drone.set_goal(-122.397325, 37.794037, 5) # Home
    #drone.set_goal(-122.395975, 37.795270, 5) # Other location
    #drone.set_goal(-122.400020, 37.794514, 18) # Top of building
    #drone.set_goal(-122.401903, 37.796390, 18) # Another Top of building
    
    time.sleep(2)

    drone.start()
