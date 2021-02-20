## Project: 3D Motion Planning
![Quad Image](./misc/enroute.png)

---

# Installation Requirements
* I'm [Rtree](https://pypi.org/project/Rtree/) for collision detection with polygons, so this package must be installed for this project to work, in addition to networkx, matplotlib, numpy

# Required Steps for a Passing Submission:
1. Load the 2.5D map in the colliders.csv file describing the environment.
2. Discretize the environment into a grid or graph representation.
3. Define the start and goal locations.
4. Perform a search using A* or other search algorithm.
5. Use a collinearity test or ray tracing method (like Bresenham) to remove unnecessary waypoints.
6. Return waypoints in local ECEF coordinates (format for `self.all_waypoints` is [N, E, altitude, heading], where the drone’s start location corresponds to [0, 0, 0, 0].
7. Write it up.
8. Congratulations!  Your Done!

## [Rubric](https://review.udacity.com/#!/rubrics/1534/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it! Below I describe how I addressed each rubric point and where in my code each point is handled.

### Explain the Starter Code

#### 1. Explain the functionality of what's provided in `motion_planning.py` and `planning_utils.py`
These scripts contain a basic planning implementation that includes:

`motion_planning.py`:
* A definition of the states: MANUAL, ARMING, TAKEOFF, WAYPOINT, LANDING, DISARMING and PLANNING
* The MotionPlanning class, which inherits from the Drone api. This is what controls the drone.
* The basic implementation goes into the different states in the following order:
  * starts in manual
  * goes to arming and starts planning a path
  * once path is ready, moves to takeoff
  * then proceeds through the waypoints found during the planning state
  * once the last waypoint is reached, proceeds to land
  * then disarms and returns to manual mode
* A basic planning using a_star to find a path between the grid start and grid goal, expressed in grid coordinates.

`planning_utils.py`:
Defines a way to create a 2D grid from the csv data. Also defines A\* implementation. I ended up moving these in `graphbuilder.py` and `search_space.py` to make it easier to use different implementations.

### Implementing Your Path Planning Algorithm


And here is a lovely picture of our downtown San Francisco environment from above! This is the map we will be using in this project.
![Map of SF](./misc/map.png)

#### 1. Set your global home position
Here students should read the first line of the csv file, extract lat0 and lon0 as floating point values and use the self.set_home_position() method to set global home. Explain briefly how you accomplished this in your code.

The home position is captured in the `create_search_space` function of `motion_planning.py`. This function loads the `colliders.csv`, retrieves the home coordinates and creates a search space. This home position is passed to the constructor of MotionPlanning. It is then set in the planning phase using `self.set_home_position(*self.home)`

#### 2. Set your current local position
Here as long as you successfully determine your local position relative to global home you'll be all set. Explain briefly how you accomplished this in your code.

The current local position is just retrived using `self.local_position`. This position is then used in the search space, which has an "offset" that can then convert the local position into grid coordinates.

#### 3. Set grid start position from local position
This is another step in adding flexibility to the start location. As long as it works you're good to go!

The start position is again, just `self.local_position`. As the search space takes care of converting it into grid coordinates using the offset. The only difference is the starting height, which is the 3rd part of `self.local_position` times `-1` because the height in `local_position` is reversed, and then I add some flight altitude so that the drone starts by taking off the ground.

The goal is coming from an instance variable `global_goal` that is set using `set_goal`

#### 4. Set grid goal position from geodetic coords
This step is to add flexibility to the desired goal location. Should be able to choose any (lat, lon) within the map and have it rendered to a goal location on the grid.

The goal is set using `set_goal` using geodetic coordinates. It is then converted to local coordinates in the planning step using `global_to_local`. Again the altitude is multiplied by -1

#### 5. Modify A* to include diagonal motion (or replace A* altogether)
Minimal requirement here is to modify the code in `planning_utils()` to update the A* implementation to include diagonal motions on the grid that have a cost of sqrt(2), but more creative solutions are welcome. Explain the code you used to accomplish this step.

Here I had some fun and implemented a few different search spaces. Let's start with the basic grid search space with A\*. The types of movements allowed in a grid search is defined in `search_space.py` in the class `GridSearchSpace` and function `explore_from_node`. This function returns adjacent nodes to the current node that we can explore during A\*. In particular, the line `np.array(np.meshgrid(*([[1, -1, 0]]*n_dim))).T.reshape(-1,n_dim)[:-1]` defined the deltas from one position to another. In 2D, this is simply returning a list like `[0, 1],[0, -1],[1, 0], [-1, 0],[1, 1],[-1, -1],[1, -1],[-1, 1]` and similarly in 3D. The `[+-1,+-1]` allow diagonal motions in all 4 directions. These deltas are then applied on the current node and the resulting positions are then filtered to remove out of grid positions and positions that end up in obstacles.

The cost is always the norm between one position and another, so for a diagonal move this would be sqrt(2). (as defined in `move_cost`)

`search_space.py` also defines a graph search space, which can use A* to find paths in a graph. It also defines search spaces that combine a grid and a graph, where the graph is used for path finding, and the grid is used for pruning using Bresenham. An example of this used is for voronoi, where the path is searched in the voronoi graph and pruned in the grid search space. There is also a search space combining two grids, one for path search and the other for path pruning, which is used with the skeleton search space.

#### 6. Cull waypoints
For this step you can use a collinearity test or ray tracing method like Bresenham. The idea is simply to prune your path of unnecessary waypoints. Explain the code you used to accomplish this step.

In `search_space.py`, the class `GridSearchSpace` defines a function `prune_path` which takes as input a path found using A\* for example, and prunes it using Bresenham. This `prune_path` is called in `search_path` (of the super class `SearchSpace`) right after A\* is run.

For pruning in graph search space, I find the path in the graph and then prune it again using Bresenham in a grid search space.


### Execute the flight
#### 1. Does it work?
It works!
Using the argument `--search-space-type`, one can use different search spaces to find a path from A to B. Options include
* `GRID_2D`: a 2D grid search space. This only flies at a fixed altitude
* (default) `GRID_25D`: a 2.5D grid search space. This can fly at different altitudes
* `VORONOI`: a 2D graph search space using voronoi cell edges as edges in the search graph.
* `VORONOI_WITH_PRUNING`: same as VORONOI, but uses a 2D Grid to prune the path found using the voronoi graph.
* `SKELETON`: a 2D grid search space using the medial_axis to define the search space.
* `RANDOM_SAMPLING_2D`: a graph search space in 2D using random nodes as vertices.
* `RANDOM_SAMPLING_3D`: a graph search space in 3D using random nodes as vertices. This one is fun since the drone will use different altitudes and a bit random paths to go from A to B. I also changed the way the landing works to allow the drone to land on top of buildings.
* `RECEDING_HORIZON_RRT`: an attempt at using receding horizon using Rapidly exploring random tree (RRT), however this does not work well as it's way too slow. Also the receding horizon is fully simulated at the start instead of re-calculating the path and search space at each waypoint.

Using the argument `--goal` one can set the lon,lat,alt of the next goal from where the drone currently is.

Note: 2D Search spaces only work when the start and goal are not above buildings, as the search space assumes a default altitude of 5. For 2.5D and 3D search spaces, start and goal can be anywhere (as long as it's not inside a buildings)


Drone flying over buildings
![Drone flying over buildings](./misc/flying_over_buildings.png)

Drone landing on top of building
![Drone landing on top of building](./misc/top_of_building.png)


### Double check that you've met specifications for each of the [rubric](https://review.udacity.com/#!/rubrics/1534/view) points.

# Extra Challenges: Real World Planning

For an extra challenge, consider implementing some of the techniques described in the "Real World Planning" lesson. You could try implementing a vehicle model to take dynamic constraints into account, or implement a replanning method to invoke if you get off course or encounter unexpected obstacles.

As discussed before, a few other planning and search space algorithms have been implemented and tested.
