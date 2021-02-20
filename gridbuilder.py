import numpy as np
from skimage.morphology import medial_axis
from skimage.util import invert

class GridBuilder():
    
    def __init__(self, polygons):
        """Initialize a grid builder, assumes polygons minimum coords is 0."""
        self.polygons = polygons
    
    def create_grid_25D(self, resolution=1):
        return self.__create_grid(resolution=resolution, altitude=None)
    
    def create_grid_2D(self, resolution=1, altitude=5):
        return self.__create_grid(resolution=resolution, altitude=altitude)
    
    def create_grid_skeleton(self, resolution=1, altitude=5):
        grid = self.create_grid_2D(resolution=resolution, altitude=altitude)
        skeleton = invert(medial_axis(invert(grid))).astype(int)
        return skeleton
        
    def __create_grid(self, resolution=1, altitude=None):
        """
        Returns a grid representation of a 2D configuration space
        based on given obstacle data, drone altitude and safety distance
        arguments.
        """

        north_min, east_min = np.min([(p.bounds[0], p.bounds[1]) for p in self.polygons], axis=0)
        north_max, east_max = np.max([(p.bounds[2], p.bounds[3]) for p in self.polygons], axis=0)

        # given the minimum and maximum coordinates we can
        # calculate the size of the grid.
        north_size = int(np.ceil(north_max - north_min)/resolution)
        east_size = int(np.ceil(east_max - east_min)/resolution)

        # Initialize an empty grid
        grid = np.zeros((north_size, east_size))

        # Populate the grid with obstacles
        for polyheight in self.polygons:
            if altitude is not None and polyheight.height < altitude:
                # Obstacle is below the altitude of the map
                continue
            # minx, miny, maxx, maxy
            bounds = np.array(polyheight.bounds)/resolution
            minx, miny, maxx, maxy = bounds
            north_min, east_min = np.floor([bounds[0], bounds[1]]).astype(int)
            north_max, east_max = np.ceil([bounds[2], bounds[3]]).astype(int)
            north_min, north_max = np.clip([north_min, north_max], 0, north_size-1)
            east_min, east_max = np.clip([east_min, east_max], 0, east_size-1)
            assert north_min < north_max+1
            assert east_min < east_max+1
            grid[north_min:north_max+1, east_min:east_max+1] = polyheight.height if altitude is None else 1

        return grid