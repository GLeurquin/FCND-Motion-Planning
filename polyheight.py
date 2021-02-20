from shapely.geometry import Polygon, Point, LineString
import numpy as np

class PolyHeight(Polygon):
    def __init__(self, coords, height):
        super().__init__(coords)
        self._height = height

    @property
    def height(self):
        return self._height
    
    def collides_with_point(self, point):
        """Returns True if the point collides with the polygon.
        
        If the point is above the polygon, it does not collide.
        If the point is 2D, height is ignored.
        """
        if len(point) >= 3 and self.height < point[2]:
            # Point safely above poly
            return False
        x, y = point[:2]
        return self.contains(Point(x,y))
    
    def collides_with_line(self, line):
        """Returns True if this polygon intersects with the line
        line: a tuple of two points. 
        If the line is 2D, ignores height
        """
        p1, p2 = line
        assert len(p1) == len(p2), "Points are not of the same dimension"
        if len(p1) >= 3 and self.height < min(p1[2], p2[2]):
            # Line is safely above poly
            return False
        shapely_line = LineString([p1[:2], p2[:2]])
        return self.crosses(shapely_line)

def extract_polyheights(data, safety_distance=0):
    # North, east, alt
    min_bounds = np.floor(np.min(data[:, :3]-data[:, 3:], axis=0))-safety_distance
    max_bounds = np.ceil(np.max(data[:, :3]+data[:, 3:], axis=0))+safety_distance
    
    polygons = []
    for i in range(data.shape[0]):
        min_poly = data[i, :3]-data[i, 3:]-safety_distance
        max_poly = data[i, :3]+data[i, 3:]+safety_distance
        
        # Make the min be 0
        low = min_poly-min_bounds
        high = max_poly-min_bounds
        coords = [
            [low[0], low[1]],
            [low[0], high[1]],
            [high[0], high[1]],
            [high[0], low[1]]
        ]
        
        height = high[2]
        polygons.append(PolyHeight(coords, height))

    return polygons, min_bounds