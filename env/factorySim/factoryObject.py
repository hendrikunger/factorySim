
import random
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, rotate, scale

class FactoryObject:

    def __init__(self, gid="not_set", name="no_name", origin=(0,0), poly:Polygon=box(0.0, 0.0, 1.0, 1.0), color=None, rotation=0):
        
        self.gid = gid
        self.name = name
        if color is None or not np.any(color):
            self.color = [random.random(), random.random(), random.random()]
        else:
            self.color = color
        self.origin = origin 
        self.rotation = rotation # roational change
        self.poly = poly #Element Multi Polygon Representation
        bounds = poly.bounds
        self.width = bounds[2] - bounds[0]
        self.height =  bounds[3] - bounds[1]
        self.center = poly.representative_point()
        self.group = None

    

    def rotate_Item(self, r: float) -> None:
        """_summary_ Rotate the item to the given angle in radians

        Args:
            r (float): Rotation in radians
        """

        rotShift = r - self.rotation
        self.rotation = r
        self.poly = rotate(self.poly, rotShift, origin='center', use_radians=True)

        self.center = self.poly.representative_point()
        bounds = self.poly.bounds
        self.width = bounds[2] - bounds[0]
        self.height =  bounds[3] - bounds[1]
        self.origin = (bounds[0], bounds[1])


    def translate_Item(self, x: float, y: float) -> None:
        """_summary_ Translate the item by x and y

        Args:
            x (float): x Coordinate in factory space
            y (float): y Coordinate in factory space
        """
                    
        xShift = x - self.origin[0]
        yShift = y - self.origin[1]
        self.origin = (x, y)
        self.poly = translate(self.poly, xShift, yShift)
        self.center = self.poly.representative_point()


    def __del__(self):
        del(self.poly)



 
def main():
    testmachine = FactoryObject("AABBAA")
    complexmachine = FactoryObject("COMPLEX", 1, 2, 3)



    print(F"Testmachine: {testmachine.gid}, {testmachine.origin.x}, {testmachine.origin.y}, {testmachine.origin.z}")
    print(F"Testmachine: {complexmachine.gid}, {complexmachine.origin.x}, {complexmachine.origin.y}, {complexmachine.origin.z}")

 

if __name__ == "__main__":
    main()
