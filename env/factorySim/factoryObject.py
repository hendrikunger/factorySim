
import random
import math
import time

from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, rotate, scale

class FactoryObject:

    def __init__(self, gid="not_set", name="no_name", origin=(0,0), poly:Polygon=box(0.0, 0.0, 1.0, 1.0)):
        
        self.gid = gid
        self.name = name
        self.color = [random.random(),random.random(),random.random()]
        self.origin = origin 
        self.rotation = 0 # roational change
        self.poly = poly #Element Multi Polygon Representation
        bounds = poly.bounds
        self.width = bounds[2] - bounds[0]
        self.height =  bounds[3] - bounds[1]
        self.center = poly.representative_point()
        self.group = None

    

    def rotate_Item(self, r):
        rotShift = r - self.rotation
        self.rotation = r
        
        self.poly = rotate(self.poly, rotShift, origin=self.origin, use_radians=True)

        bounds = self.poly.bounds
        self.width = bounds[2] - bounds[0]
        self.height =  bounds[3] - bounds[1]
        self.center = self.poly.representative_point()


    def translate_Item(self, x, y):
                    
        xShift = x - self.origin[0]
        yShift = y - self.origin[1]
        self.origin = (x, y)
        self.poly = translate(self.poly, xShift, yShift)
        self.center = self.poly.representative_point()
     

    # def scale_Points(self, xScale, yScale, minx, miny):    
    #     translate(self.poly, minx, miny)
    #     self.poly = scale(self.poly, xfact=xScale, yfact=yScale, origin=(0, 0))

    #     self.origin = ((self.origin[0] + minx) * xScale, (self.origin[1] + miny) * yScale)
    #     self.center = self.poly.representative_point()


    def __del__(self):
        del(self.poly)



 
def main():
    testmachine = FactoryObject("AABBAA")
    complexmachine = FactoryObject("COMPLEX", 1, 2, 3)



    print(F"Testmachine: {testmachine.gid}, {testmachine.origin.x}, {testmachine.origin.y}, {testmachine.origin.z}")
    print(F"Testmachine: {complexmachine.gid}, {complexmachine.origin.x}, {complexmachine.origin.y}, {complexmachine.origin.z}")

 

if __name__ == "__main__":
    main()
