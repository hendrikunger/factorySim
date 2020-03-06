#!/usr/bin/env python3

from Helpers.Point3D import Point3D
import random
import time
from Polygon import Polygon as Poly
import Polygon.Utils
import Polygon.IO


class MFO:

    def __init__(self, gid="not_set", name="no_name", origin_x=0, origin_y=0, origin_z=0, rotation=0 ):
        
        self.gid = gid
        self.name = name
        self.color = [random.random(),random.random(),random.random()]
        self.origin = Point3D(origin_x, origin_y, origin_z)
        self.rotation = rotation  #in Radians
        self.poly = Poly()
        self.hull = Poly()
        self.polylist = []
        self.center = None
        

    def add_Loop(self, polygonpoints, isHole=False):
        self.poly.addContour(polygonpoints, isHole)
    
    def saveImg(self, index):
        Polygon.IO.writeSVG(f'test{index}.svg', self.poly)
    
    def close_Item(self):
        self.polylist.append(self.poly)
        self.poly = Poly()
    
    def finish(self):
        self.poly = Poly()
        self.hull = Poly()
        for item in self.polylist:
            for i, contour in enumerate(item):
                self.poly.addContour(contour, item.isHole(i))
                self.hull.addContour(contour)
        self.hull.simplify()
        centerx, centery = self.hull.center()
        self.center = Point3D(centerx, centery, 0)

  
    def updatePosition(self, xShift = None, yShift = None, rotShift = None):
        wholePoly = Poly()
        for item in self.polylist:
            if(xShift is None or yShift is None):    
                item.shift(self.origin.x, self.origin.y)
            else:
                item.shift(xShift, yShift)
            for contour in item:
                wholePoly.addContour(contour)
            boundingBox = wholePoly.boundingBox()
            if(rotShift is not None):
                item.rotate(rotShift, boundingBox[0], boundingBox[2])

    def rotate_translate_Item(self, x, y, r=None):
        if(r is not None):
            self.rotation = r
        xShift = x - self.origin.x
        yShift = y - self.origin.y
        self.origin.x = x
        self.origin.y = y
        if(r is not None):
            rotShift = r - self.rotation
            self.rotation = r
            self.updatePosition(xShift, yShift, rotShift)
        else:
            self.updatePosition(xShift, yShift)
        self.finish()

            

    def scale_Points(self, xScale, yScale, minx, miny):    
        for item in self.polylist:
            item.shift(minx, miny)
            item.scale(xScale, yScale, 0, 0)

        self.origin.x = (self.origin.x + minx) * xScale
        self.origin.y = (self.origin.y + miny) * yScale
 
def main():
    testmachine = Machine("AABBAA")
    complexmachine = Machine("COMPLEX", 1, 2, 3)



    print(F"Testmachine: {testmachine.gid}, {testmachine.origin.x}, {testmachine.origin.y}, {testmachine.origin.z}")
    print(F"Testmachine: {complexmachine.gid}, {complexmachine.origin.x}, {complexmachine.origin.y}, {complexmachine.origin.z}")

 

if __name__ == "__main__":
    main()

