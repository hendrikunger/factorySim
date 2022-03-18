
import random
import math
import time
from Polygon import Polygon as Poly
import Polygon.Utils
import Polygon.IO
from factorySim.Helpers.Point3D import Point3D


class MFO:

    def __init__(self, gid="not_set", name="no_name", origin_x=0, origin_y=0, origin_z=0, rotation=0 ):
        
        self.gid = gid
        self.name = name
        self.color = [random.random(),random.random(),random.random()]
        self.origin = Point3D(origin_x, origin_y, origin_z)
        self.baseOrigin = Point3D(origin_x, origin_y, origin_z)
        self.baseRotation = rotation  #in Radians
        self.rotation = 0
        self.poly = Poly()
        self.hull = Poly()
        self.polylist = []
        self.center = None
        self.width = 0
        self.height = 0
        

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
        #Calculate Origion in lower left corner
        boundingBox = self.poly.boundingBox()
        self.origin = Point3D(boundingBox[0], boundingBox[2], 0)
        #Calculate total Dimensions
        self.width = boundingBox[1] - boundingBox[0]
        self.height = boundingBox[3] - boundingBox[2]

  
    def updatePosition(self):
        wholePoly = Poly()
        for item in self.polylist:
            item.shift(self.origin.x, self.origin.y)
            for contour in item:
                wholePoly.addContour(contour)
        
        boundingBox = wholePoly.boundingBox()
        for item in self.polylist:
            item.rotate(self.baseRotation, boundingBox[0], boundingBox[2])


    def rotate_Item(self, r):

        wholePoly = Poly()
        for item in self.polylist:       
            for contour in item:
                wholePoly.addContour(contour)
                
        boundingBox = wholePoly.boundingBox()
        rotShift = r - self.rotation
        self.rotation = r
            
        centerX = boundingBox[0] + (boundingBox[1] - boundingBox[0])/2
        centerY = boundingBox[2] + (boundingBox[3] - boundingBox[2])/2
        rotatedX = math.cos(rotShift) * (self.baseOrigin.x - centerX) - math.sin(rotShift) * (self.baseOrigin.y-centerY) + centerX
        rotatedY = math.sin(rotShift) * (self.baseOrigin.x - centerX) + math.cos(rotShift) * (self.baseOrigin.y - centerY) + centerY
        self.baseOrigin.x = rotatedX
        self.baseOrigin.y = rotatedY

        for item in self.polylist:
            item.rotate(rotShift)

        wholePoly = Poly()
        for item in self.polylist:       
            for contour in item:
                wholePoly.addContour(contour)
                
        boundingBox = wholePoly.boundingBox()
        self.origin = Point3D(boundingBox[0], boundingBox[2], 0)

        self.finish()


    def translate_Item(self, x, y):
                    
        xShift = x - self.origin.x
        yShift = y - self.origin.y
        self.origin.x = x
        self.origin.y = y
        self.baseOrigin.x += xShift
        self.baseOrigin.y += yShift
            
        for item in self.polylist:
            item.shift(xShift, yShift)
        self.finish()
     

    def scale_Points(self, xScale, yScale, minx, miny):    
        for item in self.polylist:
            item.shift(minx, miny)
            item.scale(xScale, yScale, 0, 0)

        self.origin.x = (self.origin.x + minx) * xScale
        self.origin.y = (self.origin.y + miny) * yScale
        self.baseOrigin.x = self.origin.x
        self.baseOrigin.y = self.origin.y

    def __del__(self):
        del(self.poly)
        del(self.hull)
        del(self.polylist)

 
def main():
    testmachine = MFO("AABBAA")
    complexmachine = MFO("COMPLEX", 1, 2, 3)



    print(F"Testmachine: {testmachine.gid}, {testmachine.origin.x}, {testmachine.origin.y}, {testmachine.origin.z}")
    print(F"Testmachine: {complexmachine.gid}, {complexmachine.origin.x}, {complexmachine.origin.y}, {complexmachine.origin.z}")

 

if __name__ == "__main__":
    main()
