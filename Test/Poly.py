#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Polygon import Polygon as poly
from Polygon.Shapes import Star, Circle, Rectangle, SierpinskiCarpet
import Polygon.IO
from Polygon.Utils import convexHull, tile, tileEqual, tileBSP, reducePoints, reducePointsDP, cloneGrid
import random, math
from Helpers.Point3D import Point3D
from itertools import combinations


def testPoly():
    
    
    polylist = []
    
    result = poly(((0.0, 0.0), (15.0, 0.0), (10.0, 5.0), (0.0, 5.0)))
    t = poly([[1.0, 1.0], [3.0, 3.0], [1.0, 5.0]])
    #poly([[2, 2], [5, 5], [8, 2]])
    result -= t # gives a rectangle with a triangular hole
    result.simplify()
    #z = poly()
    #q.addContour(((1.0, 1.0), (3.0, 1.0), (2.0, 3.0)))
    #q.shift(10,0)
    
    #polylist.append(q)
    polylist.append(t)
    
    a = (0,0)
    b = (0,20)
    c = (10,20)
    d = (10,0)
    
    testp = poly((a,b,c,d))

    r1 = Rectangle(10,20)
    r1.shift(9,11)
    r1.addContour((a,b,c,d))
    polylist.append(r1)
    r2 = Rectangle(20,10)
    r3 = Rectangle(10,5)
    r3.shift(5,11)
    
    r4 = Rectangle(5,10)
    r4.shift(11,5)

    rects = [r1,r2,r3,r4]
    cols = []

    for a,b in combinations(rects, 2):
        if a.overlaps(b):
            cols.append(a & b)
       
    testc = poly(((1.0, 1.0), (3.0, 1.0), (2.0, 3.0)))
    testd = poly([[15, 15], [20, 15], [20, 20], [15,20]])
    print(testp.area())
    #testp.addContour([[2, 2], [5, 5], [8, 2]],True)
    testp-=poly([[2, 2], [5, 5], [8, 2]])
    print(testp.area())
    testp.addContour([[15, 15], [20, 15], [20, 20], [15,20]])
    #print(testp.area())
    #testp.simplify()
    print(testp.area())
    print(testp)
    #print(testp.nPoints())
    #print(testp.area())
    #print(testp.center())
    
    testp.scale(2,2,0,0)
    #testp.rotate(math.pi/2)
    #print(testp)
    
    sum = poly()
    
    #for i, polygon in enumerate(polylist):
    #    for j, contour in enumerate(polygon):
    #        print(F"{i}.{j} - {contour}")
    #        sum.addContour(contour)
    #print("result")
    #print(sum)  
    
    Polygon.IO.writeSVG('test.svg', testp, width=300, height=300)
    #Polygon.IO.writeSVG('cols.svg', polylist, width=300, height=300)
def operationsExample():
    print('### Operations')
    # create a circle with a hole
    p1 = Circle(1.0) - Circle(0.5)
    # create a square
    p2 = Rectangle(0.7)
    # shift the square a little bit
    p2.shift(0.25, 0.35)
    plist = [p1, p2]
    slist = ['p1', 'p2']

    # addition, the same as logical OR (p1 | p2)
    p = p1 + p2
    p.shift(2.5, 0.0)
    plist.append(p)
    slist.append('p1 + p2 (OR)')

    # subtraction
    p = p1 - p2
    p.shift(5.0, 0.0)
    plist.append(p)
    slist.append('p1 - p2')

    # subtraction
    p = p2 - p1
    p.shift(7.5, 0.0)
    plist.append(p)
    slist.append('p2 - p1')

    # logical AND
    p = p2 & p1
    p.shift(10.0, 0.0)
    plist.append(p)
    slist.append('p2 AND p1')

    # logical XOR
    p = p2 ^ p1
    p.shift(12.5, 0.0)
    plist.append(p)
    slist.append('p2 XOR p1')

    # draw the results of the operations
    writeSVG('Operations.svg', plist, width=800, labels=slist, labels_centered=True)


def cookieExample():
    print('### Cookies!')
    # construct a christmas cookie with the help of the shapes
    star   = Star(radius=2.0, center=(1.0, 3.0), beams=5, iradius=1.4)
    circle = Circle(radius=1.0, center=(1.0, 3.0), points=64)
    cookie = star-circle
    # shift star and circle to the right to plot all polygons
    # on one page
    star.shift(5.0, 0.0)
    circle.shift(10.0, 0.0)
    # plot all three to an svg file
    writeSVG('Cookie.svg', (cookie, star, circle))

    # break a polygon object into a list of polygons by arranging
    # it on tiles
    # tile into 3x3 parts
    plist = tileEqual(cookie, 3, 3)
    writeSVG('CookieTiled1.svg', plist)
    # test tile at x = 0.3, 0.5 and y = 2.7, 3.1
    plist = tile(cookie, [0.3, 0.5], [2.7, 3.1])
    writeSVG('CookieTiled2.svg', plist)

    # let's simulate an explosion, move all parts away
    # from the cookie's center, small parts are faster
    xc, yc = cookie.center()
    for p in plist:
        if p:
            # speed/distance
            dval = 0.1 / p.area()
            x, y = p.center()
            # move the part a little bit
            p.shift(dval*(x-xc), dval*(y-yc))
            # and rotate it slightly ;-)
            p.rotate(0.2*math.pi*(random.random()-0.5))
    writeSVG('CookieExploded.svg', plist)
    

def reduceExample():
    print('### Reduce points')
    # read Polygon from file
    p = Polygon('testpoly.gpf')
    # use ireland only, I know it's contour 0
    pnew = Polygon(p[0])
    # number of points
    l = len(pnew[0])
    # get shift value to show many polygons in drawing
    bb = pnew.boundingBox()
    xs = 1.1 * (bb[1]-bb[0])
    # list with polygons to plot
    plist = [pnew]
    labels = ['%d points' % l]
    while l > 30:
        # reduce points to the half
        l = l//2
        print('Reducing contour to %d points' % l)
        pnew = Polygon(reducePoints(pnew[0], l))
        pnew.shift(xs, 0)
        plist.append(pnew)
        labels.append('%d points' % l)
    # draw the results
    writeSVG('ReducePoints.svg', plist, height=400, labels=labels)
    if hasPDFExport:
        writePDF('ReducePoints.pdf', plist)


def reduceExampleDP():
    print('### Reduce points with DP')
    # read Polygon from file
    p = Polygon('testpoly.gpf')
    # use ireland only, I know it's contour 0
    pnew = Polygon(p[0])
    # start tolerance
    tol = 0.125
    # get shift value to show many polygons in drawing
    bb = pnew.boundingBox()
    xs = 1.1 * (bb[1]-bb[0])
    # list with polygons to plot
    plist = [pnew]
    l = len(pnew[0])
    labels = ['%d points' % l]
    print('Contour has %d points' % l)
    while tol < 10:
        pnew = Polygon(reducePointsDP(pnew[0], tol))
        l = len(pnew[0])
        print('Reduced contour with tolerance %f to %d points' % (tol, l))
        pnew.shift(xs, 0)
        plist.append(pnew)
        labels.append('%d points' % l)
        tol = tol*2
    # draw the results
    writeSVG('ReducePointsDP.svg', plist, height=400, labels=labels)
    if hasPDFExport:
        writePDF('ReducePointsDP.pdf', plist)


def moonExample():
    print('### Moon')
    # a high-resolution, softly flickering moon,
    # constructed by the difference of two stars ...
    moon = Star(radius=3, center=(1.0, 2.0), beams=140, iradius=2.90) \
           - Star(radius=3, center=(-0.3, 2.0), beams=140, iradius=2.90)
    # plot the moon and its convex hull
    writeSVG('MoonAndHull.svg', (moon, convexHull(moon)), height=400, fill_opacity=(1.0, 0.3))
    # test point containment
    d = ['outside', 'inside']
    c = moon.center()
    print('Did you know that the center of gravitation of my moon is %s?' % d[moon.isInside(c[0], c[1])])


def xmlExample():
    print('### XML')
    cookie = Star(radius=2.0, center=(1.0, 3.0), beams=5, iradius=1.4)\
        - Circle(radius=1.0, center=(1.0, 3.0))
    writeXML('cookie.xml', (cookie, ), withHeader=True)
    p = readXML('cookie.xml')
    

def gnuplotExample():
    print('### Gnuplot')
    cookie = Star(radius=2.0, center=(1.0, 3.0), beams=5, iradius=1.4)\
        - Circle(radius=1.0, center=(1.0, 3.0))
    writeGnuplot('cookie.gp', (cookie,))
    writeGnuplotTriangles('cookieTri.gp', (cookie,))


def gridExample():
    print('### Grid')
    starGrid = cloneGrid(Star(beams=5), 0, 20, 20, 4, 4)
    starGrid.shift(-50, -50)
    cookie = Star(radius=30.0, beams=5, iradius=20.0) - Circle(radius=15.0)
    starCookie = cookie - starGrid
    writeSVG('StarCookie.svg', (starCookie,))
    if hasPDFExport:
        writePDF('StarCookie.pdf', (starCookie,))


def sierpinskiExample():
    print('### Sierpinski')
    for l in range(7):
        s = SierpinskiCarpet(level=l)
        print("SIERPINSKI CARPET - Level: %2d - Contours: %7d - Area: %g" % (l, len(s), s.area()))
        writeSVG('Sierpinski_%02d.svg' % l, (s,))


def tileBSPExample():
    print('### Tile BSP')
    # read Polygon from file
    p = Polygon('testpoly.gpf')
    print("tileBSP() - may need some time...")
    # generate a list of tiles
    tiles = list(tileBSP(p))
    # write results
    writeSVG('tileBSP.svg', tiles)
    if hasPDFExport:
        writePDF('tileBSP.pdf', tiles)
    print(" ...done!")


if __name__ == '__main__':
    #operationsExample()
    #cookieExample()
    #reduceExample()
    #reduceExampleDP()
    #gridExample()
    #moonExample()
    #xmlExample()
    #gnuplotExample()
    #sierpinskiExample()
    #tileBSPExample()
    testPoly()
