import numpy as np
from shapely.geometry import box, MultiPoint
from shapely.affinity import rotate, scale
from shapely.ops import unary_union



class FactoryCreator():


    def __init__(self, factoryWidth=32, factoryHeigth=18, maxShapeWidth=3, maxShapeHeight=2, amountRect=20, amountPoly=5, maxCorners=3):
        self.rng = np.random.default_rng()

        self.factoryWidth = factoryWidth
        self.factoryHeigth = factoryHeigth
        self.maxShapeWidth = maxShapeWidth
        self.maxShapeHeight = maxShapeHeight
        self.amountRect = amountRect
        self.amountPoly = amountPoly
        self.maxCorners = maxCorners

    def create_factory(self, load = False):
        polygons = []

        self.bb = box(0,0,self.factoryWidth,self.factoryHeigth)

        lowerLeftCornersRect = self.rng.integers([0,0], [self.factoryWidth - self.maxShapeWidth, self.factoryHeigth - self.maxShapeHeight], size=[self.amountRect,2], endpoint=True)
        lowerLeftCornersPoly = self.rng.integers([0,0], [self.factoryWidth - self.maxShapeWidth, self.factoryHeigth - self.maxShapeWidth], size=[self.amountPoly,2], endpoint=True)

        
        #Create Recangles
        for x,y in lowerLeftCornersRect:
            singlePoly = box(x,y,x + self.rng.integers(1, self.maxShapeWidth+1), y + self.rng.integers(1, self.maxShapeHeight+1))
            singlePoly= rotate(singlePoly, self.rng.choice([0,90,180,270]))  
            polygons.append(singlePoly)

        #Create Convex Polygons
        for x,y in lowerLeftCornersPoly: 
            corners = []
            corners.append([x,y]) # First Corner
            for _ in range(self.rng.integers(2,self.maxCorners+1)):
                corners.append([x + self.rng.integers(0, self.maxShapeWidth+1), y + self.rng.integers(0, self.maxShapeWidth+1)])

            singlePoly = MultiPoint(corners).minimum_rotated_rectangle
            singlePoly= rotate(singlePoly, self.rng.integers(0,361))  
            #Filter Linestrings
            if singlePoly.geom_type ==  'Polygon':
            #Filter small Objects
                if singlePoly.area > self.maxShapeWidth*self.maxShapeWidth*0.05:
                    polygons.append(singlePoly)
        # Flip on y because Pygames origin is in the top left corner
        self.multi = scale(unary_union(polygons), yfact=-1, origin=self.bb.centroid)

        return self.multi, self.bb

    @classmethod
    def load_pickled_factory(cls, filename):
        import pickle
        loaddata = pickle.load( open( filename, "rb" ) )
        cls.bb = loaddata["bounding_box"]
        cls.multi = loaddata["machines"]

        return cls.multi, cls.bb
    @classmethod
    def load_dxf_factory(cls, filename):
        import ezdxf
        from ezdxf.addons import geo
        from shapely.geometry import shape
        doc = ezdxf.readfile(filename)
        geo_proxy = geo.proxy(doc.modelspace())
        cls.multi = shape(geo_proxy)
        #find maximum width and height of factory
        bounds = cls.multi.bounds
        cls.bb = box(0,0,bounds[2],bounds[3])

        return cls.multi, cls.bb