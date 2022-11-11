class BaseFactoryConf():
    WIDTH = 30000
    HEIGHT = 3000
    MAXSHAPEWIDTH = 500
    MAXSHAPEHEIGHT = 500
    AMOUNTRECT = 150
    AMOUNTPOLY = 0
    MAXCORNERS = 3

    MINDEADENDLENGTH = 2000 # If Deadends are shorter than this, they are deleted
    MINPATHWIDTH = 1000  # Minimum Width of a Road to keep
    MINTWOWAYPATHWIDTH = 2000  # Minimum Width of a Road to keep
    BOUNDARYSPACING = 500  # Spacing of Points used as Voronoi Kernels
    SIMPLIFICATIONANGLE = 35 # Angle in degrees, used for support point calculation in simple path

    @classmethod
    def creationParameters(cls):
        return (cls.WIDTH, cls.HEIGHT), cls.MAXSHAPEWIDTH, cls.MAXSHAPEHEIGHT, cls.AMOUNTRECT, cls.AMOUNTPOLY, cls.MAXCORNERS

    @classmethod
    def pathParameters(cls):
        return cls.BOUNDARYSPACING, cls.MINDEADENDLENGTH, cls.MINPATHWIDTH, cls.MINTWOWAYPATHWIDTH, cls.SIMPLIFICATIONANGLE

class BIG(BaseFactoryConf):
    WIDTH = 128000
    HEIGHT = 70000
    MAXSHAPEWIDTH = 10000
    MAXSHAPEHEIGHT = 8000
    AMOUNTRECT = 60
    AMOUNTPOLY = 10
    MAXCORNERS = 3

    BOUNDARYSPACING = 800
    MINDEADENDLENGTH = 4000

class SMALL(BaseFactoryConf):
    WIDTH = 32000
    HEIGHT = 18000
    MAXSHAPEWIDTH = 3000
    MAXSHAPEHEIGHT = 2000
    AMOUNTRECT = 10
    AMOUNTPOLY = 0
    MAXCORNERS = 3

class SMALLSQUARE(BaseFactoryConf):
    WIDTH = 10000
    HEIGHT = 10000
    MAXSHAPEWIDTH = 2000
    MAXSHAPEHEIGHT = 2000
    AMOUNTRECT = 10
    AMOUNTPOLY = 0
    MAXCORNERS = 3

class EDF(BaseFactoryConf):
    WIDTH = 6000 #9,60m
    HEIGHT = 6700 #11,20m
    MAXSHAPEWIDTH = 1000
    MAXSHAPEHEIGHT = 1000
    AMOUNTRECT = 30
    AMOUNTPOLY = 5
    MAXCORNERS = 3

    BOUNDARYSPACING = 500
    MINDEADENDLENGTH = 500