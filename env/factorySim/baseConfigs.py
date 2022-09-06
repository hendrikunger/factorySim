class BaseFactoryConf():
    WIDTH = 300
    HEIGHT = 30
    MAXSHAPEWIDTH = 5
    MAXSHAPEHEIGHT = 5
    AMOUNTRECT = 150
    AMOUNTPOLY = 0
    MAXCORNERS = 3

    MINDEADENDLENGTH = 2.0 # If Deadends are shorter than this, they are deleted
    MINPATHWIDTH = 1.0  # Minimum Width of a Road to keep
    MINTWOWAYPATHWIDTH = 2.0  # Minimum Width of a Road to keep
    BOUNDARYSPACING = 1.5  # Spacing of Points used as Voronoi Kernels
    SIMPLIFICATIONANGLE = 35 # Angle in degrees, used for support point calculation in simple path

    @classmethod
    def creationParameters(cls):
        return (cls.WIDTH, cls.HEIGHT), cls.MAXSHAPEWIDTH, cls.MAXSHAPEHEIGHT, cls.AMOUNTRECT, cls.AMOUNTPOLY, cls.MAXCORNERS

    @classmethod
    def pathParameters(cls):
        return cls.BOUNDARYSPACING, cls.MINDEADENDLENGTH, cls.MINPATHWIDTH, cls.MINTWOWAYPATHWIDTH, cls.SIMPLIFICATIONANGLE

class BIG(BaseFactoryConf):
    WIDTH = 128
    HEIGHT = 70
    MAXSHAPEWIDTH = 10
    MAXSHAPEHEIGHT = 8
    AMOUNTRECT = 60
    AMOUNTPOLY = 10
    MAXCORNERS = 3

    BOUNDARYSPACING = 1.5
    MINDEADENDLENGTH = 8.0

class SMALL(BaseFactoryConf):
    WIDTH = 32
    HEIGHT = 18
    MAXSHAPEWIDTH = 3
    MAXSHAPEHEIGHT = 2
    AMOUNTRECT = 20
    AMOUNTPOLY = 0
    MAXCORNERS = 3

class SMALLSQUARE(BaseFactoryConf):
    WIDTH = 100
    HEIGHT = 100
    MAXSHAPEWIDTH = 20
    MAXSHAPEHEIGHT = 20
    AMOUNTRECT = 10
    AMOUNTPOLY = 0
    MAXCORNERS = 3

class EDF(BaseFactoryConf):
    WIDTH = 300
    HEIGHT = 30
    MAXSHAPEWIDTH = 5
    MAXSHAPEHEIGHT = 5
    AMOUNTRECT = 150
    AMOUNTPOLY = 0
    MAXCORNERS = 3

    BOUNDARYSPACING = 2.5
    MINDEADENDLENGTH = 5.0