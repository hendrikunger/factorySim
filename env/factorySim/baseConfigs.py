class BaseFactoryConf():
    NAME="BASE"
    WIDTH = 30000
    HEIGHT = 3000
    MAXSHAPEWIDTH = 500
    MAXSHAPEHEIGHT = 500
    AMOUNTRECT = 150
    AMOUNTPOLY = 0
    MAXCORNERS = 3

    MINDEADENDLENGTH = 2000 # If Deadends are shorter than this, they are deleted
    MINPATHWIDTH = 1000  # Minimum Width of a Road to keep
    MAXPATHWIDTH = 2500  # Maximum Width of a Road
    MINTWOWAYPATHWIDTH = 2000  # Minimum Width of a Road to keep
    BOUNDARYSPACING = 500  # Spacing of Points used as Voronoi Kernels
    SIMPLIFICATIONANGLE = 35 # Angle in degrees, used for support point calculation in simple path

    @classmethod
    def creationParameters(cls):
        return (cls.WIDTH, cls.HEIGHT), cls.MAXSHAPEWIDTH, cls.MAXSHAPEHEIGHT, cls.AMOUNTRECT, cls.AMOUNTPOLY, cls.MAXCORNERS

    @classmethod
    def pathParameters(cls):
        return cls.BOUNDARYSPACING, cls.MINDEADENDLENGTH, cls.MINPATHWIDTH, cls.MINTWOWAYPATHWIDTH, cls.SIMPLIFICATIONANGLE

    @classmethod
    def byStringName(cls, name):
        if name == "BIG":
            return BIG
        elif name == "SMALL":
            return SMALL
        elif name == "SMALLSQUARE":
            return SMALLSQUARE
        elif name == "EDF":
            return EDF
        elif name == "EDF_EMPTY":
            return EDF
        else:
            raise ValueError("Unknown Factory Configuration")    




class BIG(BaseFactoryConf):
    NAME="BIG"
    WIDTH = 128000
    HEIGHT = 70000
    MAXSHAPEWIDTH = 5000
    MAXSHAPEHEIGHT = 8000
    AMOUNTRECT = 30
    AMOUNTPOLY = 10
    MAXCORNERS = 3

    BOUNDARYSPACING = 1000
    MINDEADENDLENGTH = 4000

class SMALL(BaseFactoryConf):
    NAME="SMALL"
    WIDTH = 32000
    HEIGHT = 18000
    MAXSHAPEWIDTH = 2500
    MAXSHAPEHEIGHT = 1500
    AMOUNTRECT = 10
    AMOUNTPOLY = 0
    MAXCORNERS = 3

class SMALLSQUARE(BaseFactoryConf):
    NAME="SMALLSQUARE"
    WIDTH = 10000
    HEIGHT = 10000
    MAXSHAPEWIDTH = 2000
    MAXSHAPEHEIGHT = 2000
    AMOUNTRECT = 8
    AMOUNTPOLY = 3
    MAXCORNERS = 3

    BOUNDARYSPACING = 500

class EDF(BaseFactoryConf):
    NAME="EDF"
    WIDTH = 6000 #9,60m
    HEIGHT = 6700 #11,20m
    MAXSHAPEWIDTH = 1000
    MAXSHAPEHEIGHT = 1000
    AMOUNTRECT = 15
    AMOUNTPOLY = 3
    MAXCORNERS = 3

    BOUNDARYSPACING = 500
    #MINDEADENDLENGTH = 1000

class EDF_EMPTY(BaseFactoryConf):
    NAME="EDF_EMPTY"
    WIDTH = 6000 #9,60m
    HEIGHT = 6700 #10,84m
    MAXSHAPEWIDTH = 1000
    MAXSHAPEHEIGHT = 1000
    AMOUNTRECT = 2
    AMOUNTPOLY = 0
    MAXCORNERS = 3

    BOUNDARYSPACING = 500
    #MINDEADENDLENGTH = 1000
    