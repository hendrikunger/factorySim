#!/usr/bin/env python3

class Point3D:
##58 = IFCCARTESIANPOINT((0., 200., 0.));
##71 = IFCCARTESIANPOINT((717., -6.409494854920721E-29, 1976.));
    def __init__(self, x, y, z):

        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return F"Point3D({self.x},{self.y})"
    
    def to_Tuple2D(self):
        return self.x, self.y

def main():
	pass

if __name__ == "__main__":
	main()