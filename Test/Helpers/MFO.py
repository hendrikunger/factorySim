#!/usr/bin/env python3

from Helpers.Point3D import Point3D
import random
import time
from Polygon import *


class Machine:

	def __init__(self, gid="not_set", name="no_name", origin_x=0, origin_y=0, origin_z=0, rotation=0 ):
		
		self.gid = gid
		self.name = name
		self.origin = Point3D(origin_x, origin_y, origin_z)
		self.rotation = rotation  #in Radians
		self.items = []
		self.loops = []
		self.points = []
		self.center = None
		self.color = [random.random(),random.random(),random.random()]
	
	def add_Point(self, x,y,z): 
		self.points.append(Point3D(x, y, z))

	def close_Loop(self):
		self.loops.append(self.points)
		self.points = []
	
	def close_Item(self):
		self.items.append(self.loops)
		self.loops = []
		self.points = []

	def scale_Points(self, xScale, yScale, zScale):
		self.origin.x *= xScale
		self.origin.y *= yScale
		self.origin.z *= zScale
		self.center.x *= xScale
		self.center.y *= yScale
		self.center.z *= zScale
		for item in self.items:
			for loop in item:
				for point in loop:
					point.x *= xScale
					point.y *= yScale
					point.z *= zScale
	
	def find_Center(self):
		x_min = float("inf")
		y_min = float("inf")
		x_max = 0
		y_max = 0

		for item in self.items:
			for loop in item:
				for point in loop:
					if (abs(point.x) < x_min): x_min = abs(point.x)
					if (abs(point.y) < y_min): y_min = abs(point.y)
					if (abs(point.x) > x_max): x_max = abs(point.x)
					if (abs(point.y) > y_max): y_max = abs(point.y)

		self.center = Point3D((x_max - x_min) / 2, (y_max - y_min) / 2, 0)
				


	def translate_Points(self, xTrans, yTrans, zTrans):
		self.origin.x += xTrans
		self.origin.y += yTrans
		self.origin.z += zTrans

def main():
	testmachine = Machine("AABBAA")
	complexmachine = Machine("COMPLEX", 1, 2, 3)

	complexmachine.add_Point(1,1,1)
	complexmachine.add_Point(2,2,2)
	complexmachine.add_Point(3,3,3)

	print(F"Testmachine: {testmachine.gid}, {testmachine.origin.x}, {testmachine.origin.y}, {testmachine.origin.z}")
	print(F"Testmachine: {complexmachine.gid}, {complexmachine.origin.x}, {complexmachine.origin.y}, {complexmachine.origin.z}")

	for point in complexmachine.points:
		print(F"{point.x}, {point.y}, {point.z}")

if __name__ == "__main__":
	main()

