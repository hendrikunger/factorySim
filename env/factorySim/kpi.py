import networkx as nx
import numpy as np
from itertools import combinations
from shapely.geometry import  Polygon,  MultiPolygon


DEBUG = False
class FactoryRating():

    def __init__(self, machine_dict=None, wall_dict=None, fullPathGraph=None, PathGraph=None):

        self.machine_dict = machine_dict
        self.wall_dict = wall_dict
        self.fullPathGraph = fullPathGraph
        self.PathGraph = PathGraph

    def PathWideVariance(self):
        min_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'pathwidth').values())))
        max_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'max_pathwidth').values())))
        return np.mean(min_pathwidth/max_pathwidth)

    def findCollisions(self, lastUpdatedMachine=None):
        collisionAfterLastUpdate = False
        #Machines with Machines
        self.machineCollisionList = []       
        for a,b in combinations(self.machine_dict.values(), 2):
            if a.poly.intersects(b.poly):
                if(DEBUG):
                    print(f"Kollision Maschinen {a.name} und {b.name} gefunden.")
                col = a.poly.intersection(b.poly)
                if col.type != "MultiPolygon":
                    if col.type == "LineString" or col.type == "Point": continue
                    col = MultiPolygon([col])
                self.machineCollisionList.append(col)
                if(a.gid == lastUpdatedMachine or b.gid == lastUpdatedMachine): collisionAfterLastUpdate = True
        #Machines with Walls     
        self.wallCollisionList = []
        for a in self.wall_dict.values():
            for b in self.machine_dict.values():
                if a.poly.intersects(b.poly):
                    if(DEBUG):
                        print(f"Kollision Wand {a.name} und Maschine {b.name} gefunden.")
                    col = a.poly.intersection(b.poly)
                    if col.type != "MultiPolygon":
                        if col.type == "LineString" or col.type == "Point": continue
                        col = MultiPolygon([col])
                    self.wallCollisionList.append(col)
                    if(b.gid == lastUpdatedMachine): collisionAfterLastUpdate = True
        return collisionAfterLastUpdate
            