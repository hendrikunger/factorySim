import networkx as nx
import numpy as np
from itertools import combinations
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union


DEBUG = False
class FactoryRating():

    def __init__(self, machine_dict=None, wall_dict=None, fullPathGraph=None, reducedPathGraph=None, prepped_bb=None):

        self.machine_dict = machine_dict
        self.wall_dict = wall_dict
        self.fullPathGraph = fullPathGraph
        self.reducedPathGraph = reducedPathGraph
        self.prepped_bb = prepped_bb

    def PathWideVariance(self):
        min_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'pathwidth').values())))
        max_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'max_pathwidth').values())))
        return np.mean(min_pathwidth/max_pathwidth)

    def PathPolygon(self):
        polys = []

        for u,v,data in self.reducedPathGraph.edges(data=True):
            line = LineString(data["nodelist"])
            polys.append(line.buffer(data['pathwidth']/2))

        return MultiPolygon(polys)

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

        #Find machines just outside the factory (rewardgaming)
        self.outsiderList = list(filter(self.prepped_bb.touches, [x.poly for x in self.machine_dict.values()]))
        self.outsiderList.extend(list(filter(self.prepped_bb.disjoint, [x.poly for x in self.machine_dict.values()])))

        return collisionAfterLastUpdate





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import descartes
    from tqdm import tqdm
    import factorySim.baseConfigs as baseConfigs
    from factorySim.factorySimClass import FactorySim

    SAVEPLOT = True
    SAVEFORMAT = "png"
    DETAILPLOT = True
    PLOT = True
    ITERATIONS = 1


    for runs in tqdm(range(ITERATIONS)):


        ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
            "..",
            "..",
            "Input",
            "2",  
            "TestCaseZigZag" + ".ifc")
   
        factory = FactorySim(ifcpath,
        path_to_materialflow_file = None,
        factoryConfig=baseConfigs.SMALLSQUARE,
        randomPos=False,
        createMachines=True,
        verboseOutput=0,
        maxMF_Elements=None
        )


