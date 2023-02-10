import networkx as nx
import numpy as np
from itertools import combinations
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union


DEBUG = False
class FactoryRating():

    def __init__(self, machine_dict=None, wall_dict=None, fullPathGraph=None, reducedPathGraph=None, prepped_bb=None, dfMF=None):

        self.machine_dict = machine_dict
        self.wall_dict = wall_dict
        self.fullPathGraph = fullPathGraph
        self.reducedPathGraph = reducedPathGraph
        self.prepped_bb = prepped_bb
        self.dfMF = dfMF
 #------------------------------------------------------------------------------------------------------------
    def PathWideVariance(self):
        min_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'pathwidth').values())))
        max_pathwidth = np.array(list((nx.get_edge_attributes(self.PathGraph,'max_pathwidth').values())))
        return np.mean(min_pathwidth/max_pathwidth)
 #------------------------------------------------------------------------------------------------------------
    def PathPolygon(self):
        polys = []
        if self.reducedPathGraph:
            for u,v,data in self.reducedPathGraph.edges(data=True):
                line = LineString(data["nodelist"])
                polys.append(line.buffer(data['pathwidth']/2))
        if polys:
            return MultiPolygon(polys)
        else:
            return MultiPolygon()
 #------------------------------------------------------------------------------------------------------------
    def FreeSpacePolygon(self):
        polys = []
        if self.fullPathGraph:
            pos=nx.get_node_attributes(self.fullPathGraph,'pos')

            for u,v,data in self.fullPathGraph.edges(data=True):
                line = LineString([pos[u],pos[v]])
                if data['true_pathwidth']:
                    polys.append(line.buffer(data['true_pathwidth']/2))
                else:
                    polys.append(line.buffer(data['pathwidth']/2))
        if polys:
            return MultiPolygon(polys)
        else:
            return MultiPolygon()
 #------------------------------------------------------------------------------------------------------------
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


 #------------------------------------------------------------------------------------------------------------
    def evaluateMF_Helper(self, source, sink): 
        source_center = self.machine_dict[source].center
        sink_center = self.machine_dict[sink].center
        return np.sqrt(np.power(source_center.x-sink_center.x,2) + np.power(source_center.y-sink_center.y,2))

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF(self, boundingBox):
        if len(self.dfMF.index) > 0:
            self.dfMF['distance'] = self.dfMF.apply(lambda row: self.evaluateMF_Helper(row['from'], row['to']), axis=1)
            #sum of all costs /  maximum intensity (intensity sum norm * 1) 
            #find longest distance possible in factory
            maxDistance = max(boundingBox.bounds[2],  boundingBox.bounds[3])
            self.dfMF['distance_norm'] = self.dfMF['distance'] / maxDistance
            self.dfMF['costs'] = self.dfMF['distance_norm'] * self.dfMF['intensity_sum_norm']
            output = 1 - (np.power(self.dfMF['costs'].sum(),2) / self.dfMF['intensity_sum_norm'].sum())
            if(output < 0): output = 0

            return output
        else:
            return 0


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


