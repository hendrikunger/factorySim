import networkx as nx
import numpy as np
from itertools import combinations
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, snap
from shapely.prepared import prep
import scipy.cluster.hierarchy as hcluster



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
    def PathWidthVariance(self):
        '''Calculates the Variance of the pathwidths for all subroutes between crossroads and deadends'''
        min_pathwidth = np.array(list((nx.get_edge_attributes(self.reducedPathGraph,'pathwidth').values())))
        max_pathwidth = np.array(list((nx.get_edge_attributes(self.reducedPathGraph,'max_pathwidth').values())))
        temp = np.power(min_pathwidth/max_pathwidth,2)
        if len(temp) > 0:
            return np.mean(temp)
        else:
            return 0
 #------------------------------------------------------------------------------------------------------------

    def PathPolygon(self):
        polys = []
        if self.reducedPathGraph and self.fullPathGraph:
            for u,v,data in self.reducedPathGraph.edges(data=True):
                if not data.get("isMachineConnection",False):
                    pos = nx.get_node_attributes(self.fullPathGraph,'pos')
                    line = LineString([pos[node] for node in data["nodelist"]])
                    polys.append(line.buffer(data['pathwidth']/2)) 
        if polys:
            walls = unary_union(list(x.poly for x in self.wall_dict.values()))
            wallpoints = walls.boundary.interpolate(100)
            pathPoly = snap(MultiPolygon(polys),wallpoints,400)
            pathPoly = self.makeMultiPolygon(pathPoly)
            extendedPathPoly = self.makeMultiPolygon(pathPoly.buffer(500))
            return pathPoly, extendedPathPoly
        else:
            return MultiPolygon(), MultiPolygon()
        
 #------------------------------------------------------------------------------------------------------------
    def getMachinesFarFromPath(self, extendedPathPoly):
        farMachines = set()
        preppedPath = prep(extendedPathPoly)
        for machine in self.machine_dict.values():
            if not preppedPath.intersects(machine.poly):
                farMachines.add(machine.gid) 
        return farMachines
 #------------------------------------------------------------------------------------------------------------
    def PathEfficiency(self):
        '''Calculates the sum of machines sidelengths if they where squares and devides it by the total length of paths'''
        if self.reducedPathGraph:
            machineSquares = np.sqrt(np.array([x.poly.area for x in self.machine_dict.values()]))
            subview = self.reducedPathGraph.subgraph([n for n in self.reducedPathGraph.nodes() if n not in self.machine_dict.keys()]) 
            return np.clip(machineSquares.sum()/subview.size(weight='weight'),0,1)
        else:
            return 0      
 #------------------------------------------------------------------------------------------------------------
    def FreeSpacePolygon(self, pathPolygon, walkableAreaPoly, usedSpacePolygonDict):

        temp = unary_union(walkableAreaPoly) - unary_union(pathPolygon) - unary_union(list(usedSpacePolygonDict.values()))
        temp = self.makeMultiPolygon(temp)

        maxArea = 0
        maxAreaIndex = None
        for index, polygon in enumerate(temp.geoms):
            if polygon.area > maxArea:
                maxArea = polygon.area
                maxAreaIndex = index

        if maxAreaIndex != None:
            growingSpacePolygon = self.makeMultiPolygon(temp.geoms[maxAreaIndex])
            temp = unary_union(walkableAreaPoly) - unary_union(pathPolygon) - unary_union(list(usedSpacePolygonDict.values())) - unary_union(growingSpacePolygon)

            return self.makeMultiPolygon(temp), growingSpacePolygon
        else:
            return MultiPolygon(), MultiPolygon()
 #------------------------------------------------------------------------------------------------------------
    def UsedSpacePolygon(self, threshold):           
        machineCenters = np.array([x.center.coords[0] for x in self.machine_dict.values()])
        if len(machineCenters) <= 1: 
            return {}, self.machine_dict
        
        clusters = hcluster.fclusterdata(machineCenters, threshold, criterion="distance")
        grouped = {value+1: [] for value in range(len(set(clusters)))}

        for clusterID, machine in zip(clusters, self.machine_dict.values()):
            machine.group = clusterID
            grouped[clusterID].append(machine.poly)

        hulls={}
        for key, value in grouped.items():
            hulls[key] = MultiPolygon([unary_union(value).convex_hull])


        return hulls, self.machine_dict
 #------------------------------------------------------------------------------------------------------------
    def FreeSpaceRoutesPolygon(self, pathPolygon):
        polys = []
        if self.fullPathGraph:
            pos=nx.get_node_attributes(self.fullPathGraph,'pos')

            for u,v,data in self.fullPathGraph.edges(data=True):
                line = LineString([pos[u],pos[v]])
                if data['true_pathwidth']:
                    polys.append(line.buffer(data['true_pathwidth']/2))
                else:
                    polys.append(line.buffer(data['pathwidth']/2))

        temp = unary_union(MultiPolygon(polys))-unary_union(pathPolygon)
        return self.makeMultiPolygon(temp)
 #------------------------------------------------------------------------------------------------------------
    def evaluateScalability(self, freeSpacePolygon):
        '''Compares the area of a polygon against the Area of the minimum bounding rectangle of the polygon.'''
        if freeSpacePolygon.area > 0:
            return np.clip(freeSpacePolygon.area / freeSpacePolygon.minimum_rotated_rectangle.area, 0, 1)
        else:
            return 0
 #------------------------------------------------------------------------------------------------------------
    def evaluateDeadends(self):
        '''Compares amount of deadends to amount of edges in simplified graph'''
        if self.reducedPathGraph:
            subview = self.reducedPathGraph.subgraph([n for n in self.reducedPathGraph.nodes() if n not in self.machine_dict.keys()])
            if len(subview.edges()) > 0:            
                return np.clip(1-(len([x for x in subview.nodes() if subview.degree(x) == 1])/(len(subview.edges()))),0,1)
            else:
                return 0
        else :
            return 0
 #------------------------------------------------------------------------------------------------------------
    def evaluateCompactness(self, usedSpacePolygonDict):
        '''Compares the area of a polygon against the area of a circle with the same area as the polygon.'''
        #Currently not used in the evaluation
        rating = []
        for poly in usedSpacePolygonDict.values():
            if poly.area > 0:
                rating.append(np.maximum(0, 2- ((poly.boundary.length/np.pi)/(2 * np.sqrt(poly.area/np.pi)))))
 
 #------------------------------------------------------------------------------------------------------------
    def evaluateAreaUtilisation(self, walkableAreaPoly, freeSpacePolygon):
        if walkableAreaPoly.area > 0:
            return 1-(freeSpacePolygon.area / walkableAreaPoly.area)
        else:
            return 0

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
                        if col.type in {"LineString", "MultiLineString","Point"}: continue
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
            self.dfMF['distance'] = self.dfMF.apply(lambda row: self.evaluateMF_Helper(row['source'], row['target']), axis=1)
            #sum of all costs /  maximum intensity (intensity sum norm * 1) 
            #find longest distance possible in factory
            maxDistance = max(boundingBox.bounds[2],  boundingBox.bounds[3])
            self.dfMF['distance_norm'] = self.dfMF['distance'] / maxDistance
            self.dfMF['costs'] = self.dfMF['distance_norm'] * self.dfMF['intensity_sum_norm']
            output = 1 - (np.power(self.dfMF['costs'].sum(),2) / self.dfMF['intensity_sum_norm'].sum())
            if(output < 0): output = 0

            return np.power(output,2)
        else:
            return 0

 #------------------------------------------------------------------------------------------------------------
    def evaluateTrueMF(self, boundingBox):
        if len(self.dfMF.index) > 0:
            #sum of all costs /  maximum intensity (intensity sum norm * 1) 
            #find longest distance possible in factory
            subview = self.reducedPathGraph.subgraph([n for n in self.reducedPathGraph.nodes() if n not in self.machine_dict.keys()]) 
            maxDistance = subview.size(weight='weight')
            self.dfMF['trueDistance_norm'] = self.dfMF['trueDistances'] / maxDistance
            self.dfMF['trueCosts'] = self.dfMF['trueDistance_norm'] * self.dfMF['intensity_sum_norm']
            output = 1 - (np.power(self.dfMF['trueCosts'].sum(),2) / self.dfMF['intensity_sum_norm'].sum())
            if(output < 0): output = 0

            return np.power(output,2)
        else:
            return 0
 #------------------------------------------------------------------------------------------------------------
    def evaluateMFIntersection(self):
        if len(self.dfMF.index) > 0:
            lines=[]
            intersections=[]
            #get coordinates of all start and end points of Material Flows and build list of Linestrings
            totalIntensity = self.dfMF['intensity_sum_norm'].sum()
            output = 0

            endpoints = [m.center for m in self.machine_dict.values()]

            #Check for intersections between all pairs of materialflow lines
            for row1, row2 in combinations(self.dfMF.itertuples(),2):
                line1=LineString([self.machine_dict[row1.source].center, self.machine_dict[row1.target].center])
                line2=LineString([self.machine_dict[row2.source].center, self.machine_dict[row2.target].center])
                if line1.intersects(line2):
                    inter= line1.intersection(line2)
                    #check if intersection is a point and if it is an endpoint of one of the lines
                    #if not, add to intersection list
                    found = None
                    if inter.geom_type == "Point":
                        for end in endpoints:
                            if end.equals_exact(inter, tolerance=0.01):
                                found = True
                                break
                        if not found: 
                            intersections.append(inter)
                            #calculate penalty for single intersection
                            output += (row1.intensity_sum_norm + row2.intensity_sum_norm) / 2 * totalIntensity
            #calulate penalty for all intersections 
            return np.clip(1-(output)/len(self.dfMF.index), 0,1), intersections
        else:
            return 0, []
 #------------------------------------------------------------------------------------------------------------
    def evaluateRouteContinuity(self):
        #angleList holds smallest angle in degrees between two edges (0-180)
        angleList = np.array(list(nx.get_node_attributes(self.fullPathGraph, "edge_angle").values()))
        #No bends == best Rating
        numBends = len(angleList)
        if numBends == 0: return 1
        #normalise to 0-1
        angleList = angleList/180
        #square to get a higher penalty for sharper bends
        angleList = np.power(angleList, 2)
        #Devide by count of bends in relation to number of simple edges, to penalize higher bend density
        #Penalizing having more bends then simple edges via maximum function
        angleList = angleList /  np.maximum(1, numBends/self.reducedPathGraph.number_of_edges())
        #Reduce influence of single bends if there are not many bends
        angleList = (1-(1/numBends)) * angleList + (1/numBends) #* 1        
        #Normalze to a sum of 1
        normed = angleList.sum() / numBends
        return normed
    
 #------------------------------------------------------------------------------------------------------------
    def evaluateRouteAccess(self, MachinesFarFromPath):
        if len(self.machine_dict) > 0:
            return 1-(len(MachinesFarFromPath) / len(self.machine_dict))
        else:
            return 0
 #------------------------------------------------------------------------------------------------------------
    def makeMultiPolygon(self, poly):
        if type(poly) == Polygon:
            return MultiPolygon([poly])
        elif type(poly) == MultiPolygon:
            return poly
        else:
            return MultiPolygon()

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



# 	                                    Vorhandensein eindeutiger Wegachsen
# 	                                    Kontaktflächen Wegenetz - Länge der Schnittkante zwischen erweitertem Wegpolygon und Maschine
# 2 - Medienverfügbarkeit	            Möglichkeit des Anschlusses von Maschinen an Prozessmedien (z.B. Wasser, Druckluft)
# 1 - Beleuchtung	                    Erfüllung der Arbeitsplatzanforderungen
# 2 - Ruhe	                            Erfüllung der Arbeitsplatzanforderungen
# 3 - Erschütterungsfreiheit	        Erfüllung der Arbeitsplatzanforderungen
# 4 - Sauberkeit	                    Erfüllung der Arbeitsplatzanforderungen
# 5 - Temperatur	                    Erfüllung der Arbeitsplatzanforderungen



# Erledigt =================================================================

# 	  Materialflusslänge                Entfernung (wegorientiert)
# 1 - Materialflusslänge	            Entfernung (direkt)
# 2 - Überschneidungsfreiheit	        Materialflussschnittpunkte
# 4 - Intensität	                    Anzahl der Transporte
# 5 - Wegekonzept	                    Auslegung Wegbreite
# 	                                    Sackgassen
#                                       Verwinkelung
# 	                                    Wegeeffizienz - Flächenbedarf der Wege im Vergleich zu den Flächenbedarf der Maschinen          -->evtl überarbeiten
# 6 - Zugänglichkeit	                Abdeckung Wegenetz
# 7 - Flächennutzungsgrad	            genutzte Fabrikfläche (ohne zusammenhängende Freifläche)
# 1 - Skalierbarkeit 	                Ausdehnung der größten verfügbaren Freifläche