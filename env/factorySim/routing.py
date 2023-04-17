#%%
import time

from shapely.geometry import Point, MultiPoint, MultiPolygon, MultiLineString, GeometryCollection, LineString
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
from shapely.prepared import prep
from shapely.ops import split, voronoi_diagram,  unary_union, linemerge, nearest_points

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from math import dist

DETAILPLOT = False

class FactoryPath():

    fullPathGraph = None
    reducedPathGraph = None
    TIMING = False
    PLOTTING = False

    def __init__(self, boundarySpacing=150, minDeadEndLength=2000, minPathWidth=1000, maxPathWidth=2500, minTwoWayPathWidth=2000, simplificationAngle=35):

        self.minDeadEndLength = minDeadEndLength # If Deadends are shorter than this, they are deleted
        self.minPathWidth = minPathWidth  # Minimum Width of a Road to keep
        self.maxPathWidth = maxPathWidth # Maximum Width of a Road
        self.minTwoWayPathWidth = minTwoWayPathWidth  # Minimum Width of a Road to keep
        self.boundarySpacing = boundarySpacing # Spacing of Points used as Voronoi Kernels
        self.simplificationAngle = simplificationAngle # Angle in degrees, used for support point calculation in simple path
        self.fullPathGraph = nx.Graph() # initialize the graph
        self.reducedPathGraph = nx.Graph()# initialize the graph

    def timelog(self, text):
        self.nextTime = time.perf_counter()
        print(f"{text} {self.nextTime - self.startTime}")
        self.startTime = self.nextTime

    def calculateAll(self, machine_dict, wall_dict, bb):
        #Check if we have enough machines to make a path
        if len(machine_dict) <= 1:
            self.fullPathGraph = nx.Graph()
            self.reducedPathGraph = nx.Graph()
            return self.fullPathGraph, self.reducedPathGraph, MultiPolygon()
        
        machinelist = [x.poly for x in machine_dict.values()]
        union = unary_union(machinelist)
        if union.geom_type == "MultiPolygon":
            multi = MultiPolygon(union)
        elif union.geom_type == "Polygon":
            multi = MultiPolygon([union])
        elif union.geom_type == "GeometryCollection":
            multi = MultiPolygon()
        else:
            print("Error: No valid Polygon in Machine Dictionary")
            return self.fullPathGraph, self.reducedPathGraph, MultiPolygon() 

        walllist = [x.poly for x in wall_dict.values()]
        union = unary_union(walllist)
        if union.geom_type == "MultiPolygon":
            walls = MultiPolygon(union)
        elif union.geom_type == "Polygon":
            walls = MultiPolygon([union])
        elif union.geom_type == "GeometryCollection":
            walls = multi.boundary
        else:
            print("Error: No valid Polygon in Wall Dictionary")
            return self.fullPathGraph, self.reducedPathGraph, MultiPolygon()


        #Scale boundary spacing according to factory size
        bbox = bb.bounds
        scale = max((bbox[2]-bbox[0]),(bbox[3]-bbox[1])) / 30
        scale = 1

        if self.TIMING:
            self.startTime = time.perf_counter()
            self.totalTime = self.startTime

        machinesAndwalls = unary_union(machinelist + walllist)
        walkableArea = bb.difference(machinesAndwalls)
        if walkableArea.geom_type ==  'MultiPolygon':
            walkableArea = walkableArea.geoms[0]


#   Create Voronoi -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Points around boundary

        distances = np.arange(0,  walls.boundary.length, self.boundarySpacing * scale)
        points = [walls.boundary.interpolate(distance) for distance in distances]

        #Points on Machines
        distances = np.arange(0,  multi.boundary.length, self.boundarySpacing * scale)
        points.extend([multi.boundary.interpolate(distance) for distance in distances])
        bb_points = unary_union(points) 

        if self.TIMING: self.timelog("Boundary generation")

        voronoiBase = GeometryCollection([walkableArea, bb_points])
        try:
            voronoiArea = voronoi_diagram(voronoiBase, edges=True, envelope=bb)
        except:
            print("Error: Could not create Voronoi Diagram")
            return self.fullPathGraph, self.reducedPathGraph, MultiPolygon()

        if self.TIMING: self.timelog("Voronoi")

        self.route_lines = []
        self.lines_touching_machines = []
        self.lines_to_machines = []

        processed_multi = prep(multi)
        processed_machinesAndwalls = prep(machinesAndwalls)

        for line in voronoiArea.geoms[0].geoms:
            #find routes close to machines
            if not (processed_machinesAndwalls.intersects(line)): 
                self.route_lines.append(line)
            else:
                self.lines_touching_machines.append(line)

        if self.TIMING: self.timelog("Find Routes")

        if DETAILPLOT:

            #Split lines with machine objects#
            try:
                sresult = split(MultiLineString(self.lines_touching_machines), multi)
            except:
                print("Split Error")         

            if self.TIMING: self.timelog("Split")

            #Remove Geometries that are inside machines
            for line in sresult.geoms:
                if  not (processed_multi.covers(line) and (not processed_multi.disjoint(line) ) or processed_multi.crosses(line)):
                    self.lines_to_machines.append(line)

            if self.TIMING: self.timelog("Line Filtering")

        #Simplify Lines
        self.route_lines = linemerge(self.route_lines)
        #self.simple_route_lines = self.route_lines.simplify(self.boundarySpacing/2, preserve_topology=False)



        # Find closest points in voronoi cells
        if walkableArea.geom_type ==  'MultiPolygon':
            exteriorPoints = []
            for x in walkableArea.geoms:
                exteriorPoints.extend(list(x.exterior.coords))
        else:
            exteriorPoints = list(walkableArea.exterior.coords)
        self.hitpoints = points + list(MultiPoint(exteriorPoints).geoms)
        #hitpoints = MultiPoint(points+list(walkableArea.exterior.coords))
        self.hit_tree = STRtree(self.hitpoints)


# Create Graph -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.fullPathGraph = nx.Graph()
        for index, line in enumerate(self.route_lines):
            lastPoint = None
            currentPath = [] 
            for currentPoint in line.coords:
                currentPoint = Point(currentPoint)
                if lastPoint:
                    currentBoundaryPoint = self.hit_tree.nearest_geom(currentPoint)
                    currentPathWidth = currentPoint.distance(currentBoundaryPoint) * 2

                    lastPointTuple = (lastPoint.x, lastPoint.y)
                    lastPoint_str = str(lastPointTuple)

                    currentPointTuple = (currentPoint.x, currentPoint.y)
                    currentPoint_str = str(currentPointTuple)

                    self.fullPathGraph.add_node(
                        currentPoint_str,
                        pos=currentPointTuple,
                        pathwidth=currentPathWidth,
                        routeIndex=index
                        )

                    edgePathWidth = min(currentPathWidth, lastPathWidth)

                    self.fullPathGraph.add_edge(
                        lastPoint_str,
                        currentPoint_str,
                        weight=currentPoint.distance(lastPoint),
                        pathwidth=edgePathWidth if edgePathWidth <= self.maxPathWidth else self.maxPathWidth,
                        true_pathwidth=edgePathWidth if edgePathWidth > self.maxPathWidth else None,
                        routeIndex=index
                        )
                    currentPath.append((lastPoint_str,currentPoint_str,currentPoint.distance(lastPoint)))
                    lastPoint = currentPoint
                    lastBoundaryPoint = currentBoundaryPoint
                    lastPathWidth = currentPathWidth
                else:
                    lastPoint = currentPoint
                    lastBoundaryPoint = self.hit_tree.nearest_geom(currentPoint)
                    lastPathWidth = currentPoint.distance(lastBoundaryPoint) * 2
                    lastPointTuple = (lastPoint.x, lastPoint.y)
                    lastPoint_str = str(lastPointTuple)

                    self.fullPathGraph.add_node(lastPoint_str, pos=lastPointTuple, pathwidth=lastPathWidth, routeIndex=index)
                    continue

        if self.TIMING: self.timelog("Network generation")

# Filter  Graph -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Cleans road network created with voronoi method by 
        # - removing elements that are narrower than min_pathwidth
        # - removing any dangeling parts that might have been cut off
        # - removing all dead end that are shorter than min_length


        if self.PLOTTING: self.inter_unfilteredGraph = self.fullPathGraph.copy()

        self.narrowPaths = [(n1, n2) for n1, n2, w in self.fullPathGraph.edges(data="pathwidth") if w < self.minPathWidth]
        self.fullPathGraph.remove_edges_from(self.narrowPaths)

        #Find largest connected component to filter out "loose" parts
        Gcc = sorted(nx.connected_components(self.fullPathGraph), key=len, reverse=True)
        if (len(Gcc) > 0):
            self.fullPathGraph = self.fullPathGraph.subgraph(Gcc[0]).copy()

        # #find crossroads
        self.old_crossroads = [node for node, degree in self.fullPathGraph.degree() if degree >= 3]
        # #Set isCrossroads attribute on cross road nodes
        # nx.set_node_attributes(self.fullPathGraph, dict.fromkeys(self.old_crossroads, True), 'isCrossroads')
        # #find deadends
        self.old_endpoints = [node for node, degree in self.fullPathGraph.degree() if degree == 1]


        if self.TIMING: self.timelog("Network Filtering")

# Connect Machines to Network  -------------------------------------------------------------------------------------------------------------------------------------------------- 

        #Create KDTree for fast nearest neighbor search from node positions
        pos=nx.get_node_attributes(self.fullPathGraph,'pos')
        tree = KDTree(np.array(list(pos.values())))
        #Add machine center nodes to graph
        self.fullPathGraph.add_nodes_from([(k, {"pos":[v.center.x, v.center.y], "isMachineConnection":True}) for k, v in machine_dict.items()])

        #Find clostest node to machine center for every machine center
        distances, indexes = tree.query([[v.center.x, v.center.y] for v in machine_dict.values()], k=1)

        # print(np.array(list(pos.values())))
        # print(list(machine_dict.keys()))
        # print(indexes)

        for index, distance, machine in zip(indexes, distances, machine_dict.keys()):
            #Get the closest node
            closest_node = list(pos.keys())[index]

            #Add edge between the two
            self.fullPathGraph.add_edge(machine, 
                                        closest_node, 
                                        weight=distance,
                                        pathwidth=self.fullPathGraph.nodes[closest_node]["pathwidth"],
                                        true_pathwidth=self.fullPathGraph.nodes[closest_node]["pathwidth"],
                                        isMachineConnection=True,  
                                        )

        #Update positions of nodes
        pos = nx.get_node_attributes(self.fullPathGraph,'pos')
        self.endpoints = [node for node, degree in self.fullPathGraph.degree() if degree == 1]
        self.crossroads = [node for node, degree in self.fullPathGraph.degree() if degree >= 3]

        if self.TIMING: self.timelog("Machine Connection Calculation")

# Prune unused dead ends  -------------------------------------------------------------------------------------------------------------------------------------------------- 

        if self.PLOTTING: self.inter_filteredGraph = self.fullPathGraph.copy()

        for i in range(2):
            #Endpoints that are machine connections should not be pruned
            endpoints_to_prune = [endpoint for endpoint in self.endpoints if not self.fullPathGraph.nodes[endpoint].get("isMachineConnection",False)]

            if endpoints_to_prune:
                self.shortDeadEnds = self.pruneAlongPath(self.fullPathGraph, starts=endpoints_to_prune, ends=self.crossroads, min_length=3 * self.minDeadEndLength)
                if self.shortDeadEnds: self.fullPathGraph.remove_nodes_from(self.shortDeadEnds)

            
            self.endpoints = [node for node, degree in self.fullPathGraph.degree() if degree == 1]
            self.crossroads = [node for node, degree in self.fullPathGraph.degree() if degree >= 3]
            # #Set isCrossroads attribute on cross road nodes
            nx.set_node_attributes(self.fullPathGraph, dict.fromkeys(self.crossroads, True), 'isCrossroads')

        if self.TIMING: self.timelog("Dead End Pruning")

        



# Simpicification and Path Generation --------------------------------------------------------------------------------------------------------------------------------------------------

        self.reducedPathGraph = nx.Graph()

        visited = set() # Set to keep track of visited nodes.
        tempPath = [] # List to keep track of visited nodes in current path.

        ep = self.endpoints
        cross = self.crossroads
        stoppers = set(ep + cross + self.old_endpoints)

        if ep: 
            nodes_to_visit = [ep[0]]
        elif cross:
            nodes_to_visit = [cross[0]]
        else:
            nodes_to_visit = []

        maxpath = 0
        minpath = float('inf')
        totalweight = 0
        currentInnerNode = None

        #DFS Start
        while(nodes_to_visit):

            currentOuterNode = nodes_to_visit.pop()
            if currentOuterNode in visited:
                continue
            else:
                visited.add(currentOuterNode)

            for outerNeighbor in self.fullPathGraph.neighbors(currentOuterNode):
                if outerNeighbor in visited: continue

                maxpath = 0
                minpath = float('inf')
                totalweight = 0

                lastNode = currentOuterNode
                currentInnerNode = outerNeighbor
                tempPath.append(currentOuterNode)

                while True:
                    #check if next node is deadend or crossroad
                    currentEdgeKey = str((lastNode,currentInnerNode))
                    totalweight += self.fullPathGraph[lastNode][currentInnerNode]["weight"]
                    pathwidth = self.fullPathGraph[lastNode][currentInnerNode]["pathwidth"]
                    maxpath = max(maxpath, pathwidth)
                    minpath = min(minpath, pathwidth)
                   

                    if currentInnerNode in stoppers:
                        #found a crossroad or deadend
                        tempPath.append(currentInnerNode)
                        #tempPath = self.filterZigZag(tempPath, pos) 
                        tempPath = {i:(node,pos[node]) for i, node in enumerate(tempPath)}
                        tempPath = self.filterZigZag(tempPath, self.boundarySpacing * 5) 
                        #Prevent going back and forth between direct connected crossroads 
                        if lastNode != currentOuterNode:
                            visited.add(lastNode)
                        nodes_to_visit.append(currentInnerNode)

                        pathtype = "oneway"
                        if minpath > self.minTwoWayPathWidth: pathtype = "twoway"

                        if "isMachineConnection" in self.fullPathGraph.nodes[currentOuterNode] or "isMachineConnection" in self.fullPathGraph.nodes[currentInnerNode]:
                            machineConnection = True
                        else:
                            machineConnection = False

                        self.reducedPathGraph.add_node(currentOuterNode, pos=pos[currentOuterNode])
                        self.reducedPathGraph.add_node(currentInnerNode, pos=pos[currentInnerNode])
                        self.reducedPathGraph.add_edge(currentOuterNode, 
                            currentInnerNode, 
                            weight=totalweight,
                            pathwidth=minpath, 
                            max_pathwidth=maxpath, 
                            nodelist=tempPath,
                            pathtype=pathtype,
                            isMachineConnection=machineConnection
                        )

                        tempPath = [] 
                        break 
                    else:
                        #going along path
                        tempPath.append(currentInnerNode)

                    for innerNeighbor in self.fullPathGraph.neighbors(currentInnerNode):
                        #Identifying next node (there will at most be two edges connected to every node)
                        if (innerNeighbor == lastNode):
                            #this is last node
                            continue
                        else:
                            #found the next one
                            lastNode = currentInnerNode
                            currentInnerNode = innerNeighbor
                            break

        #Calculate support information
        nx.set_node_attributes(self.fullPathGraph, self.calculateNodeAngles())
        #Currently not necessary since rdp takes care of this
        #nx.set_node_attributes(self.fullPathGraph, self.findSupportNodes(cutoff=self.simplificationAngle))
        #self.support = list(nx.get_node_attributes(self.fullPathGraph, "isSupport").keys())

        if self.TIMING: 
            self.timelog("Network Path Generation")
            print(f"Algorithm Total: {self.nextTime - self.totalTime}")

        
        return self.fullPathGraph, self.reducedPathGraph, walkableArea
    








# Support Functions    --------------------------------------------------------------------------------------------------------------------------------------------------
    def calculateNodeAngles(self):
        node_data ={}
        pos=nx.get_node_attributes(self.fullPathGraph, 'pos')

        for u,v,data in self.reducedPathGraph.edges(data=True):

            for index, node in enumerate(data['nodelist'][1:-1]):
                node = str(node)
                neighbors = [data['nodelist'][index], data['nodelist'][index+2]]
                
                vector_1 = np.array(pos[neighbors[0]]) -np.array(pos[node])
                vector_2 = np.array(pos[neighbors[1]]) - np.array(pos[node])

                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.rad2deg(np.arccos(np.clip(dot_product, -1.0, 1.0)))

                theta1 = np.arctan2(vector_1[1],vector_1[0])
                theta2 = np.arctan2(vector_2[1],vector_2[0])
                
                vector_1 = np.array(pos[node]) - np.array(pos[neighbors[0]])
                gamma1 = np.arctan2(vector_1[1],vector_1[0])
                gamma2 = np.arctan2(vector_2[1],vector_2[0])

                if gamma1 > gamma2:
                    theta1, theta2 = theta2, theta1

                node_data[node] = {"edge_angle" : angle, "arcstart" : theta1, "arcend" : theta2, "gamma1" : gamma1, "gamma2" : gamma2}


        return node_data


    def findSupportNodes(self, cutoff=45):

        candidates = [node for node, degree in self.fullPathGraph.degree() if degree == 2]

        node_data ={}
        pos=nx.get_node_attributes(self.fullPathGraph,'pos')

        for node in candidates: 
            direct_neighbors = list(self.fullPathGraph.neighbors(node))
            
            pre = list(self.fullPathGraph.neighbors(direct_neighbors[0]))
            pre.remove(node)

            if len(pre) == 1:
                vector_1 = [pos[node][0] - pos[pre[0]][0], pos[node][1] - pos[pre[0]][1]]
            else:
                #direct_neighbors[0] is a endpoint or crossroad, can not use its neighbors for calculation
                vector_1 = [pos[node][0] - pos[direct_neighbors[0]][0], pos[node][1] - pos[direct_neighbors[0]][1]]

            suc = list(self.fullPathGraph.neighbors(direct_neighbors[1]))
            suc.remove(node)

            if len(suc) == 1:
                vector_2 = [pos[suc[0]][0] - pos[node][0], pos[suc[0]][1] - pos[node][1]]
            else:
                #direct_neighbors[1] is a endpoint or crossroad, can not use its neighbors for calculation
                vector_2 = [pos[direct_neighbors[1]][0] - pos[node][0], pos[direct_neighbors[1]][1] - pos[node][1]]

            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.clip(np.dot(unit_vector_1, unit_vector_2),-0.99999999,0.99999999)
            angle = np.rad2deg(np.arccos(dot_product))


            if(abs(angle) > cutoff):
                node_data[node] = {"isSupport" : True}

        for node in dict(node_data):
            if node_data[node].get("isSupport", False):
                direct_neighbors = self.fullPathGraph.neighbors(node)
                for neighbor in direct_neighbors:
                    if neighbor in node_data:
                        if node_data[neighbor].get("isSupport", False):
                            node_data.pop(node, None)

        return node_data


    def pruneAlongPath(self, F, starts=[], ends=[], min_length=1):
        shortDeadEnds =[]

        for seed in starts:

            total_length = 0
            currentNode = seed
            nextNode = None
            lastNode = None
            tempDeadEnds = []

            #check if there is something to do
            if len(F.edges())<=1:
                return []

            #Follow path from endnode to next crossroads, track length of path
            while True:

                nextNode = None
                for neighbor in F.neighbors(currentNode):
                    #Identifying next node (there will at most be two edges connected to every node)
                    if (neighbor == lastNode):
                        #this is last node
                        continue
                    else:
                        #found the next one
                        nextNode = neighbor
                        break
                # keep track of route length
                if nextNode:
                    total_length += F.edges[currentNode, nextNode]["weight"]
                #Stop if route is longer than min_length
                if total_length > min_length:
                    break

                if nextNode in ends or nextNode is None:
                    tempDeadEnds.append(currentNode)
                    shortDeadEnds.extend(tempDeadEnds)
                    break 
                else:
                    tempDeadEnds.append(currentNode)
                    lastNode = currentNode
                    currentNode = nextNode
        
        return shortDeadEnds


    def filterZigZag(self, d, epsilon):
        """
        Simplify a curve represented by a dict of node IDs and (x, y) coordinates using
        the Ramer-Douglas-Peucker algorithm.

        :param d: a dictionary with node IDs as keys and (x, y) coordinates as values
        :param epsilon: the maximum distance between a curve point and its approximation
        :return: a new dictionary with a subset of the input nodes that approximate the curve
        """
        # Create a new dictionary with integer keys and the same values as the original dictionary

        
        def rdp_recursive(start, end):
            """
            Simplify a curve segment represented by two indices using the Ramer-Douglas-Peucker algorithm.

            :param start: the index of the start node of the segment
            :param end: the index of the end node of the segment
            :return: a list of indices that approximate the segment
            """
            if end - start == 1:
                # Base case: only two nodes, return both
                return [start, end]
            elif end <= start:
                # Degenerate case: no nodes
                return []
            else:
                # Compute the perpendicular distance of all nodes to the line segment
                dmax = -1
                idx_max = -1
                for i in range(start + 1, end):
                    # dict is {0: (node_id, (x,y)), 1: (node_id, (x,y)), ...}
                    p0 = (d[start][1][0], d[start][1][1])
                    p1 = (d[end][1][0], d[end][1][1])
                    p = (d[i][1][0], d[i][1][1])
                    d1 = dist(p, p0)
                    d2 = dist(p, p1)
                    d0 = abs(d1 * (p1[1] - p0[1]) - (p1[0] - p0[0]) * d2) / dist(p1, p0)
                    if d0 > dmax:
                        dmax = d0
                        idx_max = i
                if dmax <= epsilon:
                    # All nodes are close enough to the line segment, return only the start and end nodes
                    return [start, end]
                else:
                    # Recursively simplify the left and right segments
                    left = rdp_recursive(start, idx_max)
                    right = rdp_recursive(idx_max, end)
                    # Combine the left and right segments
                    return left + right[1:]

        # Call the recursive function with the indices of the first and last nodes
        indices = rdp_recursive(0, len(d) - 1)
        # Map the integer indices back to their corresponding dictionary keys
        return [list(d.values())[i][0] for i in indices]







# Shortest Path Calculation   --------------------------------------------------------------------------------------------------------------------------------------------------





    def calculateRoutes(self, dfMF):
        """
        Calculates the routes for the given dataframe of machines and returns a list of routes and distances
        """   
        dfMF['routes'] = dfMF.apply(lambda x:nx.shortest_path(self.reducedPathGraph, source=x.source, target=x.target, weight='weight', method='dijkstra'), axis=1)
        dfMF['trueDistances'] = dfMF.apply(lambda x:nx.path_weight(self.reducedPathGraph, x.routes, weight='weight'), axis=1)

        return dfMF









#%% TESTS --------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import descartes
    from tqdm import tqdm
    from factorySim.creation import FactoryCreator
    from factorySim.kpi import FactoryRating
    import factorySim.baseConfigs as baseConfigs

    SAVEPLOT = False
    SAVEFORMAT = "svg"
    DETAILPLOT = False
    ITERATIONS = 1
    LINESCALER = 25

    rng = np.random.default_rng()

    for runs in tqdm(range(ITERATIONS)):
        factoryCreator = FactoryCreator(*baseConfigs.SMALLSQUARE.creationParameters())

        
        ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
            "..",
            "..",
            "Input",
            "FAIM2023" + ".ifc")
   
        ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
            "..",
            "..",
            "Input",
            "2",  
            "TestCaseZigZag" + ".ifc")

 
        wall_dict = factoryCreator.load_ifc_factory(ifcpath, "IFCWALL", recalculate_bb=True)
        bb = factoryCreator.bb  
        machine_dict = factoryCreator.create_factory()
        #machine_dict = factoryCreator.load_ifc_factory(ifcpath, "IFCBUILDINGELEMENTPROXY", recalculate_bb=False)

        multi = MultiPolygon(unary_union([x.poly for x in machine_dict.values()]))

        machine_colors = [rng.random(size=3) for _ in multi.geoms]
        factoryPath = FactoryPath(boundarySpacing=500, minDeadEndLength=2000, minPathWidth=1000, maxPathWidth=2500, minTwoWayPathWidth=2000, simplificationAngle=35)
        factoryPath.TIMING = True
        factoryPath.PLOTTING = True
        factoryPath.calculateAll(machine_dict, wall_dict, bb)
        pos=nx.get_node_attributes(factoryPath.fullPathGraph,'pos')

           
        factoryRating = FactoryRating(machine_dict=factoryCreator.machine_dict, wall_dict=factoryCreator.wall_dict, fullPathGraph=factoryPath.fullPathGraph, reducedPathGraph=factoryPath.reducedPathGraph, prepped_bb=factoryCreator.prep_bb)
        pathPoly = factoryRating.PathPolygon()

        if factoryPath.PLOTTING:
            pos_u=nx.get_node_attributes(factoryPath.inter_unfilteredGraph,'pos')
    #  Filtered_Lines Plot -----------------------------------------------------------------------------------------------------------------
            if DETAILPLOT:
                print("1 - Filtered_Lines")
                fig, ax = plt.subplots(1,figsize=(8, 8))
                plt.xlim(0,bb.bounds[2])
                plt.ylim(0,bb.bounds[3])
                plt.gca().invert_yaxis()
                plt.autoscale(False)
                plt.axis('off')

                for wall in wall_dict.values():
                    ax.add_patch(descartes.PolygonPatch(wall.poly, fc="darkgrey", ec='#000000', alpha=0.5))

                if multi.geom_type ==  'Polygon':
                    ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
                else:
                    for j, poly in enumerate(multi.geoms):
                        ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))
                for line in factoryPath.route_lines:
                    ax.plot(line.xy[0], line.xy[1], color='dimgrey', linewidth=3)
                for line in factoryPath.lines_touching_machines:
                    ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.5)
                for line in factoryPath.lines_to_machines:
                    ax.plot(line.xy[0], line.xy[1], color='red', alpha=0.9)

                # for point in bb_points:
                #     ax.scatter(point.xy[0], point.xy[1], color='red')
                #ax.add_patch(descartes.PolygonPatch(allEdges, fc='blue', ec='#000000', alpha=0.5))  
                if SAVEPLOT: plt.savefig(f"{runs+1}_1_Filtered_Lines.{SAVEFORMAT}", format=SAVEFORMAT, bbox_inches='tight', transparent=True)
                plt.show()

            # Pathwidth_Calculation Plot -----------------------------------------------------------------------------------------------------------------
            if DETAILPLOT:
                print("2 - Pathwidth_Calculation")
                fig, ax = plt.subplots(1,figsize=(8, 8))
                plt.xlim(0,bb.bounds[2])
                plt.ylim(0,bb.bounds[3])
                plt.gca().invert_yaxis()
                plt.autoscale(False)
                plt.axis('off')
               

                for wall in wall_dict.values():
                    ax.add_patch(descartes.PolygonPatch(wall.poly, fc="darkgrey", ec='#000000', alpha=0.5))

                for point in factoryPath.hitpoints:
                    ax.scatter(point.x, point.y, color='red')

                nx.draw_networkx_edges(factoryPath.inter_unfilteredGraph, pos=pos_u, ax=ax, edge_color="dimgrey", width=2)
                for line in factoryPath.route_lines:
                    for point in line.coords[::2]:
                        point = Point(point)
                        # Plot Circle for every line Endpoint, since Startpoint is likely connected to other line segment
                        nearest_point = factoryPath.hit_tree.nearest_geom(point)
                        #ax.plot([point.x, nearest_point.x], [point.y, nearest_point.y], color='green', alpha=1)
                        ax.add_patch(plt.Circle((point.x , point.y), point.distance(nearest_point), color='blue', fill=False, alpha=0.6))
                        #ax.add_patch(descartes.PolygonPatch(line.buffer(1), fc="black", ec='#000000', alpha=0.5))
                
                if multi.geom_type ==  'Polygon':
                    ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
                else:
                    for j, poly in enumerate(multi.geoms):
                        ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.8))           

                if SAVEPLOT: plt.savefig(f"{runs+1}_2_Pathwidth_Calculation.{SAVEFORMAT}", format=SAVEFORMAT, bbox_inches='tight', transparent=True)
                plt.show()

            #  Filtering Plot -----------------------------------------------------------------------------------------------------------------
            
            print("3 - Pruning")
            fig, ax = plt.subplots(1,figsize=(8, 8))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.gca().invert_yaxis()
            plt.autoscale(False)
            plt.axis('off')

            for wall in wall_dict.values():
                    ax.add_patch(descartes.PolygonPatch(wall.poly, fc="darkgrey", ec='#000000', alpha=0.5))

            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))

            pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.inter_unfilteredGraph,'pathwidth').values())))
            pos_f = nx.get_node_attributes(factoryPath.inter_filteredGraph,'pos')
            nx.draw_networkx_edges(factoryPath.inter_unfilteredGraph, pos=pos_u, ax=ax, edge_color="silver", width=pathwidth/LINESCALER, alpha=0.6)
            nx.draw_networkx_edges(factoryPath.inter_unfilteredGraph, pos=pos_u, ax=ax, edge_color="red", width=2, alpha=1)
            nx.draw_networkx_edges(factoryPath.inter_filteredGraph, pos=pos_f, ax=ax, edge_color="lime", width=2, alpha=1)
            nx.draw_networkx_edges(factoryPath.fullPathGraph, pos=pos, ax=ax, edge_color="dimgrey", width=5, alpha=1)
            nx.draw_networkx_edges(factoryPath.inter_unfilteredGraph, pos=pos_u, ax=ax, edgelist=factoryPath.narrowPaths, edge_color="blue", width=2, alpha=1)


            nx.draw_networkx_nodes(factoryPath.inter_unfilteredGraph, pos=pos_u, ax=ax, nodelist=factoryPath.shortDeadEnds, node_size=80, node_color='white', alpha=0.6, linewidths=4, edgecolors='green')
            nx.draw_networkx_nodes(factoryPath.inter_unfilteredGraph, pos=pos_u, ax=ax, nodelist=factoryPath.old_endpoints, node_size=150, node_color='green')
            nx.draw_networkx_nodes(factoryPath.inter_unfilteredGraph, pos=pos_u, ax=ax, nodelist=factoryPath.old_crossroads, node_size=150, node_color='white', alpha=0.6, linewidths=4, edgecolors='red')

            if SAVEPLOT: plt.savefig(f"{runs+1}_3_Pruning.{SAVEFORMAT}", format=SAVEFORMAT, bbox_inches='tight', transparent=True)
            
            plt.show()
            
            #  Clean Plot -----------------------------------------------------------------------------------------------------------------

            print("4 - Clean")
            fig, ax = plt.subplots(1,figsize=(8, 8))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.gca().invert_yaxis()
            plt.autoscale(False)
            plt.axis('off')

            for wall in wall_dict.values():
                ax.add_patch(descartes.PolygonPatch(wall.poly, fc="darkgrey", ec='#000000', alpha=0.5))


            for u,v,data in factoryPath.reducedPathGraph.edges(data=True):
                nodelist = [pos[node] for node in data["nodelist"]]
                ax.plot(*zip(*nodelist), color="dimgray", linewidth=data['pathwidth']/LINESCALER, alpha=1.0, solid_capstyle='round')


            for u,v,data in factoryPath.reducedPathGraph.edges(data=True):
                if data['pathtype'] =="twoway":
                    nodelist = [pos[node] for node in data["nodelist"]]
                    ax.plot(*zip(*nodelist), color="white", linewidth=3, alpha=0.5, solid_capstyle='round', linestyle='dashed')

            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.crossroads, node_size=120, node_color='red')
            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.endpoints, node_size=120, node_color='blue')
            #nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.support, node_size=120, node_color='green')
            
            #ax.add_patch(descartes.PolygonPatch(pathPoly, fc='red', ec='#000000', alpha=0.9))

            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5, zorder = 2))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5, zorder = 2))

            if SAVEPLOT: plt.savefig(f"{runs+1}_4_Clean.{SAVEFORMAT}", format=SAVEFORMAT, bbox_inches='tight', transparent=True)

            plt.show()

            #  Simplification Plot -----------------------------------------------------------------------------------------------------------------
            print("5 - Simplification")

            fig, ax = plt.subplots(1,figsize=(8, 8))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.gca().invert_yaxis()
            plt.autoscale(False)
            plt.axis('off')

            for wall in wall_dict.values():
                    ax.add_patch(descartes.PolygonPatch(wall.poly, fc="darkgrey", ec='#000000', alpha=0.5))

            # for u,v,a in factoryPath.reducedPathGraph.edges(data=True):
            #     linecolor = rng.random(size=3)
            #     for i, node in enumerate(data['nodelist'][1:]):
            #         ax.plot(*data['nodelist'][i-1], *node, color=linecolor, linewidth=50)

            for u,v,data in factoryPath.reducedPathGraph.edges(data=True):
                linecolor = rng.random(size=3)
                nodelist = [pos[node] for node in data["nodelist"]]
                ax.plot(*zip(*nodelist), color=linecolor, linewidth=5)

            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))


            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, node_size=20, node_color='black')
            nx.draw_networkx_nodes(factoryPath.reducedPathGraph, pos=pos, ax=ax, node_size=120, node_color='red')
            #nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.support, node_size=120, node_color='green')


            nx.draw_networkx_edges(factoryPath.fullPathGraph, pos=pos, ax=ax, edge_color="dimgray", alpha=0.5)

            min_pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.reducedPathGraph,'pathwidth').values())))
            max_pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.reducedPathGraph,'max_pathwidth').values())))

            nx.draw_networkx_edges(factoryPath.reducedPathGraph, pos=pos, ax=ax, edge_color="yellow", width=4)
            nx.draw_networkx_edges(factoryPath.reducedPathGraph, pos=pos, ax=ax, width=max_pathwidth/LINESCALER, edge_color="dimgrey", alpha=0.7)
            nx.draw_networkx_edges(factoryPath.reducedPathGraph, pos=pos, ax=ax, width=min_pathwidth/LINESCALER, edge_color="blue", alpha=0.5)
            nx.draw_networkx_edges(factoryPath.reducedPathGraph, pos=pos, ax=ax, edge_color="yellow", width=4)

            if SAVEPLOT: plt.savefig(f"{runs+1}_5_Simplification.{SAVEFORMAT}", format=SAVEFORMAT, bbox_inches='tight', transparent=True)
            plt.show()

            #  Closest Edge Plot -----------------------------------------------------------------------------------------------------------------
            if DETAILPLOT:
                print("6 - Closest_Edge")
                fig, ax = plt.subplots(1,figsize=(8, 8))
                plt.xlim(0,bb.bounds[2])
                plt.ylim(0,bb.bounds[3])
                plt.gca().invert_yaxis()
                plt.autoscale(False)
                plt.axis('off')

                for wall in wall_dict.values():
                    ax.add_patch(descartes.PolygonPatch(wall.poly, fc="darkgrey", ec='#000000', alpha=0.5))

                if multi.geom_type ==  'Polygon':
                    ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
                else:
                    for j, poly in enumerate(multi.geoms):
                        ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))


                nx.draw_networkx_edges(factoryPath.inter_filteredGraph, pos=pos_f, ax=ax, edge_color="grey", width=4)
                nx.draw_networkx_edges(factoryPath.fullPathGraph, pos=pos, ax=ax, edge_color="red", width=5)
                repPoints = [poly.representative_point() for poly in multi.geoms]

                for point in repPoints:
                    ax.plot(point.x, point.y, 'o', color='green', ms=10)
  

                if SAVEPLOT: plt.savefig(f"{runs+1}_6_Closest_Edge.{SAVEFORMAT}", format=SAVEFORMAT, bbox_inches='tight', transparent=True)
                plt.show()

            #  Path Plot --------------------------------------------------------------------------------------------------------
            print("7 - Path_Plot")
            fig, ax = plt.subplots(1,figsize=(8, 8))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.gca().invert_yaxis()
            plt.autoscale(False)
            plt.axis('off')

            for wall in wall_dict.values():
                    ax.add_patch(descartes.PolygonPatch(wall.poly, fc="darkgrey", ec='#000000', alpha=0.5))

            nx.draw(factoryPath.fullPathGraph, pos=pos, ax=ax, node_size=80, node_color='black')

            for u,v,data in factoryPath.reducedPathGraph.edges(data=True):
                linecolor = rng.random(size=3)
                nodelist = [pos[node] for node in data["nodelist"]]
                ax.plot(*zip(*nodelist), color=linecolor, linewidth=5)


            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))

            if SAVEPLOT: plt.savefig(f"{runs+1}_7_Path_Plot.{SAVEFORMAT}", format=SAVEFORMAT, bbox_inches='tight', transparent=True)
            plt.show()
            print("Fertsch")



# %%
