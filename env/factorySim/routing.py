#%%
import time

from shapely.geometry import Point, MultiPoint, MultiPolygon, MultiLineString, GeometryCollection, box, shape
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
from shapely.prepared import prep
from shapely.ops import split,  voronoi_diagram,  unary_union, triangulate, nearest_points

import numpy as np
import networkx as nx

DETAILPLOT = False

class FactoryPath():

    fullPathGraph = None
    ReducedPathGraph = None
    TIMING = False
    PLOTTING = False

    def __init__(self, boundarySpacing=1.5, minDeadEndLength=2.0, minPathWidth=1.0, minTwoWayPathWidth=2.0, simplificationAngle=35):

        self.minDeadEndLength = minDeadEndLength # If Deadends are shorter than this, they are deleted
        self.minPathWidth = minPathWidth  # Minimum Width of a Road to keep
        self.minTwoWayPathWidth = minTwoWayPathWidth  # Minimum Width of a Road to keep
        self.boundarySpacing = boundarySpacing # Spacing of Points used as Voronoi Kernels
        self.simplificationAngle = simplificationAngle # Angle in degrees, used for support point calculation in simple path
    	
    def timelog(self, text):
        self.nextTime = time.perf_counter()
        print(f"{text} {self.nextTime - self.startTime}")
        self.startTime = self.nextTime

    def calculateAll(self, machine_dict, bb):
        #Check if we have enough machines to make a path
        if len(machine_dict) <= 1:
            self.fullPathGraph = nx.Graph()
            self.ReducedPathGraph = nx.Graph()
            return self.fullPathGraph, self.ReducedPathGraph

        multi = MultiPolygon(unary_union([x.poly for x in machine_dict.values()]))
        #Scale boundary spacing according to factory size
        scale = max(bb.bounds) / 30

        if self.TIMING:
            self.startTime = time.perf_counter()
            self.totalTime = self.startTime

        walkableArea = bb.difference(unary_union(multi))
        if walkableArea.geom_type ==  'MultiPolygon':
            walkableArea = walkableArea.geoms[0]


        #   Create Voronoi -----------------------------------------------------------------------------------------------------------------
        #Points around boundary

        distances = np.arange(0,  bb.boundary.length, self.boundarySpacing * scale)
        points = [ bb.boundary.interpolate(distance) for distance in distances]

        #Points on Machines
        distances = np.arange(0,  multi.boundary.length, self.boundarySpacing * scale)
        points.extend([ multi.boundary.interpolate(distance) for distance in distances])
        bb_points = unary_union(points) 

        if self.TIMING: self.timelog("Boundary generation")

        voronoiBase = GeometryCollection([walkableArea, bb_points])
        voronoiArea = voronoi_diagram(voronoiBase, edges=True)
        if self.TIMING: self.timelog("Voronoi")

        self.route_lines = []
        self.lines_touching_machines = []
        self.lines_to_machines = []

        processed_multi = prep(multi)
        processed_bb = prep(bb)

        for line in voronoiArea.geoms[0].geoms:
            #find routes close to machines
            if not (processed_multi.intersects(line) or processed_bb.crosses(line)): 
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


        # Create Graph -----------------------------------------------------------------------------------------------------------------
        self.fullPathGraph = nx.Graph()

        memory = None
        memomry_distance = None

        for line in self.route_lines:

            first = line.boundary.geoms[0]
            firstTuple = (first.x, first.y)
            first_str = str(firstTuple)
            #find closest next point in boundary for path width calculation
            if memory == first:
                first_distance = memory_distance
            else:
                nearest_point_first = self.hit_tree.nearest_geom(first)
                first_distance = first.distance(nearest_point_first)


            second = line.boundary.geoms[1]
            secondTuple = (second.x, second.y)
            second_str = str(secondTuple)
            #find closest next point in boundary for path width calculation
            nearest_point_second = self.hit_tree.nearest_geom(second)
            second_distance = second.distance(nearest_point_second)

            memory, memory_distance = second, second_distance

            #edge width is minimum path width of the nodes making up the edge
            smallestPathwidth = min(first_distance, second_distance)


        #This is replaced by the version below. Delete Line Filtering below as well 
            self.fullPathGraph.add_node(first_str, pos=firstTuple, pathwidth=first_distance)
            self.fullPathGraph.add_node(second_str, pos=secondTuple, pathwidth=second_distance)
            self.fullPathGraph.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)


        #For later --------------------------
            # if smallestPathwidth < MINPATHWIDTH:
            #     continue
            # else:
            #     self.G.add_node(first_str, pos=firstTuple, pathwidth=smallestPathwidth)
            #     self.G.add_node(second_str, pos=secondTuple, pathwidth=smallestPathwidth)
            #     self.G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)

        if self.TIMING: self.timelog("Network generation")

        # Filter  Graph -----------------------------------------------------------------------------------------------------------------
        # Cleans road network created with voronoi method by 
        # - removing elements that are narrower than min_pathwidth
        # - removing any dangelength parts that might have been cut off
        # - removing all dead end that are shorter than min_length


        if self.PLOTTING: self.inter_unfilteredGraph = self.fullPathGraph.copy()

        self.narrowPaths = [(n1, n2) for n1, n2, w in self.fullPathGraph.edges(data="pathwidth") if w < self.minPathWidth]
        self.fullPathGraph.remove_edges_from(self.narrowPaths)

        #Find largest connected component to filter out "loose" parts
        Gcc = sorted(nx.connected_components(self.fullPathGraph), key=len, reverse=True)
        if (len(Gcc) > 0):
            self.fullPathGraph = self.fullPathGraph.subgraph(Gcc[0]).copy()

        #find crossroads
        self.old_crossroads = [node for node, degree in self.fullPathGraph.degree() if degree >= 3]
        #Set isCrossroads attribute on cross road nodes
        nx.set_node_attributes(self.fullPathGraph, dict.fromkeys(self.old_crossroads, True), 'isCrossroads')
        #find deadends
        self.old_endpoints = [node for node, degree in self.fullPathGraph.degree() if degree == 1]

        if self.old_crossroads and self.old_endpoints:
            self.shortDeadEnds = self.pruneAlongPath(self.fullPathGraph, starts=self.old_endpoints, ends=self.old_crossroads, min_length=self.minDeadEndLength)
            self.fullPathGraph.remove_nodes_from(self.shortDeadEnds)

        self.endpoints = [node for node, degree in self.fullPathGraph.degree() if degree == 1]
        self.crossroads = [node for node, degree in self.fullPathGraph.degree() if degree >= 3]

        # Prune unused dead ends
        pos=nx.get_node_attributes(self.fullPathGraph,'pos')

        repPoints = [poly.representative_point() for poly in multi.geoms]

        #Find closest crossroads or endpoint from every machine and prune deadends that are not used by any machine
        #Create Positions lists for nodes, since we need to querry shapley for shortest distance
        endpoint_pos = [pos[endpoint] for endpoint in self.endpoints ]
        crossroad_pos = [pos[crossroad] for crossroad in self.crossroads]
        total = endpoint_pos + crossroad_pos


        if self.PLOTTING: self.inter_filteredGraph = self.fullPathGraph.copy()
        if total:
            endpoints_to_prune = self.endpoints.copy()
            for point in repPoints:
                hit = nearest_points(point, MultiPoint(total))[1]
                key = str((hit.x, hit.y))
                if key in endpoints_to_prune: endpoints_to_prune.remove(key)

            if endpoints_to_prune:
                nodes_to_prune = self.pruneAlongPath(self.fullPathGraph, starts=endpoints_to_prune, ends=self.crossroads, min_length=5 * self.minDeadEndLength)
                if nodes_to_prune: self.fullPathGraph.remove_nodes_from(nodes_to_prune)


        self.endpoints = [node for node, degree in self.fullPathGraph.degree() if degree == 1]
        self.crossroads = [node for node, degree in self.fullPathGraph.degree() if degree >= 3]

        nx.set_node_attributes(self.fullPathGraph, self.findSupportNodes(self.fullPathGraph, cutoff=self.simplificationAngle))
        self.support = list(nx.get_node_attributes(self.fullPathGraph, "isSupport").keys())

        if self.TIMING: self.timelog("Network Filtering")

        # Simpicification and Path Generation ------------------------------------------------------------------------------------

        self.ReducedPathGraph = nx.Graph()

        visited = set() # Set to keep track of visited nodes.
        tempPath = [] # List to keep track of visited nodes in current path.
        paths = [] # List to keep track of all paths.

        ep = self.endpoints
        cross = self.crossroads
        stoppers = set(ep + cross +  self.support)

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
                        paths.append(tempPath)
                        
                        #Prevent going back and forth between direct connected crossroads 
                        if lastNode != currentOuterNode:
                            visited.add(lastNode)
                        nodes_to_visit.append(currentInnerNode)

                        pathtype = "oneway"
                        if minpath > self.minTwoWayPathWidth: pathtype = "twoway"


                        self.ReducedPathGraph.add_node(currentOuterNode, pos=pos[currentOuterNode])
                        self.ReducedPathGraph.add_node(currentInnerNode, pos=pos[currentInnerNode])
                        self.ReducedPathGraph.add_edge(currentOuterNode, 
                            currentInnerNode, 
                            weight=totalweight,
                            pathwidth=minpath, 
                            max_pathwidth=maxpath, 
                            nodelist=tempPath,
                            pathtype=pathtype
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

        if self.TIMING: 
            self.timelog("Network Path Generation")
            print(f"Algorithm Total: {self.nextTime - self.totalTime}")

        return self.fullPathGraph, self.ReducedPathGraph

    def calculateNodeAngles(self, G, cutoff=45):

        candidates = [node for node, degree in G.degree() if degree == 2]
        node_data ={}
        pos=nx.get_node_attributes(G,'pos')

        for node in candidates: 
            neighbors = list(G.neighbors(node))
            
            vector_1 = [pos[node][0] - pos[neighbors[0]][0], pos[node][1] - pos[neighbors[0]][1]]
            vector_2 = [pos[neighbors[1]][0] - pos[node][0], pos[neighbors[1]][1] - pos[node][1]]

            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.clip(np.dot(unit_vector_1, unit_vector_2),-0.99999999,0.99999999)
            angle = np.rad2deg(np.arccos(dot_product))


            #angle = np.arctan2(*pos[neighbors[0]]) - np.arctan2(*pos[neighbors[1]])
            node_data[node] = {"edge_angle" : angle}

            # if(angle < -np.pi/12  or angle > np.pi/12):
            #     node_data[node] = {"isSupport" : True}

            if(abs(angle) > cutoff):
                node_data[node] = {"isSupport" : True}

        return node_data


    def findSupportNodes(self, G, cutoff=45):

        candidates = [node for node, degree in G.degree() if degree == 2]

        node_data ={}
        pos=nx.get_node_attributes(G,'pos')

        for node in candidates: 
            direct_neighbors = list(G.neighbors(node))
            
            pre = list(G.neighbors(direct_neighbors[0]))
            pre.remove(node)

            if len(pre) == 1:
                vector_1 = [pos[node][0] - pos[pre[0]][0], pos[node][1] - pos[pre[0]][1]]
            else:
                #direct_neighbors[0] is a endpoint or crossroad, can not use its neighbors for calculation
                vector_1 = [pos[node][0] - pos[direct_neighbors[0]][0], pos[node][1] - pos[direct_neighbors[0]][1]]

            suc = list(G.neighbors(direct_neighbors[1]))
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

            node_data[node] = {"edge_angle" : angle}

            if(abs(angle) > cutoff):
                node_data[node] = {"isSupport" : True}

        for node in dict(node_data):
            if node_data[node].get("isSupport", False):
                direct_neighbors = G.neighbors(node)
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
                total_length += F.edges[currentNode, nextNode]["weight"]
                #Stop if route is longer than min_length
                if total_length > min_length:
                    break

                if nextNode in ends:
                    tempDeadEnds.append(currentNode)
                    shortDeadEnds.extend(tempDeadEnds)
                    break 
                else:
                    tempDeadEnds.append(currentNode)
                    lastNode = currentNode
                    currentNode = nextNode
        
        return shortDeadEnds


#%% TESTS
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import descartes
    from tqdm import tqdm
    from factorySim.creation import FactoryCreator
    import factorySim.baseConfigs as baseConfigs

    SAVEPLOT = True
    SAVEFORMAT = "png"
    DETAILPLOT = True
    PLOT = True
    ITERATIONS = 1

    rng = np.random.default_rng()

    for i in tqdm(range(ITERATIONS)):
        factoryCreator = FactoryCreator(*baseConfigs.SMALLSQUARE.creationParameters())
        machine_dict = factoryCreator.create_factory()

        multi = MultiPolygon(unary_union([x.poly for x in machine_dict.values()]))
        bb = box(*multi.bounds) 

        machine_colors = [rng.random(size=3) for _ in multi.geoms]
        factoryPath = FactoryPath()
        factoryPath.TIMING = True
        factoryPath.PLOTTING = True
        factoryPath.calculateAll(machine_dict, bb)
        pos=nx.get_node_attributes(factoryPath.inter_unfilteredGraph,'pos')

        if PLOT:
    #  Filtered_Lines Plot -----------------------------------------------------------------------------------------------------------------
            if DETAILPLOT:

                fig, ax = plt.subplots(1,figsize=(16, 16))
                plt.xlim(0,bb.bounds[2])
                plt.ylim(0,bb.bounds[3])
                plt.autoscale(False)


                if multi.geom_type ==  'Polygon':
                    ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
                else:
                    for j, poly in enumerate(multi.geoms):
                        ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))
                for line in factoryPath.route_lines:
                    ax.plot(line.xy[0], line.xy[1], color='dimgray', linewidth=3)
                for line in factoryPath.lines_touching_machines:
                    ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.5)
                for line in factoryPath.lines_to_machines:
                    ax.plot(line.xy[0], line.xy[1], color='red', alpha=0.9)

                # for point in bb_points:
                #     ax.scatter(point.xy[0], point.xy[1], color='red')
                #ax.add_patch(descartes.PolygonPatch(allEdges, fc='blue', ec='#000000', alpha=0.5))  
                if SAVEPLOT: plt.savefig(f"{i+1}_1_Filtered_Lines.{SAVEFORMAT}", format=SAVEFORMAT)
                plt.show()

            # Pathwidth_Calculation Plot -----------------------------------------------------------------------------------------------------------------
            if DETAILPLOT:
                fig, ax = plt.subplots(1,figsize=(16, 16))
                plt.xlim(0,bb.bounds[2])
                plt.ylim(0,bb.bounds[3])
                plt.autoscale(False)

                for point in factoryPath.hitpoints:
                    ax.scatter(point.x, point.y, color='red')

                for line in factoryPath.route_lines:
                    ax.plot(line.xy[0], line.xy[1], color='black')
                    # Plot Circle for every line Endpoint, since Startpoint is likely connected to other line segment
                    point = line.boundary.geoms[0]
                    nearest_point = factoryPath.hit_tree.nearest_geom(point)
                    #ax.plot([point.x, nearest_point.x], [point.y, nearest_point.y], color='green', alpha=1)
                    ax.add_patch(plt.Circle((point.x, point.y), point.distance(nearest_point), color='blue', fill=False, alpha=0.6))
                    #ax.add_patch(descartes.PolygonPatch(line.buffer(1), fc="black", ec='#000000', alpha=0.5))
                
                if multi.geom_type ==  'Polygon':
                    ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
                else:
                    for j, poly in enumerate(multi.geoms):
                        ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.8))           

                if SAVEPLOT: plt.savefig(f"{i+1}_2_Pathwidth_Calculation.{SAVEFORMAT}", format=SAVEFORMAT)
                plt.show()

            #  Filtering Plot -----------------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(1, figsize=(16, 16))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.autoscale(False)

            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))

            pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.inter_unfilteredGraph,'pathwidth').values())))

            nx.draw_networkx_edges(factoryPath.inter_unfilteredGraph, pos=pos, ax=ax, edge_color="silver", width=pathwidth * 50, alpha=0.6)
            nx.draw_networkx_edges(factoryPath.inter_unfilteredGraph, pos=pos, ax=ax, edge_color="red", width=2, alpha=1)
            nx.draw_networkx_edges(factoryPath.inter_filteredGraph, pos=pos, ax=ax, edge_color="lime", width=2, alpha=1)
            nx.draw_networkx_edges(factoryPath.fullPathGraph, pos=pos, ax=ax, edge_color="dimgrey", width=5, alpha=1)
            nx.draw_networkx_edges(factoryPath.inter_unfilteredGraph, pos=pos, ax=ax, edgelist=factoryPath.narrowPaths, edge_color="blue", width=2, alpha=1)


            nx.draw_networkx_nodes(factoryPath.inter_unfilteredGraph, pos=pos, ax=ax, nodelist=factoryPath.shortDeadEnds, node_size=80, node_color='white', alpha=0.6, linewidths=4, edgecolors='green')
            nx.draw_networkx_nodes(factoryPath.inter_unfilteredGraph, pos=pos, ax=ax, nodelist=factoryPath.old_endpoints, node_size=150, node_color='green')
            nx.draw_networkx_nodes(factoryPath.inter_unfilteredGraph, pos=pos, ax=ax, nodelist=factoryPath.old_crossroads, node_size=150, node_color='white', alpha=0.6, linewidths=4, edgecolors='red')

            if SAVEPLOT: plt.savefig(f"{i+1}_3_Pruning.{SAVEFORMAT}", format=SAVEFORMAT)
            
            plt.show()

            #  Clean Plot -----------------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(1, figsize=(16, 16))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.autoscale(False)

            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))


            for u,v,data in factoryPath.ReducedPathGraph.edges(data=True):
                temp = [pos[x] for x in data['nodelist']]
                ax.plot(*zip(*temp), color="dimgray", linewidth=data['pathwidth'] * 9, alpha=1.0, solid_capstyle='round')


            for u,v,data in factoryPath.ReducedPathGraph.edges(data=True):
                temp = [pos[x] for x in data['nodelist']]
                if data['pathtype'] =="twoway":
                    ax.plot(*zip(*temp), color="white", linewidth=3, alpha=0.5, solid_capstyle='round', linestyle='dashed')

            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.crossroads, node_size=120, node_color='red')
            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.endpoints, node_size=120, node_color='blue')
            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.support, node_size=120, node_color='green')

            if SAVEPLOT: plt.savefig(f"{i+1}_4_Clean.{SAVEFORMAT}", format=SAVEFORMAT)

            plt.show()

            #  Simplification Plot -----------------------------------------------------------------------------------------------------------------

            fig, ax = plt.subplots(1,figsize=(16, 16))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.autoscale(False)

            for u,v,a in factoryPath.ReducedPathGraph.edges(data=True):
                linecolor = rng.random(size=3)
                temp = [pos[x] for x in data['nodelist']]
                for i, node in enumerate(temp[1:]):
                    ax.plot(*temp[i-1], *node, color=linecolor, linewidth=50)


            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))


            min_pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.ReducedPathGraph,'pathwidth').values())))
            max_pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.ReducedPathGraph,'max_pathwidth').values())))

            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, node_size=20, node_color='black')
            nx.draw_networkx_nodes(factoryPath.ReducedPathGraph, pos=pos, ax=ax, node_size=120, node_color='red')
            nx.draw_networkx_nodes(factoryPath.fullPathGraph, pos=pos, ax=ax, nodelist=factoryPath.support, node_size=120, node_color='green')
            #old simplification
            #nx.draw_networkx_edges(factoryPath.PathGraph, pos=pos, ax=ax, width=max_pathwidth * 9, edge_color="dimgrey")
            #nx.draw_networkx_edges(factoryPath.PathGraph, pos=pos, ax=ax, width=min_pathwidth * 9, edge_color="blue", alpha=0.5)
            nx.draw_networkx_edges(factoryPath.fullPathGraph, pos=pos, ax=ax, edge_color="dimgray", alpha=0.5)

            min_pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.ReducedPathGraph,'pathwidth').values())))
            max_pathwidth = np.array(list((nx.get_edge_attributes(factoryPath.ReducedPathGraph,'max_pathwidth').values())))

            nx.draw_networkx_edges(factoryPath.ReducedPathGraph, pos=pos, ax=ax, edge_color="yellow", width=4)
            nx.draw_networkx_edges(factoryPath.ReducedPathGraph, pos=pos, ax=ax, width=max_pathwidth * 9, edge_color="dimgrey", alpha=0.7)
            nx.draw_networkx_edges(factoryPath.ReducedPathGraph, pos=pos, ax=ax, width=min_pathwidth * 9, edge_color="blue", alpha=0.5)


            if SAVEPLOT: plt.savefig(f"{i+1}_5_Simplification.{SAVEFORMAT}", format=SAVEFORMAT)
            plt.show()

            #  Closest Edge Plot -----------------------------------------------------------------------------------------------------------------
            if DETAILPLOT:

                fig, ax = plt.subplots(1,figsize=(16, 16))
                plt.xlim(0,bb.bounds[2])
                plt.ylim(0,bb.bounds[3])
                plt.autoscale(False)

                if multi.geom_type ==  'Polygon':
                    ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
                else:
                    for j, poly in enumerate(multi.geoms):
                        ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))


                nx.draw_networkx_edges(factoryPath.inter_filteredGraph, pos=pos, ax=ax, edge_color="grey", width=4)
                nx.draw_networkx_edges(factoryPath.fullPathGraph, pos=pos, ax=ax, edge_color="red", width=5)
                repPoints = [poly.representative_point() for poly in multi.geoms]
                endpoint_pos = [pos[endpoint] for endpoint in factoryPath.endpoints ]
                crossroad_pos = [pos[crossroad] for crossroad in factoryPath.crossroads]
                total = endpoint_pos + crossroad_pos

                endpoints_to_prune = factoryPath.endpoints.copy()



                for point in repPoints:
                    ax.plot(point.x, point.y, 'o', color='green', ms=10)
                    hit = nearest_points(point, MultiPoint(total))[1]
                    ax.plot([point.x, hit.x],[ point.y, hit.y], color=rng.random(size=3),linewidth=3)
                    key = str((hit.x, hit.y))
                    if key in endpoints_to_prune: endpoints_to_prune.remove(key)


                if SAVEPLOT: plt.savefig(f"{i+1}_5_Closest_Edge.{SAVEFORMAT}", format=SAVEFORMAT)
                plt.show()

            #  Path Plot --------------------------------------------------------------------------------------------------------
            fig, ax = plt.subplots(1,figsize=(16, 16))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.autoscale(False)

            nx.draw(factoryPath.fullPathGraph, pos=pos, ax=ax, node_size=80, node_color='black')

            for u,v,data in factoryPath.ReducedPathGraph.edges(data=True):
                temp = [pos[x] for x in data['nodelist']]
                linecolor = rng.random(size=3)
                ax.plot(*zip(*temp), color=linecolor, linewidth=5)


            if multi.geom_type ==  'Polygon':
                ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
            else:
                for j, poly in enumerate(multi.geoms):
                    ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))

            if SAVEPLOT: plt.savefig(f"{i+1}_6_Path_Plot.{SAVEFORMAT}", format=SAVEFORMAT)
            plt.show()
            print("Fertsch")


        


# 2 - Überschneidungsfreiheit	        Materialflussschnittpunkte
# 3 - Stetigkeit	                    Richtungswechsel im Materialfluss



# 	                                    Verwinkelung
# 	                                    Vorhandensein eindeutiger Wegachsen
# 	                                    Wegeeffizienz
# 6 - Zugänglichkeit	                Abdeckung Wegenetz
# 	                                    Kontaktflächen Wegenetz
# 7 - Flächennutzungsgrad	            genutzte Fabrikfläche (ohne zusammenhängende Freifläche)
# 1 - Skalierbarkeit 	                Ausdehnung der größten verfügbaren Freifläche
# 2 - Medienverfügbarkeit	            Möglichkeit des Anschlusses von Maschinen an Prozessmedien (z.B. Wasser, Druckluft)
# 1 - Beleuchtung	                    Erfüllung der Arbeitsplatzanforderungen
# 2 - Ruhe	                            Erfüllung der Arbeitsplatzanforderungen
# 3 - Erschütterungsfreiheit	        Erfüllung der Arbeitsplatzanforderungen
# 4 - Sauberkeit	                    Erfüllung der Arbeitsplatzanforderungen
# 5 - Temperatur	                    Erfüllung der Arbeitsplatzanforderungen



# Erledigt =================================================================

# 1 - Materialflusslänge	            Entfernung (direkt)
# 	                                    Entfernung (wegorientiert)
# 4 - Intensität	                    Anzahl der Transporte
# 5 - Wegekonzept	                    Auslegung Wegbreite
# 	                                    Sackgassen




# %%
