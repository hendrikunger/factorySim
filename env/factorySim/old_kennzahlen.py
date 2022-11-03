# %%
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, MultiPolygon, MultiLineString, GeometryCollection, box, shape
from shapely.strtree import STRtree
from shapely.prepared import prep
from shapely.ops import split,  voronoi_diagram,  unary_union, triangulate, nearest_points
import descartes
import networkx as nx
from tqdm import tqdm

from routing import FactoryPath
from creation import FactoryCreator
from kpi import FactoryRating
import baseConfigs

import time
# %%
SAVEPLOT = False
SAVEFORMAT = "png"
DETAILPLOT = True
PLOT = True
TIMING = True
LOADDATA = False
LOADDXF = False
ITERATIONS = 1

# savedata = { "bounding_box": bb, "machines": multi }
# pickle.dump( savedata, open( "FigureFactory.p", "wb" ) )


factoryConfig = baseConfigs.SMALLSQUARE
factoryPath = FactoryPath(factoryConfig.pathParameters())


#%% Create Layout -----------------------------------------------------------------------------------------------------------------
for i in tqdm(range(ITERATIONS)):

    starttime = time.perf_counter()
    totaltime = starttime

    rng = np.random.default_rng()
    multi, bb = FactoryCreator(*factoryConfig.creationParameters()).create_factory()
    if LOADDATA:
        multi, bb = FactoryCreator.load_pickled_factory("FigureFactory.p")

    if LOADDXF:
        multi, bb = FactoryCreator.load_dxf_factory("Test.dxf")

    walkableArea = bb.difference(multi)
    if walkableArea.geom_type ==  'MultiPolygon':
        walkableArea = walkableArea.geoms[0]

    nextTime = time.perf_counter()
    if TIMING: print(f"Factory generation {nextTime - starttime}")
    starttime = nextTime

#   Create Voronoi -----------------------------------------------------------------------------------------------------------------
    #Points around boundary
    distances = np.arange(0,  bb.boundary.length, factoryConfig.BOUNDARYSPACING)
    points = [ bb.boundary.interpolate(distance) for distance in distances]
    
    #Points on Machines
    distances = np.arange(0,  multi.boundary.length, factoryConfig.BOUNDARYSPACING)
    points.extend([ multi.boundary.interpolate(distance) for distance in distances])
    bb_points = unary_union(points) 

    nextTime = time.perf_counter()
    if TIMING: print(f"Boundary generation {nextTime - starttime}")
    starttime = nextTime

    voronoiBase = GeometryCollection([walkableArea, bb_points])
    voronoiArea = voronoi_diagram(voronoiBase, edges=True)

    nextTime = time.perf_counter()
    if TIMING: print(f"Voronoi {nextTime - starttime}")
    starttime = nextTime

    route_lines = []
    lines_touching_machines = []
    lines_to_machines = []

    processed_multi = prep(multi)
    processed_bb = prep(bb)

    for line in voronoiArea.geoms[0].geoms:
        #find routes close to machines
        if  processed_multi.intersects(line) or processed_bb.crosses(line): 
            lines_touching_machines.append(line)
        #rest are main routes and dead ends
        else:
            route_lines.append(line)
        

    nextTime = time.perf_counter()
    if TIMING: print(f"Find Routes {nextTime - starttime}")
    starttime = nextTime

    if DETAILPLOT:
        #Split lines with machine objects#
        try:
            sresult = split(MultiLineString(lines_touching_machines), multi)
        except:
            print("Split Error")
            continue

        nextTime = time.perf_counter()
        if TIMING: print(f"Split {nextTime - starttime}")
        starttime = nextTime


        #Remove Geometries that are inside machines
        for line in sresult.geoms:
            if  not (processed_multi.covers(line) and (not processed_multi.disjoint(line) ) or processed_multi.crosses(line)):
                lines_to_machines.append(line)

        nextTime = time.perf_counter()
        if TIMING: print(f"Line Filtering {nextTime - starttime}")
        starttime = nextTime

    # Find closest points in voronoi cells
    if walkableArea.geom_type ==  'MultiPolygon':
        exteriorPoints = []
        for x in walkableArea.geoms:
            exteriorPoints.extend(list(x.exterior.coords))
    else:
        exteriorPoints = list(walkableArea.exterior.coords)
    hitpoints = points + list(MultiPoint(exteriorPoints).geoms)
    #hitpoints = MultiPoint(points+list(walkableArea.exterior.coords))
    hit_tree = STRtree(hitpoints)


    # Create Graph -----------------------------------------------------------------------------------------------------------------
    G = nx.Graph()

    memory = None
    memomry_distance = None

    for line in route_lines:

        first = line.boundary.geoms[0]
        firstTuple = (first.x, first.y)
        first_str = str(firstTuple)
        #find closest next point in boundary for path width calculation
        if memory == first:
            first_distance = memomry_distance
        else:
            nearest_point_first = hit_tree.nearest_geom(first)
            first_distance = first.distance(nearest_point_first)
            #nearest_point_first = nearest_points(hitpoints,first)[0]
            #first_distance = first.distance(nearest_point_first)

        second = line.boundary.geoms[1]
        secondTuple = (second.x, second.y)
        second_str = str(secondTuple)
        #find closest next point in boundary for path width calculation
        nearest_point_second = hit_tree.nearest_geom(second)
        second_distance = second.distance(nearest_point_second)
        #nearest_point_second = nearest_points(hitpoints,second)[0]
        #second_distance = second.distance(nearest_point_second)
        memory, memomry_distance = second, second_distance

        #edge width is minimum path width of the nodes making up the edge
        smallestPathwidth = min(first_distance, second_distance)


    #This is replaced by the version below. Delete Line Filtering below as well 
        G.add_node(first_str, pos=firstTuple)
        G.add_node(second_str, pos=secondTuple)
        G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)


    #For later --------------------------
        # if smallestPathwidth < factoryConfig.MINPATHWIDTH:
        #     continue
        # else:
        #     G.add_node(first_str, pos=firstTuple, pathwidth=smallestPathwidth)
        #     G.add_node(second_str, pos=secondTuple, pathwidth=smallestPathwidth)
        #     G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)



    nextTime = time.perf_counter()
    if TIMING: print(f"Network generation {nextTime - starttime}")
    starttime = nextTime
    # Filter  Graph -----------------------------------------------------------------------------------------------------------------
    """Cleans road network created with voronoi method by 
    - removing elements that are narrower than min_pathwidth
    - removing any dangelength parts that might have been cut off
    - removing all dead end that are shorter than min_length

    """

    narrowPaths = [(n1, n2) for n1, n2, w in G.edges(data="pathwidth") if w < factoryConfig.MINPATHWIDTH]
    F = G.copy()
    F.remove_edges_from(narrowPaths)

    #Find largest connected component to filter out "loose" parts
    Fcc = sorted(nx.connected_components(F), key=len, reverse=True)

    #print(f"Connected components ratio: {len(Fcc[0])/len(Fcc[1])}", )
    #if len(Fcc[0])/len(Fcc[1]) < 10: continue

 
    F = F.subgraph(Fcc[0]).copy()

    #find crossroads
    old_crossroads = [node for node, degree in F.degree() if degree >= 3]
    #Set isCrossroads attribute on cross road nodes

    nx.set_node_attributes(F, dict.fromkeys(old_crossroads, True), 'isCrossroads')
    #find deadends
    old_endpoints = [node for node, degree in F.degree() if degree == 1]


    shortDeadEnds = FactoryPath.pruneAlongPath(FactoryPath, F, starts=old_endpoints, ends=old_crossroads, min_length=factoryConfig.MINDEADENDLENGTH)

    F.remove_nodes_from(shortDeadEnds)
    endpoints = [node for node, degree in F.degree() if degree == 1]
    crossroads = [node for node, degree in F.degree() if degree >= 3]


# Prune unused dead ends
    pos=nx.get_node_attributes(G,'pos')

    repPoints = [poly.representative_point() for poly in multi.geoms]
    #Create Positions lists for nodes, since we need to querry shapley for shortest distance
    endpoint_pos = [pos[endpoint] for endpoint in endpoints ]
    crossroad_pos = [pos[crossroad] for crossroad in crossroads]
    total = endpoint_pos + crossroad_pos

    endpoints_to_prune = endpoints.copy()

    for point in repPoints:
        hit = nearest_points(point, MultiPoint(total))[1]
        key = str((hit.x, hit.y))
        if key in endpoints_to_prune: endpoints_to_prune.remove(key)

    nodes_to_prune = FactoryPath.pruneAlongPath(FactoryPath, F, starts=endpoints_to_prune, ends=crossroads, min_length=10)

    E = F.copy()
    E.remove_nodes_from(nodes_to_prune)

    endpoints = [node for node, degree in E.degree() if degree == 1]
    crossroads = [node for node, degree in E.degree() if degree >= 3]

    nx.set_node_attributes(E, FactoryPath.findSupportNodes(FactoryPath, E, cutoff=factoryConfig.SIMPLIFICATIONANGLE))
    support = list(nx.get_node_attributes(E, "isSupport").keys())


    nextTime = time.perf_counter()
    if TIMING: print(f"Network Filtering {nextTime - starttime}")
    starttime = nextTime



    # Alternative Simpicification and Path Generation ------------------------------------------------------------------------------------
    I = nx.Graph()

    visited = set() # Set to keep track of visited nodes.
    tempPath = [] # List to keep track of visited nodes in current path.
    paths = [] # List to keep track of all paths.

    ep=[node for node, degree in E.degree() if degree == 1]
    cross= [node for node, degree in E.degree() if degree >= 3]
    stoppers = set(ep + cross + support)

    if ep: 
        nodes_to_visit = [ep[0]]
    else:
        nodes_to_visit = [cross[0]]

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

        for outerNeighbor in E.neighbors(currentOuterNode):
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
                totalweight += E[lastNode][currentInnerNode]["weight"]
                pathwidth = E[lastNode][currentInnerNode]["pathwidth"]
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
                    if minpath > factoryConfig.MINTWOWAYPATHWIDTH: pathtype = "twoway"


                    I.add_node(currentOuterNode, pos=pos[currentOuterNode])
                    I.add_node(currentInnerNode, pos=pos[currentInnerNode])
                    I.add_edge(currentOuterNode, 
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

                for innerNeighbor in E.neighbors(currentInnerNode):
                    #Identifying next node (there will at most be two edges connected to every node)
                    if (innerNeighbor == lastNode):
                        #this is last node
                        continue
                    else:
                        #found the next one
                        lastNode = currentInnerNode
                        currentInnerNode = innerNeighbor
                        break


    nextTime = time.perf_counter()
    if TIMING: 
        print(f"Network Path Generation {nextTime - starttime}")
        print(f"Algorithm Total: {nextTime - totaltime}")
    starttime = nextTime

    # Create Machine Colors
    machine_colors = []

    if multi.geom_type ==  'Polygon':
        machine_colors.append(rng.random(size=3))
    else:
        machine_colors = rng.random(size=(len(multi.geoms),3))

    
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
            for line in route_lines:
                ax.plot(line.xy[0], line.xy[1], color='dimgray', linewidth=3)
            for line in lines_touching_machines:
                ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.5)
            for line in lines_to_machines:
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

            # for line in voronoiArea_:
            #     ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.5)
            for line in voronoiArea.geoms[0].geoms:
                ax.plot(line.xy[0], line.xy[1], color='red', alpha=0.0)

            for point in hitpoints:
                ax.scatter(point.x, point.y, color='red')

            for line in route_lines:
                ax.plot(line.xy[0], line.xy[1], color='black')
                # Plot Circle for every line Endpoint, since Startpoint is likely connected to other line segment
                point = line.boundary.geoms[0]
                nearest_point = hit_tree.nearest_geom(point)
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

        pathwidth = np.array(list((nx.get_edge_attributes(G,'pathwidth').values())))

        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="silver", width=pathwidth * 50, alpha=0.6)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="red", width=2, alpha=1)
        nx.draw_networkx_edges(F, pos=pos, ax=ax, edge_color="lime", width=2, alpha=1)
        nx.draw_networkx_edges(E, pos=pos, ax=ax, edge_color="dimgrey", width=5, alpha=1)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=narrowPaths, edge_color="blue", width=2, alpha=1)


        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=shortDeadEnds, node_size=80, node_color='white', alpha=0.6, linewidths=4, edgecolors='green')
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=old_endpoints, node_size=150, node_color='green')
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=old_crossroads, node_size=150, node_color='white', alpha=0.6, linewidths=4, edgecolors='red')

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


        for u,v,data in I.edges(data=True):
            temp = [pos[x] for x in data['nodelist']]
            ax.plot(*zip(*temp), color="dimgray", linewidth=data['pathwidth'] * 9, alpha=1.0, solid_capstyle='round')


        for u,v,data in I.edges(data=True):
            temp = [pos[x] for x in data['nodelist']]
            if data['pathtype'] =="twoway":
                ax.plot(*zip(*temp), color="white", linewidth=3, alpha=0.5, solid_capstyle='round', linestyle='dashed')

        nx.draw_networkx_nodes(E, pos=pos, ax=ax, nodelist=crossroads, node_size=120, node_color='red')
        nx.draw_networkx_nodes(E, pos=pos, ax=ax, nodelist=endpoints, node_size=120, node_color='blue')
        nx.draw_networkx_nodes(E, pos=pos, ax=ax, nodelist=support, node_size=120, node_color='green')

        if SAVEPLOT: plt.savefig(f"{i+1}_4_Clean.{SAVEFORMAT}", format=SAVEFORMAT)

        plt.show()

        #  Simplification Plot -----------------------------------------------------------------------------------------------------------------

        fig, ax = plt.subplots(1,figsize=(16, 16))
        plt.xlim(0,bb.bounds[2])
        plt.ylim(0,bb.bounds[3])
        plt.autoscale(False)

        for u,v,a in I.edges(data=True):
            linecolor = rng.random(size=3)
            temp = [pos[x] for x in data['nodelist']]
            for i, node in enumerate(temp[1:]):
                ax.plot(*temp[i-1], *node, color=linecolor, linewidth=50)


        if multi.geom_type ==  'Polygon':
            ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
        else:
            for j, poly in enumerate(multi.geoms):
                ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))


        min_pathwidth = np.array(list((nx.get_edge_attributes(I,'pathwidth').values())))
        max_pathwidth = np.array(list((nx.get_edge_attributes(I,'max_pathwidth').values())))

        nx.draw_networkx_nodes(E, pos=pos, ax=ax, node_size=20, node_color='black')
        nx.draw_networkx_nodes(I, pos=pos, ax=ax, node_size=120, node_color='red')
        nx.draw_networkx_nodes(E, pos=pos, ax=ax, nodelist=support, node_size=120, node_color='green')
        #old simplification
        #nx.draw_networkx_edges(I, pos=pos, ax=ax, width=max_pathwidth * 9, edge_color="dimgrey")
        #nx.draw_networkx_edges(I, pos=pos, ax=ax, width=min_pathwidth * 9, edge_color="blue", alpha=0.5)
        nx.draw_networkx_edges(E, pos=pos, ax=ax, edge_color="dimgray", alpha=0.5)

        min_pathwidth = np.array(list((nx.get_edge_attributes(I,'pathwidth').values())))
        max_pathwidth = np.array(list((nx.get_edge_attributes(I,'max_pathwidth').values())))

        nx.draw_networkx_edges(I, pos=pos, ax=ax, edge_color="yellow", width=4)
        nx.draw_networkx_edges(I, pos=pos, ax=ax, width=max_pathwidth * 9, edge_color="dimgrey", alpha=0.7)
        nx.draw_networkx_edges(I, pos=pos, ax=ax, width=min_pathwidth * 9, edge_color="blue", alpha=0.5)




        if SAVEPLOT: plt.savefig(f"{i+1}_5_Simplification.{SAVEFORMAT}", format=SAVEFORMAT)
        plt.show()

        #  Closest Edge Plot -----------------------------------------------------------------------------------------------------------------
        if DETAILPLOT:

            fig, ax = plt.subplots(1,figsize=(16, 16))
            plt.xlim(0,bb.bounds[2])
            plt.ylim(0,bb.bounds[3])
            plt.autoscale(False)

            ax.add_patch(descartes.PolygonPatch(walkableArea, alpha=0.5))
            triangles = triangulate(walkableArea)

            # for tri in triangles:
            #     ax.add_patch(descartes.PolygonPatch(tri, alpha=0.5))

            nx.draw_networkx_edges(F, pos=pos, ax=ax, edge_color="grey", width=4)
            nx.draw_networkx_edges(E, pos=pos, ax=ax, edge_color="red", width=5)
            repPoints = [poly.representative_point() for poly in multi.geoms]
            endpoint_pos = [pos[endpoint] for endpoint in endpoints ]
            crossroad_pos = [pos[crossroad] for crossroad in crossroads]
            total = endpoint_pos + crossroad_pos

            endpoints_to_prune = endpoints.copy()



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

        nx.draw(E, pos=pos, ax=ax, node_size=80, node_color='black')

        for path in paths:
            linecolor = rng.random(size=3)
            temp = [pos[x] for x in path]
            ax.plot(*zip(*temp), color=linecolor, linewidth=5)


        if multi.geom_type ==  'Polygon':
            ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
        else:
            for j, poly in enumerate(multi.geoms):
                ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))

        if SAVEPLOT: plt.savefig(f"{i+1}_6_Path_Plot.{SAVEFORMAT}", format=SAVEFORMAT)
        plt.show()

        nextTime = time.perf_counter()
        if TIMING: print(f"Plotting {nextTime - starttime}")
        starttime = nextTime
        


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



#%%
print(f"I.size: {I.size()}\n")
print(f"I.weight: {I.size(weight='weight')}\n")
print(f"I.pathwidth: {I.size(weight='pathwidth')}\n")
print(f"I.max_pathwidth: {I.size(weight='max_pathwidth')}\n")

factoryRating = FactoryRating(None, E, I)

print(f"Mean Road Dimension Variability: {factoryRating.PathWideVariance()}")


