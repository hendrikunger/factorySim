# %%

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, MultiPolygon, MultiLineString, GeometryCollection, box
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
from shapely.ops import split,  voronoi_diagram,  unary_union
import descartes
import networkx as nx
import pickle

import time
# %%
SAVEPLOT = True
DETAILPLOT = False
TIMING = False
LOADDATA = False
ITERATIONS = 50

# savedata = { "bounding_box": bb, "machines": multi }
# pickle.dump( savedata, open( "FigureFactory.p", "wb" ) )


#%% Single Machines
WIDTH = 8
HEIGHT = 8
MAXSHAPEWIDTH = 4
MAXSHAPEHEIGHT = 5
AMOUNTRECT = 2
AMOUNTPOLY = 1
MAXCORNERS = 3

#%% Full Layout
WIDTH = 32
HEIGHT = 32
MAXSHAPEWIDTH = 4
MAXSHAPEHEIGHT = 4
AMOUNTRECT = 25
AMOUNTPOLY = 5
MAXCORNERS = 3


#%%
MINDEADEND_LENGTH = 2.0 # If Deadends are shorter than this, they are deleted
MINPATHWIDTH = 1.0  # Minimum Width of a Road to keep
BOUNDARYSPACING = 0.5  # Spacing of Points used as Voronoi Kernels
#%% Create Layout -----------------------------------------------------------------------------------------------------------------
for i in range(ITERATIONS):

    starttime = time.perf_counter()

    rng = np.random.default_rng()
    bb = box(0,0,WIDTH,HEIGHT)

    lowerLeftCornersRect = rng.integers([0,0], [WIDTH - MAXSHAPEWIDTH, HEIGHT - MAXSHAPEHEIGHT], size=[AMOUNTRECT,2], endpoint=True)
    lowerLeftCornersPoly = rng.integers([0,0], [WIDTH - MAXSHAPEWIDTH, HEIGHT - MAXSHAPEWIDTH], size=[AMOUNTPOLY,2], endpoint=True)

    polygons = []
    #Create Recangles
    for x,y in lowerLeftCornersRect:
        singlePoly = box(x,y,x + rng.integers(1, MAXSHAPEWIDTH+1), y + rng.integers(1, MAXSHAPEHEIGHT+1))
        singlePoly= rotate(singlePoly, rng.choice([0,90,180,270]))  
        polygons.append(singlePoly)

    #Create Convex Polygons
    for x,y in lowerLeftCornersPoly: 
        corners = []
        corners.append([x,y]) # First Corner
        for _ in range(rng.integers(2,MAXCORNERS+1)):
            corners.append([x + rng.integers(0, MAXSHAPEWIDTH+1), y + rng.integers(0, MAXSHAPEWIDTH+1)])

        singlePoly = MultiPoint(corners).minimum_rotated_rectangle
        singlePoly= rotate(singlePoly, rng.integers(0,361))  
        #Filter Linestrings
        if singlePoly.geom_type ==  'Polygon':
        #Filter small Objects
            if singlePoly.area > MAXSHAPEWIDTH*MAXSHAPEWIDTH*0.05:
                polygons.append(singlePoly)
        
    multi = unary_union(MultiPolygon(polygons))

    if LOADDATA:
        loaddata = pickle.load( open( "FigureFactory.p", "rb" ) )
        bb = loaddata["bounding_box"]
        multi = loaddata["machines"]


    walkableArea = bb.difference(multi)
    if walkableArea.geom_type ==  'MultiPolygon':
        walkableArea = walkableArea.geoms[0]

    nextTime = time.perf_counter()
    if TIMING: print(f"Factory generation {nextTime - starttime}")
    starttime = nextTime

    # %% Create Voronoi -----------------------------------------------------------------------------------------------------------------

    #Points around boundary
    distances = np.arange(0,  bb.boundary.length, BOUNDARYSPACING)
    points = [ bb.boundary.interpolate(distance) for distance in distances]
    
    #Points on Machines
    distances = np.arange(0,  multi.boundary.length, BOUNDARYSPACING)
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

    for line in voronoiArea.geoms[0].geoms:
        #find routes close to machines
        if  multi.intersects(line) or bb.crosses(line): 
            lines_touching_machines.append(line)
        #rest are main routes and dead ends
        else:
            route_lines.append(line)
        

    nextTime = time.perf_counter()
    if TIMING: print(f"Find Routes {nextTime - starttime}")
    starttime = nextTime


    #Split lines with machine objects
    sresult = split(MultiLineString(lines_touching_machines), multi)

    nextTime = time.perf_counter()
    if TIMING: print(f"Split {nextTime - starttime}")
    starttime = nextTime


    #Remove Geometries that are inside machines
    for line in sresult.geoms:
        if  not (multi.covers(line) and (not multi.disjoint(line) ) or multi.crosses(line)):
            lines_to_machines.append(line)

    nextTime = time.perf_counter()
    if TIMING: print(f"Line Filtering {nextTime - starttime}")
    starttime = nextTime

    # Find closest points in voronoi cells
    hitpoints = points + list(MultiPoint(walkableArea.exterior.coords).geoms)
    hit_tree = STRtree(hitpoints)

    # Create Graph -----------------------------------------------------------------------------------------------------------------
    G = nx.Graph()

    for line in route_lines:

        first = line.boundary.geoms[0]
        firstTuple = (first.x, first.y)
        first_str = str(firstTuple)
        #find closest next point in boundary for path width calculation
        nearest_point_first = hit_tree.nearest_geom(first)
        first_distance = first.distance(nearest_point_first)

        second = line.boundary.geoms[1]
        secondTuple = (second.x, second.y)
        second_str = str(secondTuple)
        #find closest next point in boundary for path width calculation
        nearest_point_second = hit_tree.nearest_geom(line.boundary.geoms[1])
        second_distance = second.distance(nearest_point_second)

        #edge weigth is minimum path width of the nodes making up the edge
        smallestPathwidth = min(first_distance, second_distance)

    #This is replaced by the version below. Delete Line Filtering below as well 
        G.add_node(first_str, pos=firstTuple, pathwidth=smallestPathwidth)
        G.add_node(second_str, pos=secondTuple, pathwidth=smallestPathwidth)
        G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)


    #For later --------------------------
        # if smallestPathwidth < MINPATHWIDTH:
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

    narrowPaths = [(n1, n2) for n1, n2, w in G.edges(data="pathwidth") if w < MINPATHWIDTH]
    F = G.copy()
    F.remove_edges_from(narrowPaths)

    #Find largest connected component to filter out "loose" parts
    Fcc = max(nx.connected_components(F), key=len)
    F = F.subgraph(Fcc).copy()

    #find crossroads
    old_crossroads = [node for node, degree in F.degree() if degree >= 3]
    #Set isCrossroads attribute on cross road nodes

    nx.set_node_attributes(F, dict.fromkeys(old_crossroads, True), 'isCrossroads')
    #find deadends
    old_endpoints = [node for node, degree in F.degree() if degree == 1]

    shortDeadEnds =[]

    for endpoint in old_endpoints:

        total_length = 0
        currentNode = endpoint
        nextNode = None
        lastNode = None
        tempDeadEnds = []

        #Follow path from endnnode to next crossroads, track length of path
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
            if total_length > MINDEADEND_LENGTH:
                break

            if "isCrossroads" in F.nodes[nextNode]:
                tempDeadEnds.append(currentNode)
                shortDeadEnds.extend(tempDeadEnds)
                break
            else:
                tempDeadEnds.append(currentNode)
                lastNode = currentNode
                currentNode = nextNode

    F.remove_nodes_from(shortDeadEnds)
    endpoints = [node for node, degree in F.degree() if degree == 1]
    crossroads = [node for node, degree in F.degree() if degree >= 3]

    nextTime = time.perf_counter()
    if TIMING: print(f"Network Filtering {nextTime - starttime}")
    starttime = nextTime

    # Create Machine Colors
    machine_colors = []

    if multi.geom_type ==  'Polygon':
        machine_colors.append(rng.random(size=3))
    else:
        machine_colors = rng.random(size=(len(multi.geoms),3))

    

    # %% Filtered_Lines Plot -----------------------------------------------------------------------------------------------------------------
    pos=nx.get_node_attributes(G,'pos')
    if DETAILPLOT:
        
        fig, ax = plt.subplots(1,figsize=(16, 16))
        plt.xlim(0,WIDTH)
        plt.ylim(0,HEIGHT)
        plt.autoscale(False)


        if multi.geom_type ==  'Polygon':
            ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
        else:
            for j, poly in enumerate(multi.geoms):
                ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))
        for line in route_lines:
            ax.plot(line.xy[0], line.xy[1], color='black')
        for line in lines_touching_machines:
            ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.5)
        for line in lines_to_machines:
            ax.plot(line.xy[0], line.xy[1], color='red', alpha=0.5)

        # for point in bb_points:
        #     ax.scatter(point.xy[0], point.xy[1], color='red')
        #ax.add_patch(descartes.PolygonPatch(allEdges, fc='blue', ec='#000000', alpha=0.5))  
        if SAVEPLOT: plt.savefig("1_Filtered_Lines.svg", format="svg")
        plt.show()

    # %% Pathwidth_Calculation Plot -----------------------------------------------------------------------------------------------------------------
    if DETAILPLOT:
        fig, ax = plt.subplots(1,figsize=(16, 16))
        plt.xlim(0,WIDTH)
        plt.ylim(0,HEIGHT)
        plt.autoscale(False)


        if multi.geom_type ==  'Polygon':
            ax.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
        else:
            for j, poly in enumerate(multi.geoms):
                ax.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))

        # for line in voronoiArea_:
        #     ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.5)
        for line in voronoiArea.geoms[0].geoms:
            ax.plot(line.xy[0], line.xy[1], color='red', alpha=0.0)



        for point in hitpoints:
            ax.scatter(point.x, point.y, color='red')

        for line in route_lines:
            ax.plot(line.xy[0], line.xy[1], color='black')
            for point in line.boundary.geoms:
                nearest_point = hit_tree.nearest_geom(point)
                #ax.plot([point.x, nearest_point.x], [point.y, nearest_point.y], color='green', alpha=1)
                ax.add_patch(plt.Circle((point.x, point.y), point.distance(nearest_point), color='blue', fill=False, alpha=0.3))
            #ax.add_patch(descartes.PolygonPatch(line.buffer(1), fc="black", ec='#000000', alpha=0.5))

        if SAVEPLOT: plt.savefig("2_Pathwidth_Calculation.svg", format="svg")
        plt.show()

    # %% Filtering Plot -----------------------------------------------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(32, 16))
    plt.tight_layout()
    ax1.set_xlim(0,WIDTH)
    ax1.set_ylim(0,HEIGHT)
    plt.autoscale(False)
    if multi.geom_type ==  'Polygon':
        ax1.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
    else:
        for j, poly in enumerate(multi.geoms):
            ax1.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))

    pathwidth = np.array(list((nx.get_edge_attributes(G,'pathwidth').values())))

    nx.draw_networkx_edges(G, pos=pos, ax=ax1, edge_color="silver", width=pathwidth * 50, alpha=0.6)
    nx.draw_networkx_edges(G, pos=pos, ax=ax1, edge_color="red", width=2, alpha=0.8)
    nx.draw_networkx_edges(F, pos=pos, ax=ax1, edge_color="dimgrey", width=5, alpha=1)
    nx.draw_networkx_edges(G, pos=pos, ax=ax1, edgelist=narrowPaths, edge_color="blue", width=2, alpha=0.9)


    nx.draw_networkx_nodes(G, pos=pos, ax=ax1, nodelist=shortDeadEnds, node_size=80, node_color='white', alpha=0.6, linewidths=4, edgecolors='green')
    nx.draw_networkx_nodes(G, pos=pos, ax=ax1, nodelist=old_endpoints, node_size=150, node_color='green')
    nx.draw_networkx_nodes(G, pos=pos, ax=ax1, nodelist=old_crossroads, node_size=150, node_color='white', alpha=0.6, linewidths=4, edgecolors='red')

  

    # %% Deadends Plot -----------------------------------------------------------------------------------------------------------------



    ax2.set_xlim(0,WIDTH)
    ax2.set_ylim(0,HEIGHT)
    plt.autoscale(False)

    if multi.geom_type ==  'Polygon':
        ax2.add_patch(descartes.PolygonPatch(multi, fc=machine_colors[0], ec='#000000', alpha=0.5))
    else:
        for j, poly in enumerate(multi.geoms):
            ax2.add_patch(descartes.PolygonPatch(poly, fc=machine_colors[j], ec='#000000', alpha=0.5))


    weights = np.array(list((nx.get_edge_attributes(F,'weight').values())))
    pathwidth = np.array(list((nx.get_edge_attributes(F,'pathwidth').values())))

    #nx.draw_networkx_nodes(F, pos=pos, ax=ax2, node_size=20, node_color='black', alpha=0.5)
    nx.draw_networkx_nodes(F, pos=pos, ax=ax2, nodelist=crossroads, node_size=120, node_color='red')
    nx.draw_networkx_nodes(F, pos=pos, ax=ax2, nodelist=endpoints, node_size=120, node_color='blue')
    nx.draw_networkx_edges(F, pos=pos, ax=ax2, width=pathwidth * 9, edge_color="dimgray", alpha=0.8)
    nx.draw_networkx_edges(F, pos=pos, ax=ax2, width=3, edge_color="black", alpha=0.5)

    if SAVEPLOT: plt.savefig(f"{i+1}_factory.png", format="png")
    
    plt.show()

    nextTime = time.perf_counter()
    if TIMING: print(f"Plotting {nextTime - starttime}")

   