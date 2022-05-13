# %%
from ast import If
from operator import ne
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString, GeometryCollection, box, mapping
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
from shapely.ops import split, snap, voronoi_diagram, triangulate, unary_union, nearest_points
import descartes
import networkx as nx

import time

SAVEFILE = False 
# %%
def plot(multipoly:MultiPolygon, bb:MultiPolygon = None):
    rng = np.random.default_rng()
    fig, ax = plt.subplots(1,figsize=(16, 16))
    #plt.axis('off')
    plt.xlim(0,WIDTH)
    plt.ylim(0,HEIGHT)
    plt.autoscale(False)
    if bb is not None:
        ax.add_patch(descartes.PolygonPatch(bb, fc='red', ec='#000000', alpha=0.5))

    if multi.geom_type ==  'Polygon':
        ax.add_patch(descartes.PolygonPatch(multi, fc=rng.random(size=3), ec='#000000', alpha=0.5))
    else:
        for poly in multipoly.geoms:
            ax.add_patch(descartes.PolygonPatch(poly, fc=rng.random(size=3), ec='#000000', alpha=0.5))
    plt.show()


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
MAXSHAPEHEIGHT = 6
AMOUNTRECT = 20
AMOUNTPOLY = 10
MAXCORNERS = 3

#%% 

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
walkableArea = bb.difference(multi)

nextTime = time.perf_counter()
print(f"Factory generation {nextTime - starttime}")
starttime = nextTime

# %%
BOUNDARYSPACING = 0.5
# if multi.geom_type ==  'Polygon':
#     allCenters = multi.convex_hull
# else:
#     allCenters = MultiPolygon([x.convex_hull for x in multi.geoms])
#     allCenters = unary_union(allCenters)

#Points around boundary
distances = np.arange(0,  bb.boundary.length, BOUNDARYSPACING)
points = [ bb.boundary.interpolate(distance) for distance in distances]
 
#Points on Machines
distances = np.arange(0,  multi.boundary.length, BOUNDARYSPACING)
points.extend([ multi.boundary.interpolate(distance) for distance in distances])
bb_points = unary_union(points) 

nextTime = time.perf_counter()
print(f"Boundary generation {nextTime - starttime}")
starttime = nextTime

voronoiBase = GeometryCollection([walkableArea, bb_points])
voronoiArea = voronoi_diagram(voronoiBase, edges=True)

nextTime = time.perf_counter()
print(f"Voronoi {nextTime - starttime}")
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
print(f"Find Routes {nextTime - starttime}")
starttime = nextTime


#Split lines with machine objects
sresult = split(MultiLineString(lines_touching_machines), multi)

nextTime = time.perf_counter()
print(f"Split {nextTime - starttime}")
starttime = nextTime


#Remove Geometries that are inside machines
for line in sresult.geoms:
    if  not (multi.covers(line) and (not multi.disjoint(line) ) or multi.crosses(line)):
        lines_to_machines.append(line)

nextTime = time.perf_counter()
print(f"Line Filtering {nextTime - starttime}")
starttime = nextTime



# fig, ax = plt.subplots(1,figsize=(16, 16))
# plt.xlim(0,WIDTH)
# plt.ylim(0,HEIGHT)
# plt.autoscale(False)


# if multi.geom_type ==  'Polygon':
#     ax.add_patch(descartes.PolygonPatch(multi, fc=rng.random(size=3), ec='#000000', alpha=0.5))
# else:
#     for poly in multi.geoms:
#         ax.add_patch(descartes.PolygonPatch(poly, fc=rng.random(size=3), ec='#000000', alpha=0.5))
# for line in route_lines:
#     ax.plot(line.xy[0], line.xy[1], color='black')
# for line in lines_touching_machines:
#     ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.5)
# for line in lines_to_machines:
#     ax.plot(line.xy[0], line.xy[1], color='red', alpha=0.5)

# # for point in bb_points:
# #     ax.scatter(point.xy[0], point.xy[1], color='red')
# #ax.add_patch(descartes.PolygonPatch(allEdges, fc='blue', ec='#000000', alpha=0.5))  
# plt.show()

# Find closest points in voronoi cells
hitpoints = points + list(MultiPoint(walkableArea.exterior.coords).geoms)
hit_tree = STRtree(hitpoints)

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
    minPathWidth = min(first_distance, second_distance)

    G.add_node(first_str, pos=firstTuple, pathwidth=minPathWidth)
    G.add_node(second_str, pos=secondTuple, pathwidth=minPathWidth)
    G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=minPathWidth)



nextTime = time.perf_counter()
print(f"Network generation {nextTime - starttime}")
starttime = nextTime

#Find largest connected component to filter out "loose" parts
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])
#Drop dead ends
G = nx.k_core(G)

nextTime = time.perf_counter()
print(f"Network Filtering {nextTime - starttime}")
starttime = nextTime

pos=nx.get_node_attributes(G,'pos')



fig, ax = plt.subplots(1,figsize=(16, 16))
plt.xlim(0,WIDTH)
plt.ylim(0,HEIGHT)
plt.autoscale(False)
if multi.geom_type ==  'Polygon':
    ax.add_patch(descartes.PolygonPatch(multi, fc=rng.random(size=3), ec='#000000', alpha=0.5))
else:
    for poly in multi.geoms:
        ax.add_patch(descartes.PolygonPatch(poly, fc=rng.random(size=3), ec='#000000', alpha=0.5))
for line in route_lines:
    ax.plot(line.xy[0], line.xy[1], color='red', alpha=0.0)
    for point in line.boundary.geoms:
        nearest_point = hit_tree.nearest_geom(point)
        #ax.plot([point.x, nearest_point.x], [point.y, nearest_point.y], color='green', alpha=1)
       #ax.add_patch(plt.Circle((point.x, point.y), point.distance(nearest_point), color='blue', fill=False, alpha=0.3))
for line in lines_to_machines:
    ax.plot(line.xy[0], line.xy[1], color='green', alpha=0.0)
# for line in route_lines:
#     ax.plot(line.xy[0], line.xy[1], color='black')
# for point in bb_points.geoms:
#     ax.scatter(point.xy[0], point.xy[1], color='red')

pathwidth = np.array(list((nx.get_edge_attributes(G,'pathwidth').values())))
nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=0, )
nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="silver", width=pathwidth * 50, edge_cmap=plt.cm.Greys, alpha=0.7)
nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=pathwidth, width=2, edge_cmap=plt.cm.Spectral)

if SAVEFILE: plt.savefig("Pathwidth.svg", format="svg")
plt.show()

nextTime = time.perf_counter()
print(f"Plotting {nextTime - starttime}")



# %%

#voronoiBase_ = GeometryCollection([walkableArea, voronoiArea.geoms[0]])
#voronoiArea_ = MultiLineString(triangulate(voronoiBase, edges=True))



fig, ax = plt.subplots(1,figsize=(16, 16))
plt.xlim(0,WIDTH)
plt.ylim(0,HEIGHT)
plt.autoscale(False)


if multi.geom_type ==  'Polygon':
    ax.add_patch(descartes.PolygonPatch(multi, fc=rng.random(size=3), ec='#000000', alpha=0.5))
else:
    for poly in multi.geoms:
        ax.add_patch(descartes.PolygonPatch(poly, fc=rng.random(size=3), ec='#000000', alpha=0.5))

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

if SAVEFILE: plt.savefig("Pathwidth_Calculation.svg", format="svg")
plt.show()


# %% Filter narrow edges
F = G.copy()
F.remove_edges_from([(n1, n2) for n1, n2, w in F.edges(data="pathwidth") if w < 1])
#Find largest connected component to filter out "loose" parts
Fcc = sorted(nx.connected_components(F), key=len, reverse=True)
F = F.subgraph(Fcc[0])


fig, ax = plt.subplots(1,figsize=(16, 16))
plt.xlim(0,WIDTH)
plt.ylim(0,HEIGHT)
plt.autoscale(False)


if multi.geom_type ==  'Polygon':
    ax.add_patch(descartes.PolygonPatch(multi, fc=rng.random(size=3), ec='#000000', alpha=0.5))
else:
    for poly in multi.geoms:
        ax.add_patch(descartes.PolygonPatch(poly, fc=rng.random(size=3), ec='#000000', alpha=0.5))


weights = np.array(list((nx.get_edge_attributes(F,'weight').values())))
pathwidth = np.array(list((nx.get_edge_attributes(F,'pathwidth').values())))

crossroads = [node for node, degree in F.degree() if degree >= 3]
endpoints = [node for node, degree in F.degree() if degree == 1]
nx.draw_networkx_nodes(F, pos=pos, ax=ax, nodelist=crossroads[:1], node_size=50, node_color='red')
nx.draw_networkx_nodes(F, pos=pos, ax=ax, node_size=5, node_color='black')
nx.draw_networkx_nodes(F, pos=pos, ax=ax, nodelist=endpoints[:1], node_size=50, node_color='blue')
nx.draw_networkx_edges(F, pos=pos, ax=ax, edge_color="silver", width=pathwidth * 50, edge_cmap=plt.cm.Greys, alpha=0.7)
nx.draw_networkx_edges(F, pos=pos, ax=ax, edge_color=pathwidth, width=2, edge_cmap=plt.cm.Spectral)

if SAVEFILE: plt.savefig("Deadends.svg", format="svg")
plt.show()

# %%

print(nx.shortest_path_length(G, source=crossroads[0], target=endpoints[3], weight="weight"))
# %%

# %%
