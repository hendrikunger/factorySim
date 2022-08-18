import pygame
import cairo
import numpy as np
import pickle

from shapely.geometry import MultiPoint, MultiPolygon, box, GeometryCollection
from shapely.affinity import rotate, scale, translate
import networkx as nx
from shapely.strtree import STRtree
from shapely.prepared import prep
from shapely.ops import split,  voronoi_diagram,  unary_union, nearest_points

from helpers import pruneAlongPath, findSupportNodes
 
SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080



WIDTH = 32
HEIGHT = 32
MAXSHAPEWIDTH = 4
MAXSHAPEHEIGHT = 4
AMOUNTRECT = 25
AMOUNTPOLY = 0
MAXCORNERS = 3

MINDEADEND_LENGTH = 2.0 # If Deadends are shorter than this, they are deleted
MINPATHWIDTH = 1.0  # Minimum Width of a Road to keep
MINTWOWAYPATHWIDTH = 2.0  # Minimum Width of a Road to keep
BOUNDARYSPACING = 1.5  # Spacing of Points used as Voronoi Kernels
SIMPLIFICATION_ANGLE = 35 # Angle in degrees, used for support point calculation in simple path




def draw_BG(ctx):
    ctx.rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    #Color is actually BGR in Pygame
    if is_darkmode:
        ctx.set_source_rgba(0.0, 0.0, 0.0)
    else:
        ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.fill()


def draw_rect(ctx, rect, color):
    ctx.rectangle(rect.left, rect.top, rect.width, rect.height)
    #Color is actually BGR in Pygame
    ctx.set_source_rgba(*color, 0.8)
    ctx.fill()

def draw_poly(ctx, poly, color):
    ctx.set_source_rgba(*color, 0.8)
    ctx.set_line_width(1)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    ctx.move_to(*poly.exterior.coords[0])
    for x,y in poly.exterior.coords[1:]:
        ctx.line_to(x,y)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.stroke()

def draw_paths(ctx, G, I):

    ctx.set_source_rgba(0.3, 0.3, 0.3, 1.0)
    ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    pos=nx.get_node_attributes(G,'pos')

    for u,v,data in I.edges(data=True):
        temp = [pos[x] for x in data['nodelist']]
        ctx.set_line_width(data['pathwidth'])
        ctx.move_to(*temp[0])
        for node in temp[1:]:
            ctx.line_to(*node)
        ctx.stroke()
    ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)

    for u,v,data in I.edges(data=True):
        temp = [pos[x] for x in data['nodelist']]
        if data['pathtype'] =="twoway":
            ctx.set_line_width(1)
            ctx.move_to(*temp[0])
            for node in temp[1:]:
                ctx.line_to(*node)
    ctx.stroke()

def update_fps():
	fps = str(int(clock.get_fps()))
	fps_text = font.render(fps, 1, pygame.Color("white"))
	return fps_text


def create_factory(load = False):
    polygons = []

    if load:
        loaddata = pickle.load( open( "FigureFactory.p", "rb" ) )
        bb = loaddata["bounding_box"]
        polygons = loaddata["machines"]
    else:
        bb = box(0,0,WIDTH,HEIGHT)

        lowerLeftCornersRect = rng.integers([0,0], [WIDTH - MAXSHAPEWIDTH, HEIGHT - MAXSHAPEHEIGHT], size=[AMOUNTRECT,2], endpoint=True)
        lowerLeftCornersPoly = rng.integers([0,0], [WIDTH - MAXSHAPEWIDTH, HEIGHT - MAXSHAPEWIDTH], size=[AMOUNTPOLY,2], endpoint=True)

        
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
    # Flip on y because Pygames origin is in the top left corner
    multi = scale(unary_union(polygons), yfact=-1, origin=bb.centroid)

    boundingBox = bb.bounds   
    min_value_x = boundingBox[0]     
    max_value_x = boundingBox[2]     
    min_value_y = boundingBox[1]     
    max_value_y = boundingBox[3]     

    if((max_value_x < SCREEN_WIDTH) or (max_value_y < SCREEN_WIDTH)):
        #Calculate new scale
        scale_x = SCREEN_WIDTH / (max_value_x - min_value_x)
        scale_y = SCREEN_HEIGHT / (max_value_y - min_value_y)
        sugesstedscale = min(scale_x, scale_y)
    
    return multi, bb, sugesstedscale


    


def scale_factory(multi, bb, aScale = 1.0, newScale = 1.0):
    reverseScale = 1.0 / aScale if aScale else 1.0
    multi = scale(multi, xfact= reverseScale, yfact=-reverseScale, zfact=-reverseScale, origin=(0,0))
    multi = scale(multi, xfact= newScale, yfact=-newScale, zfact=-newScale, origin=(0,0))
    bb = scale(bb, xfact= reverseScale, yfact=-reverseScale, zfact=-reverseScale, origin=(0,0))
    bb = scale(bb, xfact= newScale, yfact=-newScale, zfact=-newScale, origin=(0,0))

    rects = []
    for poly in multi.geoms:
        rects.append(pygame.Rect(poly.bounds[0], poly.bounds[1], poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]))
    return multi, bb, rects



def create_paths(multi, bb):
    walkableArea = bb.difference(unary_union(multi))
    if walkableArea.geom_type ==  'MultiPolygon':
        walkableArea = walkableArea.geoms[0]


    #   Create Voronoi -----------------------------------------------------------------------------------------------------------------
    #Points around boundary

    distances = np.arange(0,  bb.boundary.length, BOUNDARYSPACING * currentScale)
    points = [ bb.boundary.interpolate(distance) for distance in distances]

    #Points on Machines
    distances = np.arange(0,  multi.boundary.length, BOUNDARYSPACING * currentScale)
    points.extend([ multi.boundary.interpolate(distance) for distance in distances])
    bb_points = unary_union(points) 

    voronoiBase = GeometryCollection([walkableArea, bb_points])
    voronoiArea = voronoi_diagram(voronoiBase, edges=True)

    route_lines = []
    lines_touching_machines = []


    processed_multi = prep(multi)
    processed_bb = prep(bb)

    for line in voronoiArea.geoms[0].geoms:
        #find routes close to machines
        if not (processed_multi.intersects(line) or processed_bb.crosses(line)): 
            route_lines.append(line)

    # Find closest points in voronoi cells
    hitpoints = points + list(MultiPoint(walkableArea.exterior.coords).geoms)
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
        # if smallestPathwidth < MINPATHWIDTH:
        #     continue
        # else:
        #     G.add_node(first_str, pos=firstTuple, pathwidth=smallestPathwidth)
        #     G.add_node(second_str, pos=secondTuple, pathwidth=smallestPathwidth)
        #     G.add_edge(first_str, second_str, weight=first.distance(second), pathwidth=smallestPathwidth)



    # Filter  Graph -----------------------------------------------------------------------------------------------------------------

    narrowPaths = [(n1, n2) for n1, n2, w in G.edges(data="pathwidth") if w < MINPATHWIDTH * currentScale]
    G.remove_edges_from(narrowPaths)

    #Find largest connected component to filter out "loose" parts
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0]).copy()

    #find crossroads
    old_crossroads = [node for node, degree in G.degree() if degree >= 3]
    #Set isCrossroads attribute on cross road nodes
    nx.set_node_attributes(G, dict.fromkeys(old_crossroads, True), 'isCrossroads')
    #find deadends
    old_endpoints = [node for node, degree in G.degree() if degree == 1]

    shortDeadEnds = pruneAlongPath(G, starts=old_endpoints, ends=old_crossroads, min_length=MINDEADEND_LENGTH * currentScale * 2)

    G.remove_nodes_from(shortDeadEnds)
    endpoints = [node for node, degree in G.degree() if degree == 1]
    crossroads = [node for node, degree in G.degree() if degree >= 3]

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

    nodes_to_prune = pruneAlongPath(G, starts=endpoints_to_prune, ends=crossroads, min_length=10 * currentScale)

    G.remove_nodes_from(nodes_to_prune)

    endpoints = [node for node, degree in G.degree() if degree == 1]
    crossroads = [node for node, degree in G.degree() if degree >= 3]

    nx.set_node_attributes(G, findSupportNodes(G, cutoff=SIMPLIFICATION_ANGLE))
    support = list(nx.get_node_attributes(G, "isSupport").keys())


    # Alternative Simpicification and Path Generation ------------------------------------------------------------------------------------
    I = nx.Graph()

    visited = set() # Set to keep track of visited nodes.
    tempPath = [] # List to keep track of visited nodes in current path.
    paths = [] # List to keep track of all paths.

    ep = endpoints
    cross = crossroads
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

        for outerNeighbor in G.neighbors(currentOuterNode):
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
                totalweight += G[lastNode][currentInnerNode]["weight"]
                pathwidth = G[lastNode][currentInnerNode]["pathwidth"]
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
                    if minpath > MINTWOWAYPATHWIDTH * currentScale: pathtype = "twoway"


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

                for innerNeighbor in G.neighbors(currentInnerNode):
                    #Identifying next node (there will at most be two edges connected to every node)
                    if (innerNeighbor == lastNode):
                        #this is last node
                        continue
                    else:
                        #found the next one
                        lastNode = currentInnerNode
                        currentInnerNode = innerNeighbor
                        break
    return G, I

rng = np.random.default_rng()

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.SCALED)
#screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.SCALED | pygame.FULLSCREEN)

cairo_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, SCREEN_WIDTH, SCREEN_HEIGHT)
ctx = cairo.Context(cairo_surface)

font = pygame.font.SysFont("Arial", 18)

selected = None
currentScale = 1.0
appliedScale = None

multi, bb, currentScale = create_factory(load=False)
rects = []

colors = [rng.random(size=3) for _ in multi.geoms]

   
# --- mainloop ---
 
clock = pygame.time.Clock()
is_running = True
is_darkmode = True
is_dirty = True
 
while is_running:
 
    # --- events ---
   
    for event in pygame.event.get():
 
        # --- global events ---
       
        if event.type == pygame.QUIT:
            is_running = False
 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                is_running = False
            if event.key == pygame.K_f:
                pygame.display.toggle_fullscreen()
            if event.key == pygame.K_b:
                is_darkmode = not is_darkmode
 
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for i, r in enumerate(rects):
                    if r.collidepoint(event.pos):
                        selected = i
                        selected_offset_x = r.x - event.pos[0]
                        selected_offset_y = r.y - event.pos[1]
               
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                selected = None
               
        elif event.type == pygame.MOUSEMOTION:
            if selected is not None: # selected can be `0` so `is not None` is required
                # move object
                rects[selected].x = event.pos[0] + selected_offset_x
                rects[selected].y = (event.pos[1]) + selected_offset_y
                temp = list(multi.geoms)
                temp_x = temp[selected].bounds[0]
                temp_y = temp[selected].bounds[1] # max y for shapely, min y for pygame coordinates
                temp[selected] = translate(multi.geoms[selected], 
                    xoff=((event.pos[0] - temp_x) + selected_offset_x),
                    yoff=((event.pos[1] - temp_y) + selected_offset_y)
                    )
                multi = MultiPolygon(temp)
                is_dirty = True

        pressed_keys = pygame.key.get_pressed()
 
        if pressed_keys[pygame.K_PLUS]:
            currentScale += 0.1
        if pressed_keys[pygame.K_MINUS]:
            currentScale -= 0.1
               
        # --- objects events ---
    if currentScale != appliedScale:
        multi, bb, rects = scale_factory(multi, bb, appliedScale, currentScale)
        is_dirty = True
        appliedScale = currentScale
       
    # --- draws ---

    # draw rect
    
    draw_BG(ctx)

    # for r in rects:
    #     draw_rect(ctx, r, rng.random(size=3))
    
    for poly, color in zip(multi.geoms, colors):
        draw_poly(ctx, poly, color)

    if is_dirty:
        G, I = create_paths(multi, bb)
        is_dirty = False

    draw_paths(ctx, G, I)

    buf = cairo_surface.get_data()
    # Convert Color order, but pay performance penalty
    #rgb = np.ndarray(shape=(SCREEN_WIDTH, SCREEN_HEIGHT,4), dtype=np.uint8, buffer=buf)[...,[3,2,1,0]]
    #rgb= np.ascontiguousarray(rgb)
    #pygame_surface = pygame.image.frombuffer(rgb, (SCREEN_WIDTH, SCREEN_HEIGHT), 'ARGB')
    pygame_surface = pygame.image.frombuffer(buf, (SCREEN_WIDTH, SCREEN_HEIGHT), 'RGBA')
    screen.blit(pygame_surface, (0,0))
    screen.blit(update_fps(), (10,0)) 
    
    pygame.display.flip()

#Set FPS 
    clock.tick()
    
pygame.quit()