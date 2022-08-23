from array import array

import cairo
import moderngl
import moderngl_window as mglw

import numpy as np
import pickle

from shapely.geometry import Point, MultiPoint, MultiPolygon, box, GeometryCollection
from shapely.affinity import rotate, scale, translate
import networkx as nx
from shapely.strtree import STRtree
from shapely.prepared import prep
from shapely.ops import  voronoi_diagram,  unary_union, nearest_points

from helpers import pruneAlongPath, findSupportNodes

from concurrent.futures import ThreadPoolExecutor


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





class factorySimLive(mglw.WindowConfig):
    title = "factorySimLive"
    lastTime = 1
    fps_counter = 30
    #window_size = (3840, 2160)
    #window_size = (1920, 1080)
    window_size = (1280, 720)
    aspect_ratio = None
    fullscreen = False
    resizable = True
    selected = None
    currentScale = 1.0
    is_darkmode = False
    is_dirty = False
    is_calculating = False
    update_during_calculation = False
    

    


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rng = np.random.default_rng()
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.create_factory(load=False)
        self.currentScale = self.scale_factory()

        self.future = self.executor.submit(self.create_Paths)
        
        self.colors = [self.rng.random(size=3) for _ in self.multi.geoms]
        self.G, self.I = self.future.result()

        self.prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec2 in_texcoord_0;
            out vec2 uv;
            void main() {
                gl_Position = vec4(in_position, 1.0);
                uv = in_texcoord_0;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D texture0;
            in vec2 uv;
            out vec4 outColor;
            void main() {
                outColor = texture(texture0, uv);
            }
            """,
        )
        # Create a simple screen rectangle. The texture coordinates
        # are reverted on the y axis here to make the cairo texture appear correctly.
        vertices = [
            # x, y | u, v
            -1,  1,  0, 0,
            -1, -1,  0, 1,
             1,  1,  1, 0,
             1, -1,  1, 1,
        ]
        self.screen_rectangle = self.ctx.vertex_array(
            self.prog,
            [
                (
                    self.ctx.buffer(array('f', vertices)),
                    '2f 2f',
                    'in_position', 'in_texcoord_0',
                )
            ],
        )


    def resize(self, width: int, height: int):
        self.window_size = (width, height)
        self.currentScale = self.scale_factory()

    def close(self):
        print("closing")
        self.executor.shutdown()
        

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        # Key presses

        if action == keys.ACTION_PRESS:
            # Toggle Fullscreen
            if key == keys.F:
                self.wnd.fullscreen = not self.wnd.fullscreen
            
            # Zoom
            if key == 43: # +
                self.currentScale += 0.2
            if key == keys.MINUS:
                self.currentScale -= 0.2
            # Darkmode
            if key == keys.B:
                self.is_darkmode = not self.is_darkmode

            # Toggle mouse exclusivity
            if key == keys.M:
                self.wnd.mouse_exclusivity = not self.wnd.mouse_exclusivity


    def mouse_drag_event(self, x, y, dx, dy):
        if self.selected is not None: # selected can be `0` so `is not None` is required
            # move object  
            temp = list(self.multi.geoms)
            temp_x = temp[self.selected].bounds[0]
            temp_y = temp[self.selected].bounds[1] # max y for shapely, min y for pygame coordinates
            temp[self.selected] = translate(self.multi.geoms[self.selected], 
                xoff=((x / self.currentScale)- temp_x) + self.selected_offset_x,
                yoff=((y / self.currentScale) - temp_y) + self.selected_offset_y
                )            
            self.multi = MultiPolygon(temp)
            self.is_dirty = True
            if self.is_calculating:
                self.update_during_calculation = True


    def mouse_press_event(self, x, y, button):
        if button == 1:
            for i, poly in enumerate(self.multi.geoms):
                point_scaled = Point(x/self.currentScale, y/self.currentScale)
                if poly.contains(point_scaled):
                    self.selected = i
                    self.selected_offset_x = poly.bounds[0] - point_scaled.x
                    self.selected_offset_y = poly.bounds[1] - point_scaled.y


    def mouse_release_event(self, x: int, y: int, button: int):
        if button == 1:
            self.selected = None

    def render(self, time, frame_time):
        if time > self.lastTime + 0.5:
            self.fps_counter = 1/(frame_time+0.00000001)
            self.lastTime = time

        texture = self.render_cairo_to_texture()
        texture.use(location=0)
        self.screen_rectangle.render(mode=moderngl.TRIANGLE_STRIP)
        texture.release()


    def render_cairo_to_texture(self):
        # Draw with cairo to surface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.window_size[0], self.window_size[1])
        cctx = cairo.Context(surface)
        
        cctx.scale(self.currentScale, self.currentScale)

        self.draw_BG(cctx)
        
        for poly, color in zip(self.multi.geoms, self.colors):
            self.draw_poly(cctx, poly, color)

        if self.is_dirty:
            if self.is_calculating:
                if self.future.done():
                    self.G, self.I = self.future.result()
                    self.is_dirty = False
                    self.is_calculating = False
                    #if we had changes during last calulation, recalulate
                    if self.update_during_calculation:
                        self.update_during_calculation = False
                        self.is_dirty = True
                        self.is_calculating = True
                        self.future = self.executor.submit(self.create_Paths)
            else:
                self.future = self.executor.submit(self.create_Paths)
                self.is_calculating = True


        self.draw_paths(cctx, self.G, self.I)
        self.draw_fps(cctx, self.fps_counter)

        # Copy surface to texture
        texture = self.ctx.texture((self.window_size[0], self.window_size[1]), 4, data=surface.get_data())
        texture.swizzle = 'BGRA' # use Cairo channel order (alternatively, the shader could do the swizzle)
        surface.finish()
        del(cctx)
        del(surface)
        return texture



#--------------------------------------------------------------------------------------------------------------------------------

    def draw_fps(self, ctx, fps):

        ctx.move_to(*ctx.device_to_user_distance(20, 20))
        ctx.set_font_size(ctx.device_to_user_distance(12, 12)[0])
        if self.is_darkmode:
            ctx.set_source_rgb(1.0, 1.0, 1.0)
        else:
            ctx.set_source_rgba(0.0, 0.0, 0.0)

        ctx.show_text(f"{fps:.0f}")


    def draw_BG(self, ctx):
        ctx.rectangle(0, 0, self.window_size[0], self.window_size[1])
        #Color is actually BGR in Pygame
        if self.is_darkmode:
            ctx.set_source_rgba(0.0, 0.0, 0.0)
        else:
            ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.fill()


    def draw_rect(self, ctx, rect, color):
        ctx.rectangle(rect.left, rect.top, rect.width, rect.height)
        #Color is actually BGR in Pygame
        ctx.set_source_rgba(*color, 0.8)
        ctx.fill()

    def draw_poly(self, ctx, poly, color):
        ctx.set_source_rgba(*color, 0.8)
        ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        ctx.move_to(*poly.exterior.coords[0])
        for x,y in poly.exterior.coords[1:]:
            ctx.line_to(x,y)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.stroke()

    def draw_paths(self, ctx, G, I):

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
                ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
                ctx.move_to(*temp[0])
                for node in temp[1:]:
                    ctx.line_to(*node)
        ctx.set_dash(list(ctx.device_to_user_distance(10, 10)))
        ctx.stroke()
        ctx.set_dash([])

    def create_factory(self, load = False):
        polygons = []

        if load:
            loaddata = pickle.load( open( "FigureFactory.p", "rb" ) )
            self.bb = loaddata["bounding_box"]
            polygons = loaddata["machines"]
        else:
            self.bb = box(0,0,WIDTH,HEIGHT)

            lowerLeftCornersRect = self.rng.integers([0,0], [WIDTH - MAXSHAPEWIDTH, HEIGHT - MAXSHAPEHEIGHT], size=[AMOUNTRECT,2], endpoint=True)
            lowerLeftCornersPoly = self.rng.integers([0,0], [WIDTH - MAXSHAPEWIDTH, HEIGHT - MAXSHAPEWIDTH], size=[AMOUNTPOLY,2], endpoint=True)

            
            #Create Recangles
            for x,y in lowerLeftCornersRect:
                singlePoly = box(x,y,x + self.rng.integers(1, MAXSHAPEWIDTH+1), y + self.rng.integers(1, MAXSHAPEHEIGHT+1))
                singlePoly= rotate(singlePoly, self.rng.choice([0,90,180,270]))  
                polygons.append(singlePoly)

            #Create Convex Polygons
            for x,y in lowerLeftCornersPoly: 
                corners = []
                corners.append([x,y]) # First Corner
                for _ in range(self.rng.integers(2,MAXCORNERS+1)):
                    corners.append([x + self.self.rng.integers(0, MAXSHAPEWIDTH+1), y + self.rng.integers(0, MAXSHAPEWIDTH+1)])

                singlePoly = self.multiPoint(corners).minimum_rotated_rectangle
                singlePoly= rotate(singlePoly, self.rng.integers(0,361))  
                #Filter Linestrings
                if singlePoly.geom_type ==  'Polygon':
                #Filter small Objects
                    if singlePoly.area > MAXSHAPEWIDTH*MAXSHAPEWIDTH*0.05:
                        polygons.append(singlePoly)
        # Flip on y because Pygames origin is in the top left corner
        self.multi = scale(unary_union(polygons), yfact=-1, origin=self.bb.centroid)




    def scale_factory(self):
        boundingBox = self.bb.bounds   
        min_value_x = boundingBox[0]     
        max_value_x = boundingBox[2]     
        min_value_y = boundingBox[1]     
        max_value_y = boundingBox[3]     

        scale_x = self.window_size[0] / (max_value_x - min_value_x)
        scale_y = self.window_size[1] / (max_value_y - min_value_y)
        sugesstedscale = min(scale_x, scale_y)

        return sugesstedscale





    def create_Paths(self):
        walkableArea = self.bb.difference(unary_union(self.multi))
        if walkableArea.geom_type ==  'self.multiPolygon':
            walkableArea = walkableArea.geoms[0]


        #   Create Voronoi -----------------------------------------------------------------------------------------------------------------
        #Points around boundary

        distances = np.arange(0,  self.bb.boundary.length, BOUNDARYSPACING)
        points = [self.bb.boundary.interpolate(distance) for distance in distances]

        #Points on Machines
        distances = np.arange(0,  self.multi.boundary.length, BOUNDARYSPACING)
        points.extend([ self.multi.boundary.interpolate(distance) for distance in distances])
        self.bb_points = unary_union(points) 

        voronoiBase = GeometryCollection([walkableArea, self.bb_points])
        voronoiArea = voronoi_diagram(voronoiBase, edges=True)

        route_lines = []
        lines_touching_machines = []


        processed_multi = prep(self.multi)
        processed_bb = prep(self.bb)

        for line in voronoiArea.geoms[0].geoms:
            #find routes close to machines
            if not (processed_multi.intersects(line) or processed_bb.crosses(line)): 
                route_lines.append(line)

        # Find closest points in voronoi cells
        if walkableArea.geom_type ==  'MultiPolygon':
            exteriorPoints = []
            for x in walkableArea.geoms:
                exteriorPoints.extend(list(x.exterior.coords))
        else:
            exteriorPoints = list(walkableArea.exterior.coords)
        hitpoints = points + list(MultiPoint(exteriorPoints).geoms)
        #hitpoints = self.multiPoint(points+list(walkableArea.exterior.coords))
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

        narrowPaths = [(n1, n2) for n1, n2, w in G.edges(data="pathwidth") if w < MINPATHWIDTH]
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

        shortDeadEnds = pruneAlongPath(G, starts=old_endpoints, ends=old_crossroads, min_length=MINDEADEND_LENGTH)

        G.remove_nodes_from(shortDeadEnds)
        endpoints = [node for node, degree in G.degree() if degree == 1]
        crossroads = [node for node, degree in G.degree() if degree >= 3]

        # Prune unused dead ends
        pos=nx.get_node_attributes(G,'pos')

        repPoints = [poly.representative_point() for poly in self.multi.geoms]
        #Create Positions lists for nodes, since we need to querry shapley for shortest distance
        endpoint_pos = [pos[endpoint] for endpoint in endpoints ]
        crossroad_pos = [pos[crossroad] for crossroad in crossroads]
        total = endpoint_pos + crossroad_pos

        endpoints_to_prune = endpoints.copy()

        for point in repPoints:
            hit = nearest_points(point, MultiPoint(total))[1]
            key = str((hit.x, hit.y))
            if key in endpoints_to_prune: endpoints_to_prune.remove(key)

        nodes_to_prune = pruneAlongPath(G, starts=endpoints_to_prune, ends=crossroads, min_length=10)

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
                        if minpath > MINTWOWAYPATHWIDTH: pathtype = "twoway"


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










if __name__ == "__main__":
    factorySimLive.run()
