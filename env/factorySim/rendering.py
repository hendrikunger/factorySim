import cairo
import networkx as nx
import numpy as np
from shapely.ops import polylabel



def draw_BG(ctx, startpos, width, height, darkmode=True):
    ctx.rectangle(*ctx.device_to_user_distance(*startpos), *ctx.device_to_user_distance(width,height))
    if darkmode:
        ctx.set_source_rgba(0.0, 0.0, 0.0)
    else:
        ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.fill()

#------------------------------------------------------------------------------------------------------------
def draw_detail_paths(ctx, fullPathGraph, reducedPathGraph, asStreets=False):
    ctx.set_source_rgba(0.3, 0.3, 0.3, 1.0)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    if fullPathGraph and reducedPathGraph:
        modifer = 0.5 if asStreets else 1.0
        pos = nx.get_node_attributes(fullPathGraph,'pos')
        for u,v,data in reducedPathGraph.edges(data=True):
            if data.get("isMachineConnection", False):
                ctx.set_line_width(ctx.device_to_user_distance(2, 1)[0])
            else:
                ctx.set_line_width(data['pathwidth']*modifer)
            nodelist = [pos[node] for node in data["nodelist"]]
            ctx.move_to(*nodelist[0])
            for position, key in zip(nodelist[1:], data["nodelist"][1:]):
                ctx.line_to(*position)
                node_data = fullPathGraph.nodes[key]
                if "edge_angle" in node_data:
                    #print(node_data["edge_angle"], node_data["arcstart"], node_data["arcend"]) 
                    ctx.arc(*position, 10, node_data["arcstart"], node_data["arcend"])
            ctx.stroke()        
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)

        for u,v,data in reducedPathGraph.edges(data=True):
            if data.get("isMachineConnection", False): continue
            if data['pathtype'] =="twoway":
                nodelist = [pos[node] for node in data["nodelist"]]
                ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
                ctx.move_to(*nodelist[0])
                for position in nodelist[1:]:
                    ctx.line_to(*position)
        ctx.set_dash(list(ctx.device_to_user_distance(10, 10)))
        ctx.stroke()
        ctx.set_dash([])

    return ctx

#------------------------------------------------------------------------------------------------------------
def draw_simple_paths(ctx, fullPathGraph, reducedPathGraph):
    if fullPathGraph and reducedPathGraph:
        ctx.set_source_rgba(0.0, 0.0, 0.5, 0.5)
        ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        pos=nx.get_node_attributes(fullPathGraph,'pos')

        for u,v in reducedPathGraph.edges():
            ctx.set_line_width(ctx.device_to_user_distance(10, 10)[0])
            ctx.move_to(*pos[u])
            ctx.line_to(*pos[v])
        ctx.stroke()

        
        for node in reducedPathGraph.nodes():
            node_data = fullPathGraph.nodes[node]
            if "isMachineConnection" in node_data:
                ctx.set_source_rgba(1.0, 1.0, 0.0, 1.0)
            elif "isCrossroads" in node_data:
                ctx.set_source_rgba(0.0, 1.0, 0.0, 1.0)
            else:
                ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
            ctx.move_to(*pos[node])
            ctx.arc(*pos[node], ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
            ctx.fill()
        
        #crossroads = list(nx.get_node_attributes(G, "isCrossroads").keys())

        # endpoints=[node for node, degree in fullPathGraph.degree() if degree == 1]
        # crossroads= [node for node, degree in fullPathGraph.degree() if degree >= 3]

        # ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
        # for point in crossroads:
        #     ctx.move_to(*pos[point])
        #     ctx.arc(*pos[point], ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
        # ctx.fill()

        # ctx.set_source_rgba(0.0, 1.0, 0.0, 1.0)
        # for point in endpoints:
        #     ctx.move_to(*pos[point])
        #     ctx.arc(*pos[point], ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
        # ctx.fill()
    return ctx

#------------------------------------------------------------------------------------------------------------
def draw_node_angles(ctx, fullPathGraph, reducedPathGraph):
    ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
    ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])

    if fullPathGraph and reducedPathGraph:

        for u,v,data in reducedPathGraph.edges(data=True):

            for node in data['nodelist']:

                node_data = fullPathGraph.nodes[node]
                if not "isMachineConnection" in node_data:
                    ctx.move_to(*node_data["pos"])
                    ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
                    ctx.arc(*node_data["pos"], ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
                    ctx.fill()  
                if "edge_angle" in node_data:
                    r = ctx.device_to_user_distance(30, 30)[0]
                    ctx.move_to(*node_data["pos"])
                    if node_data["arcstart"] > node_data["arcend"]:
                        ctx.arc_negative(node_data["pos"][0], node_data["pos"][1], r, node_data["arcstart"], node_data["arcend"])
                        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
                        
                    else:
                        ctx.arc_negative(node_data["pos"][0], node_data["pos"][1], r, node_data["arcstart"], node_data["arcend"])
                        ctx.set_source_rgba(0.0, 1.0, 0.0, 1.0)
                        
                    ctx.close_path()
                    ctx.stroke() 
                    ctx.move_to(*node_data["pos"])
                    ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                    ctx.show_text(str(round(node_data["edge_angle"], 2)))
  
                
 
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)


    return ctx
#------------------------------------------------------------------------------------------------------------
def draw_poly(ctx, poly, color, text:str=None, highlight=False, drawHoles=True):
    if poly and not poly.is_empty:
        for subpoly in poly.geoms:

            ctx.move_to(*subpoly.exterior.coords[0])
            for x,y in subpoly.exterior.coords[1:]:
                ctx.line_to(x,y)
            ctx.close_path()

            if drawHoles:
                for loop in subpoly.interiors:
                    ctx.move_to(*loop.coords[0])
                    for x,y in loop.coords[1:]:
                        ctx.line_to(x,y)
                ctx.close_path()

            if highlight:
                ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
            else:
                ctx.set_source_rgba(*color)
            ctx.fill()


        if text:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            ctx.set_font_size(ctx.device_to_user_distance(14, 14)[0])
            (x, y, width, height, dx, dy) = ctx.text_extents(text)
            point = polylabel(poly.convex_hull, tolerance=1000)
            ctx.move_to(point.x - width/2, point.y - height/2)    
            ctx.show_text(text)
        return ctx
#------------------------------------------------------------------------------------------------------------
def draw_points(ctx, points, color):

        for point in points:
            ctx.move_to(point.x, point.y)
            ctx.arc(point.x, point.y, ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)

        ctx.set_source_rgba(*color)
        ctx.fill()
        return ctx
#------------------------------------------------------------------------------------------------------------
def draw_pathwidth_circles(ctx, fullPathGraph):
    if fullPathGraph:
        for _ , data in fullPathGraph.nodes(data=True):
            if "pathwidth" in data:
                ctx.move_to(data['pos'][0] + data['pathwidth']/2, data['pos'][1])
                ctx.arc(*data['pos'], data['pathwidth']/2, 0, 2*np.pi)
        ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
        ctx.set_source_rgba(0.0, 0.0, 0.8, 0.8)
        ctx.stroke()
    return ctx

def draw_pathwidth_circles2(ctx, fullPathGraph, reducedPathGraph):
    if fullPathGraph and reducedPathGraph:
        pathwidth = (nx.get_node_attributes(fullPathGraph,'pathwidth'))
        for node , data in reducedPathGraph.nodes(data=True):
            if "pathwidth" in data:
                ctx.move_to(data['pos'][0] + pathwidth[node]/2, data['pos'][1])
                ctx.arc(*data['pos'], pathwidth[node]/2, 0, 2*np.pi)
        ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
        ctx.set_source_rgba(0.0, 0.0, 0.8, 0.8)
        ctx.stroke()
    return ctx
#------------------------------------------------------------------------------------------------------------
def draw_route_lines(ctx, route_lines):
    for line in route_lines.geoms:
        ctx.move_to(*line.coords[0])
        for x,y in line.coords[1:]:
            ctx.line_to(x,y)
    ctx.set_line_width(ctx.device_to_user_distance(3, 3)[0])
    ctx.set_source_rgba(0.5, 0.5, 0.5, 1.0)
    ctx.stroke()
    return ctx

#------------------------------------------------------------------------------------------------------------
def drawFactory(ctx, factory, materialflow_file=None, drawColors = True, drawNames = True, darkmode = True, drawWalls = True, drawMachineCenter = False, drawOrigin = False, highlight = None, isObs = False):   
    
    #Walls
    if factory.wall_dict and drawWalls:
        ctx.set_fill_rule(cairo.FillRule.EVEN_ODD)
        for wall in factory.wall_dict.values():
            #draw all walls
            for  poly in wall.poly.geoms:
                ctx.set_source_rgba(0.2, 0.2, 0.2, 1.0)
                ctx.move_to(*poly.exterior.coords[0])
                for point in poly.exterior.coords[1:]:  
                    ctx.line_to(point[0], point[1])
                ctx.close_path()
                ctx.fill()
            #draw all holes
                if darkmode:
                    ctx.set_source_rgba(0.0, 0.0, 0.0)
                else:
                    ctx.set_source_rgb(1.0, 1.0, 1.0)
                for loop in poly.interiors:
                    ctx.move_to(*loop.coords[0])
                    for point in loop.coords[1:]:
                        ctx.line_to(point[0], point[1])
                    ctx.close_path()
                    ctx.fill()
                        
    #draw machine positions
    if factory.machine_dict:
        ctx.set_fill_rule(cairo.FillRule.WINDING)
        ctx.set_line_width(ctx.device_to_user_distance(5, 5)[0])
        ctx.set_dash([])
        for index, machine in enumerate(factory.machine_dict.values()):
            for poly in machine.poly.geoms:
                ctx.move_to(*poly.exterior.coords[0])
                for point in poly.exterior.coords[1:]: 
                    ctx.line_to(point[0], point[1])
                ctx.close_path()
                if drawColors:
                    #highlighted machine
                    if(machine.gid == highlight):
                        ctx.set_source_rgba(0.5, 0.5, 0.5, 1.0)
                        ctx.set_dash(list(ctx.device_to_user_distance(10, 10)))
                        ctx.stroke_preserve()
                        ctx.set_dash([])
                        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
                    #other machines
                    else:
                        #make machines far from path transparent
                        if machine.gid in factory.MachinesFarFromPath:
                            ctx.set_source_rgba(0.8,0.8,0.8,1.0)
                            ctx.stroke_preserve()
                            ctx.set_source_rgba(machine.color[0], machine.color[1], machine.color[2], 0.1)
                        else:
                            ctx.set_source_rgba(machine.color[0], machine.color[1], machine.color[2], 1.0)
                    #other machines
                else:
                    #highlighted machine
                    if(index == highlight or machine.gid == highlight):
                        ctx.set_source_rgba(0.9, 0.9, 0.9, 1.0)
                    #other machines
                    else:
                        #make machines far from path transparent
                        if machine.gid in factory.MachinesFarFromPath:
                            ctx.set_source_rgba(0.8, 0.8, 0.8, 1.0) 
                            ctx.stroke_preserve()
                        else:
                            ctx.set_source_rgba(0.4, 0.4, 0.4, 1.0)                         
                ctx.fill()

                if drawNames:
                    ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                    ctx.set_font_size(ctx.device_to_user_distance(14, 14)[0])
                    (x, y, width, height, dx, dy) = ctx.text_extents(str(machine.name))
                    point = polylabel(poly, tolerance=1000)
                    ctx.move_to(point.x - width/2, point.y + height/2) 
                    ctx.show_text(str(machine.name))

        #Machine Centers
            if (drawMachineCenter):
                ctx.set_source_rgb(0, 0, 0)
                ctx.arc(machine.center.x, machine.center.y, ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
                ctx.fill()

        #Machine Origin 
            if (machine.origin is not None and drawOrigin):
                if drawColors:
                    ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                else:
                    ctx.set_source_rgb(0.5, 0.5, 0.5)
                ctx.arc(machine.origin[0], machine.origin[1], ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
                ctx.fill()
    ctx.set_dash([])

    #Material Flow
    drawMaterialFlow(ctx, factory.machine_dict, materialflow_file, drawColors, isObs=isObs)
    
    return ctx

def drawMaterialFlow(ctx, machine_dict,  materialflow_file=None, drawColors = True, isObs=False):
    if  materialflow_file is not None:

        for row in materialflow_file.itertuples():
            current_source_Machine = machine_dict[row.source]
            current_target_Machine = machine_dict[row.target]
            try:
                if(drawColors):
                    ctx.set_source_rgba(*current_source_Machine.color, 0.7)
                else:
                    ctx.set_source_rgba(0.6, 0.6, 0.6)

                ctx.move_to(current_source_Machine.center.x, current_source_Machine.center.y)
                ctx.line_to(current_target_Machine.center.x, current_target_Machine.center.y)
                if isObs:
                    modifer = 3.0
                else:
                    modifer = 20.0
                ctx.set_line_width(ctx.device_to_user_distance(row.intensity_sum_norm * modifer, 0)[0] )
                ctx.stroke()   
            except KeyError:
                print(f"Error in Material Flow Drawing - Machine {row.source} or {row.target} not defined")
                continue
    

    return ctx

def drawRoutedMaterialFlow(ctx, machine_dict, fullPathGraph, reducedPathGraph, materialflow_file=None, drawColors = True, selected=None):
    if  materialflow_file is not None and fullPathGraph:
        pos = nx.get_node_attributes(fullPathGraph,'pos')

        for row in materialflow_file.itertuples():
            if selected is not None and row.source != selected:
                continue
            current_source_Machine = machine_dict[row.source]

            for index, node in enumerate(row.routes[0:-1]):
                if(drawColors):
                    ctx.set_source_rgba(*current_source_Machine.color, 0.7)
                else:
                    ctx.set_source_rgba(0.6, 0.6, 0.6) 
                nodelist = reducedPathGraph.edges[node, row.routes[index + 1]]['nodelist']
                ctx.move_to(*pos[nodelist[0]])
                for nodeToDraw in nodelist[1:]:
                    ctx.line_to(*pos[nodeToDraw]) 
                ctx.set_line_width(ctx.device_to_user_distance(row.intensity_sum_norm * 20.0, 0)[0] )
                ctx.stroke()      

    return ctx

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
def drawCollisions(ctx, machineCollisionList, wallCollisionList=None, outsiderList=None, drawColors = True):
    #Drawing collisions between machines
    if(drawColors):
        ctx.set_source_rgb(1.0, 0.3, 0.0)
    else:
        ctx.set_source_rgb(0.7, 0.7, 0.7)
    #Drawing collisions between machines and machines
    for collision in machineCollisionList:
        for poly in collision.geoms:
            ctx.move_to(*poly.exterior.coords[0])
            for point in poly.exterior.coords[1:]: 
                ctx.line_to(point[0], point[1])
            ctx.close_path()
        ctx.fill()
                
    #Drawing collisions between machines and walls
    if(wallCollisionList):
        if(drawColors):
            ctx.set_source_rgb(1.0, 0.3, 0.0)
        else:
            ctx.set_source_rgb(0.7, 0.7, 0.7)
        for collision in wallCollisionList:
            for poly in collision.geoms:
                ctx.move_to(*poly.exterior.coords[0])
                for point in poly.exterior.coords[1:]: 
                    ctx.line_to(point[0], point[1])
                ctx.close_path()
            ctx.fill()
        #Drawing outsider
    if(outsiderList): 
        if(drawColors):
            ctx.set_source_rgb(1.0, 0.3, 0.0)
        else:
            ctx.set_source_rgb(0.7, 0.7, 0.7)
        for outsider in outsiderList:
            for poly in outsider.geoms:
                ctx.move_to(*poly.exterior.coords[0])
                for point in poly.exterior.coords[1:]: 
                    ctx.line_to(point[0], point[1])
                ctx.close_path()
            ctx.fill()     

    return ctx

#------------------------------------------------------------------------------------------------------------
def draw_text(ctx, text, color, pos, center=False, rightEdge=False, factoryCoordinates=True, input_width=None):
    width = input_width
    ctx.set_font_size(ctx.device_to_user_distance(12, 12)[0])
    #select monospaced font
    ctx.select_font_face("Consolas", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    if not factoryCoordinates:
        pos = ctx.device_to_user_distance(*pos)

    if center:
        ctx.move_to(pos[0] - width/2, pos[1])
    else:
        if rightEdge:
            #select monospaced font
            if width is None:
                (x, y, width, height, dx, dy) = ctx.text_extents(text)
            ctx.move_to(pos[0] - 1.5 * width, pos[1])
        else:
            ctx.move_to(*pos)

    ctx.set_source_rgba(*color)
    ctx.show_text(text)

    return width


#------------------------------------------------------------------------------------------------------------
def draw_obs_layer_A(ctx, factory,  highlight=None):
    draw_BG(ctx, factory.DRAWINGORIGIN, *factory.FACTORYDIMENSIONS, darkmode=False)
    drawFactory(ctx, factory, None, drawColors = False, drawNames=False, highlight=highlight, isObs=True, darkmode=False,)
    drawCollisions(ctx, factory.machineCollisionList, factory.wallCollisionList, outsiderList=factory.outsiderList)

    return ctx
#------------------------------------------------------------------------------------------------------------
def draw_obs_layer_B(ctx, factory,  highlight=None):
    draw_BG(ctx, factory.DRAWINGORIGIN, *factory.FACTORYDIMENSIONS, darkmode=False)
    draw_detail_paths(ctx, factory.fullPathGraph, factory.reducedPathGraph)
    drawFactory(ctx, factory, factory.dfMF, drawWalls=False, drawColors = False, drawNames=False, highlight=highlight, isObs=True)
    
    return ctx
