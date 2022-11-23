import cairo
import networkx as nx
import numpy as np
from shapely.ops import polylabel



def draw_BG(ctx, width, height, darkmode=True):
    ctx.rectangle(0,0, *ctx.device_to_user_distance(width,height))
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
    pos=nx.get_node_attributes(fullPathGraph,'pos')

 
    modifer = 0.5 if asStreets else 1.0

    for u,v,data in reducedPathGraph.edges(data=True):
        temp = [pos[x] for x in data['nodelist']]
        ctx.set_line_width(data['pathwidth']*modifer)
        ctx.move_to(*temp[0])
        for node in temp[1:]:
            ctx.line_to(*node)
        ctx.stroke()
    ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)

    for u,v,data in reducedPathGraph.edges(data=True):
        temp = [pos[x] for x in data['nodelist']]
        if data['pathtype'] =="twoway":
            ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
            ctx.move_to(*temp[0])
            for node in temp[1:]:
                ctx.line_to(*node)
    ctx.set_dash(list(ctx.device_to_user_distance(10, 10)))
    ctx.stroke()
    ctx.set_dash([])

    return ctx
#------------------------------------------------------------------------------------------------------------
def draw_simple_paths(ctx, fullPathGraph, reducedPathGraph):
    ctx.set_source_rgba(0.0, 0.0, 0.5, 0.5)
    ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    pos=nx.get_node_attributes(fullPathGraph,'pos')

    for u,v in reducedPathGraph.edges():
        ctx.set_line_width(ctx.device_to_user_distance(10, 10)[0])
        ctx.move_to(*pos[u])
        ctx.line_to(*pos[v])
    ctx.stroke()
    
    #crossroads = list(nx.get_node_attributes(G, "isCrossroads").keys())

    endpoints=[node for node, degree in fullPathGraph.degree() if degree == 1]
    crossroads= [node for node, degree in fullPathGraph.degree() if degree >= 3]

    ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
    for point in crossroads:
        ctx.move_to(*pos[point])
        ctx.arc(*pos[point], ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
    ctx.fill()

    ctx.set_source_rgba(0.0, 1.0, 0.0, 1.0)
    for point in endpoints:
        ctx.move_to(*pos[point])
        ctx.arc(*pos[point], ctx.device_to_user_distance(10, 10)[0], 0, 2*np.pi)
    ctx.fill()
    return ctx
#------------------------------------------------------------------------------------------------------------
def draw_poly(ctx, poly, color, text:str=None, highlight=False, drawHoles=True):
    
    for subpoly in poly.geoms:
        if highlight:
            ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
        else:
            ctx.set_source_rgba(*color, 1.0)
        ctx.move_to(*subpoly.exterior.coords[0])
        for x,y in subpoly.exterior.coords[1:]:
            ctx.line_to(x,y)
        ctx.close_path()
        ctx.fill()

        if drawHoles:
            for loop in subpoly.interiors:
                ctx.move_to(*loop.coords[0])
                for x,y in loop.coords[1:]:
                    ctx.line_to(x,y)
            ctx.set_source_rgba(*color, 1.0)
            ctx.close_path()
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
def draw_pathwidth_circles(ctx, fullPathGraph):
    for _ , data in fullPathGraph.nodes(data=True):
        ctx.move_to(data['pos'][0] + data['pathwidth']/2, data['pos'][1])
        ctx.arc(*data['pos'], data['pathwidth']/2, 0, 2*np.pi)
    ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
    ctx.set_source_rgba(0.0, 0.0, 0.8, 0.8)
    ctx.stroke()
    return ctx

def draw_pathwidth_circles2(ctx, fullPathGraph, reducedPathGraph):
    pathwidth = (nx.get_node_attributes(fullPathGraph,'pathwidth'))
    for node , data in reducedPathGraph.nodes(data=True):
        ctx.move_to(data['pos'][0] + pathwidth[node]/2, data['pos'][1])
        ctx.arc(*data['pos'], pathwidth[node]/2, 0, 2*np.pi)
    ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
    ctx.set_source_rgba(0.0, 0.0, 0.8, 0.8)
    ctx.stroke()
    return ctx
#------------------------------------------------------------------------------------------------------------
def draw_route_lines(ctx, route_lines):
    for line in route_lines:
        ctx.move_to(line.xy[0][0], line.xy[1][0])
        ctx.line_to(line.xy[0][1], line.xy[1][1])
    ctx.set_line_width(ctx.device_to_user_distance(3, 3)[0])
    ctx.set_source_rgba(0.5, 0.5, 0.5, 1.0)
    ctx.stroke()
    return ctx

#------------------------------------------------------------------------------------------------------------
def drawFactory(ctx, machine_dict=None, wall_dict=None, materialflow_file=None, drawColors = True, drawNames = True, wallInteriorColor = (0, 0, 0), drawMachineCenter = False, drawOrigin = False, highlight = None):   

    #Walls
    if wall_dict:
        ctx.set_fill_rule(cairo.FillRule.EVEN_ODD)
        for wall in wall_dict.values():
            #draw all walls
            for  poly in wall.poly.geoms:
                ctx.set_source_rgb((0.2), 0.2, 0.2)
                ctx.move_to(*poly.exterior.coords[0])
                for point in poly.exterior.coords[1:]:  
                    ctx.line_to(point[0], point[1])
                ctx.close_path()
                ctx.fill()
            #draw all holes
                ctx.set_source_rgb(*wallInteriorColor)
                for loop in poly.interiors:
                    ctx.move_to(*loop.coords[0])
                    for point in loop.coords[1:]:
                        ctx.line_to(point[0], point[1])
                    ctx.close_path()
                    ctx.fill()
                        
    #draw machine positions
    if machine_dict:
        ctx.set_fill_rule(cairo.FillRule.WINDING)
        ctx.set_line_width(ctx.device_to_user_distance(3, 3)[0])
        for index, machine in enumerate(machine_dict.values()):
            for poly in machine.poly.geoms:
                ctx.move_to(*poly.exterior.coords[0])
                for point in poly.exterior.coords[1:]: 
                    ctx.line_to(point[0], point[1])
                ctx.close_path()
                if drawColors:
                    #highlighted machine
                    if(index == highlight or machine.gid == highlight):
                        ctx.set_source_rgb(1.0, 0.0, 0.0)
                    #other machines
                    else:
                        ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                    #other machines
                else:
                    #highlighted machine
                    if(index == highlight or machine.gid == highlight):
                        ctx.set_source_rgb(0.9, 0.9, 0.9)
                    #other machines
                    else:
                        ctx.set_source_rgb(0.4, 0.4, 0.4)                         
                ctx.fill()

                if drawNames:
                    ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                    ctx.set_font_size(ctx.device_to_user_distance(14, 14)[0])
                    (x, y, width, height, dx, dy) = ctx.text_extents(str(machine.gid))
                    point = polylabel(poly.convex_hull, tolerance=1000)
                    ctx.move_to(point.x - width/2, point.y - height/2)    
                    ctx.show_text(str(machine.gid))

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

    #Material Flow
    drawMaterialFlow(ctx, machine_dict, materialflow_file, drawColors)
    

    return ctx

def drawMaterialFlow(ctx, machine_dict,  materialflow_file=None, drawColors = True):
    if  materialflow_file is not None:

        for index, row in materialflow_file.iterrows():
            current_from_Machine = machine_dict[row['from']]
            current_to_Machine = machine_dict[row['to']]
            try:
                if(drawColors):
                    ctx.set_source_rgba(*current_from_Machine.color, 0.7)
                else:
                    ctx.set_source_rgba(0.6, 0.6, 0.6)

                ctx.move_to(current_from_Machine.center.x, current_from_Machine.center.y)
                ctx.line_to(current_to_Machine.center.x, current_to_Machine.center.y)
                ctx.set_line_width(row["intensity_sum_norm"] * ctx.device_to_user_distance(20, 30)[0])
                ctx.stroke()   
            except KeyError:
                print(f"Error in Material Flow Drawing - Machine {row[0]} or {row[1]} not defined")
                continue
    

    return ctx

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
def drawCollisions(ctx, machineCollisionList, wallCollisionList=None, outsiderList=None, drawColors = True):
    #Drawing collisions between machines
    if(drawColors):
        ctx.set_source_rgb(1.0, 0.3, 0.0)
    else:
        ctx.set_source_rgb(0.7, 0.7, 0.7)

    for collision in machineCollisionList:
        for poly in collision.geoms:   
            for point in poly.exterior.coords:
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
                for point in poly.exterior.coords:
                    ctx.line_to(point[0], point[1])
            ctx.close_path()
            ctx.fill()     
        #Drawing collisions between machines and walls
    if(outsiderList):
        if(drawColors):
            ctx.set_source_rgb(1.0, 0.3, 0.0)
        else:
            ctx.set_source_rgb(0.7, 0.7, 0.7)
        for outsider in outsiderList:
            for poly in outsider.geoms:   
                for point in poly.exterior.coords:
                    ctx.line_to(point[0], point[1])
            ctx.close_path()
            ctx.fill()       


    return ctx

#------------------------------------------------------------------------------------------------------------
def draw_text_topleft(ctx, text, color):
    ctx.move_to(*ctx.device_to_user_distance(20, 20))
    ctx.set_font_size(ctx.device_to_user_distance(12, 12)[0])
    ctx.set_source_rgba(*color)
    ctx.show_text(text)

def draw_text_topleft2(ctx, text, color):
    ctx.move_to(*ctx.device_to_user_distance(20, 40))
    ctx.set_font_size(ctx.device_to_user_distance(12, 12)[0])
    ctx.set_source_rgba(*color)
    ctx.show_text(text)

def draw_text_pos(ctx, text, color, pos):
    ctx.set_font_size(ctx.device_to_user_distance(12, 12)[0])
    (x, y, width, height, dx, dy) = ctx.text_extents(text)
    mousepos = ctx.device_to_user_distance(*pos)
    ctx.move_to(mousepos[0] - width/2, mousepos[1] + ctx.device_to_user_distance(30, 30)[0])
    
    ctx.set_source_rgba(*color)
    ctx.show_text(text)



