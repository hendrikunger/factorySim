import cairo
import networkx as nx
import numpy as np

def draw_detail_paths(ctx, G, I):
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

    return ctx

def draw_simple_paths(ctx, G, I):
    ctx.set_source_rgba(0.0, 0.0, 0.5, 0.5)
    ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    pos=nx.get_node_attributes(G,'pos')

    for u,v in I.edges():
        ctx.set_line_width(ctx.device_to_user_distance(10, 10)[0])
        ctx.move_to(*pos[u])
        ctx.line_to(*pos[v])
        ctx.stroke()
    
    #crossroads = list(nx.get_node_attributes(G, "isCrossroads").keys())

    endpoints=[node for node, degree in G.degree() if degree == 1]
    crossroads= [node for node, degree in G.degree() if degree >= 3]

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

def draw_poly(ctx, poly, color, highlight=False):
    if highlight:
        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
    else:
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
    return ctx

def draw_pathwidth_calculation(ctx, route_lines, G):

    for line in route_lines:
        ctx.move_to(line.xy[0][0], line.xy[1][0])
        ctx.line_to(line.xy[0][1], line.xy[1][1])
    ctx.set_line_width(ctx.device_to_user_distance(3, 3)[0])
    ctx.set_source_rgba(0.5, 0.5, 0.5, 1.0)
    ctx.stroke()

    for _ , data in G.nodes(data=True):
        ctx.move_to(data['pos'][0] + data['pathwidth'], data['pos'][1])
        ctx.arc(*data['pos'], data['pathwidth'], 0, 2*np.pi)
    ctx.set_line_width(ctx.device_to_user_distance(1, 1)[0])
    ctx.set_source_rgba(0.0, 0.0, 0.8, 0.8)
    ctx.stroke()
    return ctx