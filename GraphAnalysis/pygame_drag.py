import pygame
import cairo
import numpy as np

from shapely.geometry import MultiPoint, MultiPolygon, box
from shapely.affinity import rotate
from shapely.ops import  unary_union
 
 
SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080
BLOCK_SIZE = 50


WIDTH = 320
HEIGHT = 320
MAXSHAPEWIDTH = 40
MAXSHAPEHEIGHT = 40
AMOUNTRECT = 25
AMOUNTPOLY = 0
MAXCORNERS = 3

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


def update_fps():
	fps = str(int(clock.get_fps()))
	fps_text = font.render(fps, 1, pygame.Color("white"))
	return fps_text


def create_factory():
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
        
    return unary_union(MultiPolygon(polygons))



 

rng = np.random.default_rng()

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.SCALED)
#screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.SCALED | pygame.FULLSCREEN)

cairo_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, SCREEN_WIDTH, SCREEN_HEIGHT)
ctx = cairo.Context(cairo_surface)

font = pygame.font.SysFont("Arial", 18)

rects = []
 
for x in range(30):
    for y in range (2):
        rects.append( pygame.Rect(x*(BLOCK_SIZE+5), y*(BLOCK_SIZE+5), BLOCK_SIZE, BLOCK_SIZE))

multi = create_factory()

selected = None
   
# --- mainloop ---
 
clock = pygame.time.Clock()
is_running = True
is_darkmode = True
 
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
                        selected_offset_y = r.y - (event.pos[1])
               
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                selected = None
               
        elif event.type == pygame.MOUSEMOTION:
            if selected is not None: # selected can be `0` so `is not None` is required
                # move object
                rects[selected].x = event.pos[0] + selected_offset_x
                rects[selected].y = (event.pos[1]) + selected_offset_y
               
        # --- objects events ---
 
       
    # --- draws ---

    # draw rect
    
    draw_BG(ctx)

    for r in rects:
        draw_rect(ctx, r, rng.random(size=3))
    
    #for poly in multi:
        #draw_poly(ctx, poly, rng.random(size=3))

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