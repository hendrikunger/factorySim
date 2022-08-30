from array import array
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import queue
import json

import cairo
import moderngl
import moderngl_window as mglw
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.affinity import translate
from paho.mqtt import client as mqtt

from routing import FactoryPath
from rendering import draw_simple_paths, draw_detail_paths, draw_poly, draw_pathwidth_circles, draw_route_lines
from creation import FactoryCreator
import baseConfigs 


class DrawingModes(Enum):
    NONE = None
    RECTANGLE = 114 # R Key
    POLYGON = 112 # P Key

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 


class Modes (Enum):
    MODE1 = 49 # 1 Key
    MODE2 = 50 # 2 Key
    MODE3 = 51 # 3 Key
    MODE4 = 52 # 4 Key
    MODE5 = 53 # 5 Key
    MODE6 = 54 # 6 Key
    MODE7 = 55 # 7 Key
    MODE8 = 56 # 8 Key
    MODE9 = 57 # 9 Key
    MODE0 = 48 # 0 Key
    DRAWING = DrawingModes.NONE

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 



class factorySimLive(mglw.WindowConfig):
    title = "factorySimLive"
    lastTime = 1
    fps_counter = 30
    #window_size = (3840, 2160)
    #window_size = (1920, 1080)
    window_size = (1280, 720)
    #window_size = (1920*6, 1080)
    aspect_ratio = None
    fullscreen = False
    resizable = True
    selected = None
    currentScale = 1.0
    is_darkmode = True
    is_dirty = False
    is_calculating = False
    update_during_calculation = False
    clickedPoints = []
    factoryPath = None
    factoryConfig = baseConfigs.SMALL
    mqtt_Q = None # Holds mqtt messages till they are processed
      

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.multi, self.bb = FactoryCreator(*self.factoryConfig.creationParameters()).create_factory()
        self.currentScale = self.scale_factory()
        self.factoryPath=FactoryPath(*self.factoryConfig.pathParameters())
        #self.factoryPath.TIMING = True
        self.future = self.executor.submit(self.factoryPath.calculateAll, self.multi, self.bb)
        
        self.colors = [self.rng.random(size=3) for _ in self.multi.geoms]
        self.G, self.I = self.future.result()
        self.setupKeys()
        #MQTT Connection
        self.mqtt_Q = queue.Queue(maxsize=100)
        self.mqtt_client = mqtt.Client(client_id="factorySimLive")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect("broker.hivemq.com", 1883)
        self.mqtt_client.loop_start()
        
        

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
        self.mqtt_client.loop_stop()
        

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
            # End Drawing Mode
            if key == keys.ESCAPE:
                self.clickedPoints.clear()
                self.activeModes[Modes.DRAWING] = DrawingModes.NONE
                self.wnd.exit_key = keys.ESCAPE

            if(Modes.has_value(key)):
                self.activeModes[Modes(key)] = not self.activeModes[Modes(key)]

            if(DrawingModes.has_value(key)):
                self.activeModes[Modes.DRAWING] = DrawingModes(key)
                self.clickedPoints.clear()
                self.wnd.exit_key = None


    def mouse_position_event(self, x, y, dx, dy):
        self.cursorPosition = (x, y)

    def mouse_drag_event(self, x, y, dx, dy):
        if self.selected is not None and self.activeModes[Modes.DRAWING] == DrawingModes.NONE: # selected can be `0` so `is not None` is required
            # move object  
            temp = list(self.multi.geoms)
            #Get coordinates of objects top left corner
            temp_x = temp[self.selected].bounds[0]
            temp_y = temp[self.selected].bounds[1] # max y for shapely, min y for pygame coordinates
            temp[self.selected] = translate(self.multi.geoms[self.selected], 
                xoff=((x / self.currentScale)- temp_x) + self.selected_offset_x,
                yoff=((y / self.currentScale) - temp_y) + self.selected_offset_y
                )            
            self.multi = MultiPolygon(temp)
            
            self.update_needed()


    def mouse_press_event(self, x, y, button):
        #Highlight and prepare for Drag
        if button == 1:  
            #Draw Rectangle         
            if self.activeModes[Modes.DRAWING] == DrawingModes.RECTANGLE:
                self.clickedPoints.append((x,y))
                if len(self.clickedPoints) >= 2:
                    self.factory_add_rect(self.clickedPoints[0], self.clickedPoints[1], useWindowCoordinates=True)
                    self.clickedPoints.clear()
                    #self.activeModes[Modes.DRAWING] = DrawingModes.NONE
                    self.update_needed()

             #Draw Polygon         
            elif self.activeModes[Modes.DRAWING] == DrawingModes.POLYGON:
                self.clickedPoints.append((x,y))

            #Prepare Mouse Drag
            else:
                for i, poly in enumerate(self.multi.geoms):
                    point_scaled = Point(x/self.currentScale, y/self.currentScale)
                    if poly.contains(point_scaled):
                        self.selected = i
                        self.selected_offset_x = poly.bounds[0] - point_scaled.x
                        self.selected_offset_y = poly.bounds[1] - point_scaled.y


        if button == 2:
            #Finish Polygon
            if self.activeModes[Modes.DRAWING] == DrawingModes.POLYGON:
                if len(self.clickedPoints) >= 3:
                    self.factory_add_poly(self.clickedPoints)
                    self.clickedPoints.clear()
                    #self.activeModes[Modes.DRAWING] = DrawingModes.NONE
                    self.update_needed()

        #Shift Click to delete Objects
        if button == 1 and self.wnd.modifiers.shift and self.selected is not None and len(self.multi.geoms) > 1:
            temp = list(self.multi.geoms)
            temp.pop(self.selected)
            self.colors.pop(self.selected)
            self.multi = MultiPolygon(temp)
            
            self.colors
            self.selected = None
            self.update_needed()



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
        self.process_mqtt()


    def render_cairo_to_texture(self):
        # Draw with cairo to surface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.window_size[0], self.window_size[1])
        cctx = cairo.Context(surface)
        
        cctx.scale(self.currentScale, self.currentScale)

        self.draw_BG(cctx)
        

        if self.is_dirty:
            if self.is_calculating:
                if self.future.done():
                    self.future.result()
                    self.G = self.factoryPath.fullPathGraph
                    self.I = self.factoryPath.PathGraph
                    self.is_dirty = False
                    self.is_calculating = False
                    #if we had changes during last calulation, recalulate
                    if self.update_during_calculation:
                        self.update_during_calculation = False
                        self.is_dirty = True
                        self.is_calculating = True
                        self.future = self.executor.submit(self.factoryPath.calculateAll, self.multi, self.bb)
            else:
                self.future = self.executor.submit(self.factoryPath.calculateAll, self.multi, self.bb)
                self.is_calculating = True

        if self.activeModes[Modes.MODE1]:
            cctx = draw_detail_paths(cctx, self.G, self.I)

        if self.activeModes[Modes.MODE2]:
            cctx = draw_simple_paths(cctx, self.G, self.I)
        
        if self.activeModes[Modes.MODE3]:
            cctx = draw_route_lines(cctx, self.factoryPath.route_lines)

        if self.activeModes[Modes.MODE4]:
            cctx = draw_pathwidth_circles(cctx, self.G)

        for i, (poly, color) in enumerate(zip(self.multi.geoms, self.colors)):
            cctx = draw_poly(cctx, poly, color, highlight= True if i == self.selected else False)

        if self.activeModes[Modes.DRAWING] == DrawingModes.RECTANGLE and len(self.clickedPoints) > 0:
            self.draw_live_rect(cctx, self.clickedPoints[0], self.cursorPosition)
        if self.activeModes[Modes.DRAWING] == DrawingModes.POLYGON and len(self.clickedPoints) > 0:
            self.draw_live_poly(cctx, self.clickedPoints, self.cursorPosition)


        self.draw_fps(cctx, self.fps_counter)

        # Copy surface to texture
        texture = self.ctx.texture((self.window_size[0], self.window_size[1]), 4, data=surface.get_data())
        texture.swizzle = 'BGRA' # use Cairo channel order (alternatively, the shader could do the swizzle)
        surface.finish()
        del(cctx)
        del(surface)
        return texture



#--------------------------------------------------------------------------------------------------------------------------------
    def update_needed(self):            
        self.is_dirty = True
        if self.is_calculating:
            self.update_during_calculation = True

    def setupKeys(self):
        keys = self.wnd.keys

        self.activeModes = {Modes.MODE0 : False,
                        Modes.MODE1 : True,
                        Modes.MODE2 : False,
                        Modes.MODE3 : False,
                        Modes.MODE4 : False,
                        Modes.MODE5 : False,
                        Modes.MODE6 : False,
                        Modes.MODE7 : False,
                        Modes.MODE8 : False,
                        Modes.MODE9 : False,
                        Modes.DRAWING : DrawingModes.NONE
        }


    def draw_fps(self, ctx, fps):

        ctx.move_to(*ctx.device_to_user_distance(20, 20))
        ctx.set_font_size(ctx.device_to_user_distance(12, 12)[0])
        if self.is_darkmode:
            ctx.set_source_rgb(1.0, 1.0, 1.0)
        else:
            ctx.set_source_rgba(0.0, 0.0, 0.0)
        if self.activeModes[Modes.DRAWING].value:
            ctx.show_text(f"{fps:.0f}    {self.activeModes[Modes.DRAWING].name}")
        else:
            ctx.show_text(f"{fps:.0f}")


    def draw_BG(self, ctx):
        ctx.rectangle(0, 0, self.window_size[0], self.window_size[1])
        #Color is actually BGR in Pygame
        if self.is_darkmode:
            ctx.set_source_rgba(0.0, 0.0, 0.0)
        else:
            ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.fill()


    def draw_live_rect(self, ctx, topleft, bottomright):
        ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(1.0, 0.0, 0.0, 0.3)
        ctx.set_line_width(ctx.device_to_user_distance(2, 2)[0])
        ctx.set_dash(list(ctx.device_to_user_distance(5, 5)))

        ctx.rectangle(*ctx.device_to_user_distance(*topleft), *ctx.device_to_user_distance(*np.subtract(bottomright, topleft)))
        ctx.fill_preserve()
        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
        ctx.stroke()

    def draw_live_poly(self, ctx, points, cursorPosition):

        ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(1.0, 0.0, 0.0, 0.3)
        ctx.set_line_width(ctx.device_to_user_distance(2, 2)[0])
        ctx.set_dash(list(ctx.device_to_user_distance(5, 5)))

        ctx.move_to(*ctx.device_to_user_distance(*points[0]))
        if len(points) > 1:
            for x,y in points[1:]:
                ctx.line_to(*ctx.device_to_user_distance(x,y))
        ctx.line_to(*ctx.device_to_user_distance(*cursorPosition))
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)
        ctx.stroke()


    def factory_add_rect(self, topleft, bottomright, useWindowCoordinates = False):
        if useWindowCoordinates:
            newRect = box(topleft[0]/self.currentScale, topleft[1]/self.currentScale, bottomright[0]/self.currentScale, bottomright[1]/self.currentScale)
        else:
            newRect = box(topleft[0], topleft[1], bottomright[0], bottomright[1])

        temp = list(self.multi.geoms)
        temp.append(newRect)
        self.colors.append(self.rng.random(size=3))
        self.multi = MultiPolygon(temp)
        

    def factory_add_poly(self, points):
        scaledPoints = np.array(points)/self.currentScale
        newPoly = Polygon(scaledPoints)
        if newPoly.is_valid:
            temp = list(self.multi.geoms)
            temp.append(newPoly)
            self.colors.append(self.rng.random(size=3))
            self.multi = MultiPolygon(temp)
            
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



# MQTT Stuff ----------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        client.subscribe("EDF/#")

    def on_disconnect(self, client, userdata, rc):
        print("Disconnected with result code "+str(rc))

    def on_message(self, client, userdata, msg):
        try:
            self.mqtt_Q.put_nowait((msg.topic, msg.payload))
        except queue.Full:
            print("Dropping message because queue is full.")
            return

    def process_mqtt(self):
        if not self.mqtt_Q.empty():
            try:
                topic, payload = self.mqtt_Q.get_nowait()
            except queue.Empty:
                print("Tried reading from empty Queue.")
                return
            if topic == "EDF/bg":
                if payload == b"True":
                    self.is_darkmode = True
                elif payload == b"False":
                    self.is_darkmode = False
                else:
                    print("Unknown payload for EDF/bg: " + payload)

            if topic.startswith("EDF/machines/"):
                if topic.endswith("/pos"):
                    self.handleMQTT_Position(topic, payload)
                if topic.endswith("/geom"):
                    self.handleMQTT_Geometry(topic, payload)


    def extractID(self, topic):
        index = topic.split("/")
        #safeguard against misformed topics
        if len(index) >=2:
            try:
                index = int(index[2])
                return index
            except ValueError:
                print("Not an integer: ", index[2])
                return None


    def handleMQTT_Position(self, topic, payload):
        pp = json.loads(payload)
        index = self.extractID(topic)
        if index is not None and "x" in pp and "y" in pp:
            temp = list(self.multi.geoms)
            temp_x = temp[index].bounds[0]
            temp_y = temp[index].bounds[1] # max y for shapely, min y for pygame coordinates
            temp[index] = translate(self.multi.geoms[index], 
            xoff=(pp["x"] - temp_x),
            yoff=(pp["y"] - temp_y)
            )            
            self.multi = MultiPolygon(temp)
            self.update_needed()
        else:
            print("MQTT message malformed. Needs JSON Payload containing x and y coordinates and valid machine index")
    
    def handleMQTT_Geometry(self, topic, payload):
        pp = json.loads(payload)
        index = self.extractID(topic)
        if index is not None and "topleft_x" in pp and "topleft_y" in pp and "bottomright_x" in pp and "bottomright_y" in pp:
            self.factory_add_rect((pp["topleft_x"],pp["topleft_y"]),(pp["bottomright_x"],pp["bottomright_y"]))
            self.update_needed()
        else:
            print("MQTT message malformed. Needs JSON Payload containing topleft_x, topleft_y, bottomright_x, bottomright_y")



if __name__ == "__main__":
    factorySimLive.run()
