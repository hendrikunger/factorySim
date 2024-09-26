from array import array
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import queue
import json
import os
import yaml
import sys


import cairo
import moderngl
import moderngl_window as mglw
import numpy as np
from shapely.geometry import Point, Polygon, box, MultiPolygon
from paho.mqtt import client as mqtt

from factorySim.rendering import *
from factorySim.creation import FactoryCreator
import factorySim.baseConfigs as baseConfigs
from factorySim.factoryObject import FactoryObject
from factorySim.factorySimEnv import FactorySimEnv
from factorySim.utils import check_internet_conn

from ray.rllib.policy.policy import Policy




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
    MODE_N0 = 65456 # Num 0 Key
    MODE_N1 = 65457 # Num 1 Key 
    MODE_N2 = 65458 # Num 2 Key 
    MODE_N3 = 65459 # Num 3 Key 
    MODE_N4 = 65460 # Num 4 Key 
    MODE_N5 = 65461 # Num 5 Key 
    MODE_N6 = 65462 # Num 6 Key 
    MODE_N7 = 65463 # Num 7 Key 
    MODE_N8 = 65464 # Num 8 Key 
    MODE_N9 = 65465 # Num 9 Key 
    DRAWING = DrawingModes.NONE
    AGENTDEBUG = 100 # D Key


    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 


class factorySimLive(mglw.WindowConfig):
    title = "factorySimLive"
    lastTime = 1
    fps_counter = 30
    #window_size = (3840, 4320)
    #window_size = (3840, 2160)
    #window_size = (1920, 1080)
    window_size = (1280, 720)
    #window_size = (1920*6, 1080)
    mqtt_broker = "broker.emqx.io"
    #mqtt_broker = "10.54.129.47"
    aspect_ratio = None
    fullscreen = False
    resizable = True
    selected = None
    currentScale = 1.0
    is_darkmode = True
    is_EDF = False
    is_dirty = False
    is_calculating = False
    is_shrunk = False
    update_during_calculation = False
    clickedPoints = []
    #
    factoryConfig = baseConfigs.SMALLSQUARE
    #factoryConfig = baseConfigs.EDF_EMPTY
    #factoryConfig = baseConfigs.EDF
    mqtt_Q = None # Holds mqtt messages till they are processed
    cursorPosition = None
    currenDebugMode = 0
    dpiScaler = 2 if sys.platform == "darwin" else 1
    is_online = check_internet_conn()
    EVALUATION = False

      
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng()
        self.cmap = self.rng.random(size=(200, 3))
        self.executor = ThreadPoolExecutor(max_workers=1)


        
        self.factoryCreator = FactoryCreator(*self.factoryConfig.creationParameters())
        basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input")
        configpath = os.path.join(basePath, "..", "config.yaml")
    
        self.ifcPath = os.path.join(basePath, "2", "Simple.ifc")
        self.ifcPath = os.path.join(basePath, "2")
        #self.ifcPath = os.path.join(basePath, "2", "EDF.ifc")
        self.ifcPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation", "04.ifc")



        with open(configpath, 'r') as f:
            self.f_config = yaml.load(f, Loader=yaml.FullLoader)


        self.f_config['env_config']['inputfile'] = self.ifcPath
        print(str(self.factoryConfig.NAME))
        self.f_config['env_config']['factoryconfig'] = str(self.factoryConfig.NAME)
        self.f_config["env_config"]["reward_function"] = 1


        self.f_config['evaluation_config']["env_config"]["inputfile"] = self.ifcPath
        self.f_config['evaluation_config']["env_config"]["reward_function"] = 1
        

        #self.ifcPath=None

        self.create_factory()

        self.factoryCreator.bb = self.env.factory.creator.bb
        self.factoryCreator.factoryWidth = self.env.factory.creator.factoryWidth
        self.factoryCreator.factoryHeight = self.env.factory.creator.factoryHeight

        ifcPath = os.path.join(basePath, "FTS.ifc")

        self.mobile_dict =self.factoryCreator.load_ifc_factory(ifcPath, "IFCBUILDINGELEMENTPROXY", recalculate_bb=False)
        print(list(self.mobile_dict.values())[0].gid)
        self.nextGID = len(self.env.factory.machine_dict)
        self.set_factoryScale()


        self.setupKeys()


        #MQTT Connection
        self.mqtt_Q = queue.Queue(maxsize=100)
        if self.is_online:
            self.mqtt_client = mqtt.Client(client_id="factorySimLive")
            self.mqtt_client.on_connect = self.on_connect
            self.mqtt_client.on_disconnect = self.on_disconnect
            self.mqtt_client.on_message = self.on_message
            self.mqtt_client.connect(self.mqtt_broker, 1883)
            self.mqtt_client.loop_start()
        
        #Agent 
        if False:
            checkpointPath = os.path.join(basePath, "..", "artifacts", "checkpoint_PPO_latest")
            self.Agent = Policy.from_checkpoint(checkpointPath)["default_policy"]
        

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
        self.set_factoryScale()


    def close(self):
        print("closing")
        self.executor.shutdown()
        if self.is_online:
            self.mqtt_client.loop_stop()
        

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        # 65453  Num Minus
        # 65451  Num Plus
        # Key presses
        if action == keys.ACTION_PRESS:
            # Toggle Fullscreen
            if key == keys.F:
                self.wnd.fullscreen = not self.wnd.fullscreen   
            # Toggle Text Rendering under selection
            if key == keys.E:
                self.is_EDF = not self.is_EDF    
            # Agent Prediction
            if key == keys.A:
                self.agentInference()   
            # Debug Mode Rendering
            if key == 65451: # Num Plus
                self.currenDebugMode = self.currenDebugMode + 1 if self.currenDebugMode < 2 else 0
            if key == 65453: # Num Minus
                self.currenDebugMode = self.currenDebugMode - 1 if self.currenDebugMode > 0 else 2
            # Zoom
            if key == 43: # +
                self.currentScale += 0.005
                self.recreateCairoContext()
            if key == keys.MINUS:
                self.currentScale -= 0.005
                self.recreateCairoContext()
            # ShrunkMode
            if key == keys.M:
                if self.is_shrunk:
                    self.wnd.size = self.old_window_size
                    self.is_shrunk = False
                else:
                    self.old_window_size = self.window_size
                    self.is_shrunk = True
                    self.wnd.size = (84,84)
            # Save Factory
            if key == keys.S:
                self.env.factory.creator.save_ifc_factory(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"live.ifc"))
                self.env.factory.creator.saveMaterialFlow(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"live_mf.csv"), self.env.factory.dfMF)
            # Load Factory
            if key == keys.L:
                path_to_ifc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "live.ifc")
                if os.path.exists(path_to_ifc):
                    self.create_factory(path_to_ifc)
                    path_to_materialflow_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"live_mf.csv")
                    if os.path.exists(path_to_materialflow_file):
                        self.env.factory.dfMF = self.env.factory.creator.loadMaterialFlow(path_to_materialflow_file)
                    self.set_factoryScale()
                    self.nextGID = len(self.env.factory.machine_dict)
                    self.selected = None
                else:
                    print("No live.ifc found")
            #Save Positions
            if key == keys.END:
                self.env.factory.creator.save_position_json(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"live_pos.json"))
            # Load Positions
            if key == keys.HOME:
                self.env.factory.creator.load_position_json(os.path.join(os.path.dirname(os.path.realpath(__file__)), "live_pos.json"))
                self.set_factoryScale()
                self.nextGID = len(self.env.factory.machine_dict)
                self.selected = None
                self.is_dirty = True
            # Darkmode
            if key == keys.B:
                self.is_darkmode = not self.is_darkmode
            #Restart
            if key == keys.N:
                self.create_factory()
                self.set_factoryScale()
                self.nextGID = len(self.env.factory.machine_dict) 
                self.selected = None
            # End Drawing Mode
            if key == keys.ESCAPE:
                self.clickedPoints.clear()
                self.activeModes[Modes.DRAWING] = DrawingModes.NONE
                self.wnd.exit_key = keys.ESCAPE
            # Del to delete
            if key == keys.BACKSPACE and self.selected is not None and not self.is_calculating:
                self.F_delete_item(self.selected)
                self.selected = None

            if(Modes.has_value(key)):
                self.activeModes[Modes(key)] = not self.activeModes[Modes(key)]

            if(DrawingModes.has_value(key)):
                self.activeModes[Modes.DRAWING] = DrawingModes(key)
                self.clickedPoints.clear()
                self.wnd.exit_key = None


    def mouse_position_event(self, x, y, dx, dy):
        x *= self.dpiScaler
        y *= self.dpiScaler
        self.cursorPosition = (x  + self.env.factory.DRAWINGORIGIN[0] * self.currentScale, y + self.env.factory.DRAWINGORIGIN[1] * self.currentScale)

    def mouse_drag_event(self, x, y, dx, dy):
        x *= self.dpiScaler
        y *= self.dpiScaler
        self.cursorPosition = (x  + self.env.factory.DRAWINGORIGIN[0] * self.currentScale, y + self.env.factory.DRAWINGORIGIN[1] * self.currentScale)

        if self.wnd.mouse_states.left == True and self.selected is not None and self.activeModes[Modes.DRAWING] == DrawingModes.NONE: # selected can be `0` so `is not None` is required
            # move object  
            self.env.factory.machine_dict[self.selected].translate_Item(((x / self.currentScale) + self.env.factory.DRAWINGORIGIN[0]) + self.selected_offset_x,
                ((y / self.currentScale) + self.env.factory.DRAWINGORIGIN[1]) + self.selected_offset_y)
            self.update_needed()
            

    def mouse_press_event(self, x, y, button):
        #Highlight and prepare for Drag
        x *= self.dpiScaler
        y *= self.dpiScaler
        if button == 1:  
            #Draw Rectangle         
            if self.activeModes[Modes.DRAWING] == DrawingModes.RECTANGLE:
                self.clickedPoints.append((x + self.env.factory.DRAWINGORIGIN[0] * self.currentScale ,y + self.env.factory.DRAWINGORIGIN[1] * self.currentScale))
                if len(self.clickedPoints) >= 2:
                    self.F_add_rect(self.clickedPoints[0], self.clickedPoints[1], useWindowCoordinates=True)
                    self.clickedPoints.clear()                   

             #Draw Polygon         
            elif self.activeModes[Modes.DRAWING] == DrawingModes.POLYGON:
                self.clickedPoints.append((x + self.env.factory.DRAWINGORIGIN[0] * self.currentScale, y + self.env.factory.DRAWINGORIGIN[1] * self.currentScale))

            #Prepare Mouse Drag
            else:
                for key, machine in reversed(self.env.factory.machine_dict.items()):
                    point_scaled = Point(x/self.currentScale + self.env.factory.DRAWINGORIGIN[0], y/self.currentScale + self.env.factory.DRAWINGORIGIN[1])
                    if machine.poly.contains(point_scaled):
                        self.selected = key
                        self.selected_offset_x = machine.poly.bounds[0] - point_scaled.x
                        self.selected_offset_y = machine.poly.bounds[1] - point_scaled.y
                        break
                    else:
                        self.selected = None


        if button == 2:
            #Finish Polygon
            if self.activeModes[Modes.DRAWING] == DrawingModes.POLYGON:
                if len(self.clickedPoints) >= 3:
                    self.F_add_poly(self.clickedPoints, useWindowCoordinates = True)
                    self.clickedPoints.clear()

            #Add Materialflow
            elif self.selected is not None:
                for key, machine in reversed(self.env.factory.machine_dict.items()):
                    point_scaled = Point(x/self.currentScale + self.env.factory.DRAWINGORIGIN[0], y/self.currentScale + self.env.factory.DRAWINGORIGIN[1])
                    if machine.poly.contains(point_scaled) and key is not self.selected:
                        self.env.factory.addMaterialFlow(self.selected, key, np.random.randint(1,100))
                        self.update_needed()
                        break

        #Shift Click to delete Objects
        if button == 1 and self.wnd.modifiers.shift and self.selected is not None :
            self.F_delete_item(self.selected)
            self.selected = None


                


    def mouse_release_event(self, x: int, y: int, button: int):
        if button == 1:
            #self.selected = None
            pass

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
        draw_BG(self.cctx, self.env.factory.DRAWINGORIGIN,*self.env.factory.FACTORYDIMENSIONS, self.is_darkmode)
        
        if self.is_dirty:
            if self.is_calculating:
                if self.future.done():
                    _, _ , self.rating, _ = self.future.result()
                    self.is_dirty = False
                    self.is_calculating = False
                    #if we had changes during last calulation, recalulate
                    if self.update_during_calculation:
                        self.update_during_calculation = False
                        self.is_dirty = True
                        self.is_calculating = True
                        self.future = self.executor.submit(self.env.factory.evaluate)
            else:
                self.future = self.executor.submit(self.env.factory.evaluate)
                self.is_calculating = True
        color = (0.0, 0.0, 0.0) if self.is_darkmode else (1.0, 1.0, 1.0)
        
        if self.activeModes[Modes.AGENTDEBUG]:
            match self.currenDebugMode:
                case 0:
                    draw_obs_layer_A(self.cctx, self.env.factory, highlight=self.selected)
                case 1:
                    draw_obs_layer_B(self.cctx, self.env.factory, highlight=self.selected)
                case 2:
                    draw_text(self.cctx,(f"Easteregg"), (0.7, 0.0, 0.0, 1.0), (self.window_size[0]/2,self.window_size[1]/2), factoryCoordinates=False)
        else:
            drawFactory(self.cctx, self.env.factory, drawColors=True, highlight=self.selected, drawNames=True, darkmode=self.is_darkmode, drawWalls=True, drawOrigin=True)
            if self.activeModes[Modes.MODE9]: 
                draw_poly(self.cctx, self.env.factory.walkableArea, (0.9, 0.0, 0.0, 0.5), drawHoles=True)
            if self.activeModes[Modes.MODE7]: 
                draw_poly(self.cctx, self.env.factory.freeSpacePolygon, (0.0, 0.0, 1.0, 0.5), drawHoles=True)
                draw_poly(self.cctx, self.env.factory.growingSpacePolygon, (1.0, 1.0, 0.0, 0.5), drawHoles=True)
            if self.activeModes[Modes.MODE_N0]: draw_poly(self.cctx,  self.env.factory.freespaceAlongRoutesPolygon, (0.0, 0.6, 0.0, 0.5))
            if self.activeModes[Modes.MODE_N9]: draw_poly(self.cctx, self.env.factory.extendedPathPolygon, (0.0, 0.3, 0.0, 1.0))
            if self.activeModes[Modes.MODE3]: draw_poly(self.cctx, self.env.factory.pathPolygon, (0.0, 0.3, 0.0, 1.0))
            if self.activeModes[Modes.MODE1]: draw_detail_paths(self.cctx, self.env.factory.fullPathGraph, self.env.factory.reducedPathGraph, asStreets=True)
            if self.activeModes[Modes.MODE2]: draw_simple_paths(self.cctx, self.env.factory.fullPathGraph, self.env.factory.reducedPathGraph)
            if self.activeModes[Modes.MODE_N8]: draw_route_lines(self.cctx, self.env.factory.factoryPath.route_lines)
            drawFactory(self.cctx, self.env.factory, drawColors=True, highlight=self.selected, drawNames=True, drawWalls=False)
            if self.activeModes[Modes.MODE8]:   
                for key, poly in self.env.factory.usedSpacePolygonDict.items():
                    draw_poly(self.cctx, poly, (*self.cmap[key], 0.3))
            if self.activeModes[Modes.MODE_N7]: draw_pathwidth_circles(self.cctx, self.env.factory.fullPathGraph)
            if self.activeModes[Modes.MODE0]:draw_node_angles(self.cctx, self.env.factory.fullPathGraph, self.env.factory.reducedPathGraph)
            if self.activeModes[Modes.MODE5]: 
                drawMaterialFlow(self.cctx, self.env.factory.machine_dict, self.env.factory.dfMF, drawColors=True)
                draw_points(self.cctx, self.env.factory.MFIntersectionPoints, (1.0, 1.0, 0.0, 1.0))
            if self.activeModes[Modes.MODE4]: 
                drawCollisions(self.cctx, self.env.factory.machineCollisionList, wallCollisionList=self.env.factory.wallCollisionList, outsiderList=self.env.factory.outsiderList)
            if self.activeModes[Modes.MODE6]: 
                drawRoutedMaterialFlow(self.cctx, self.env.factory.machine_dict, self.env.factory.fullPathGraph, self.env.factory.reducedPathGraph, materialflow_file=self.env.factory.dfMF, selected=None)

            if self.is_EDF:
                for key, mobile in self.mobile_dict.items():
                    draw_poly(self.cctx, mobile.poly, mobile.color, text=str(mobile.name), drawHoles=True)
       

        if self.activeModes[Modes.DRAWING] == DrawingModes.RECTANGLE and len(self.clickedPoints) > 0:
            self.draw_live_rect(self.cctx, self.clickedPoints[0], self.cursorPosition)
        if self.activeModes[Modes.DRAWING] == DrawingModes.POLYGON and len(self.clickedPoints) > 0:
            self.draw_live_poly(self.cctx, self.clickedPoints, self.cursorPosition)

        color = (1.0, 1.0, 1.0) if self.is_darkmode else (0.0, 0.0, 0.0)
        mode = self.activeModes[Modes.DRAWING].name if self.activeModes[Modes.DRAWING].value else ""
        draw_text(self.cctx,(f"{self.fps_counter:.0f}   {mode}"), color, (20, 200))

        #Draw every rating on a new line
        textwidth = None
        for i, text in enumerate(self.env.factory.generateRatingText(multiline=True).split("\n")):
            pos = (self.window_size[0], (40 + i*20))
            textwidth = draw_text(self.cctx, text, color, pos, rightEdge=True, factoryCoordinates=False, input_width=textwidth)

        if self.selected != None and self.is_EDF: 
            #Calculate the position of bottom center of selected objects bounding box
            bbox = self.env.factory.machine_dict[self.selected].poly.bounds
            x = (bbox[0] + bbox[2])/2
            y = bbox[3]

            for i, text in enumerate(self.env.factory.generateRatingText(multiline=True).split("\n")):
                textwidth = draw_text(self.cctx, text, color, (x, y+ 200+ i*200), center=True, input_width=textwidth)

        
        # Copy surface to texture
        texture = self.ctx.texture((self.window_size[0], self.window_size[1]), 4, data=self.surface.get_data())
        texture.swizzle = 'BGRA' # use Cairo channel order (alternatively, the shader could do the swizzle)

        return texture



#--------------------------------------------------------------------------------------------------------------------------------
    def update_needed(self):
        self.is_dirty = True
        if self.is_calculating:
            self.update_during_calculation = True 

    def recreateCairoContext(self):
        self.surface, self.cctx = self.env.factory.provideCairoDrawingData(self.window_size[0], self.window_size[1], scale=self.currentScale)

    def setupKeys(self):
 
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
                        Modes.MODE_N0 : False,
                        Modes.MODE_N1 : True,
                        Modes.MODE_N2 : False,
                        Modes.MODE_N3 : False,
                        Modes.MODE_N4 : False,
                        Modes.MODE_N5 : False,
                        Modes.MODE_N6 : False,
                        Modes.MODE_N7 : False,
                        Modes.MODE_N8 : False,
                        Modes.MODE_N9 : False,
                        Modes.DRAWING : DrawingModes.NONE,
                        Modes.AGENTDEBUG : False,
        }


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


    def F_add_rect(self, topleft, bottomright, gid = None, useWindowCoordinates = False):
        if useWindowCoordinates:
            newRect = box(topleft[0]/self.currentScale, topleft[1]/self.currentScale, bottomright[0]/self.currentScale, bottomright[1]/self.currentScale)
        else:
            newRect = box(topleft[0], topleft[1], bottomright[0], bottomright[1])
        if gid:
            gid_to_use = gid
        else:
            gid_to_use = self.nextGID
            self.nextGID += 1
        bbox = newRect.bounds
        origin=(bbox[0],bbox[1])
        self.env.factory.machine_dict[gid_to_use] = FactoryObject(gid=str(gid_to_use), 
                                            name="M_" + str(gid_to_use),
                                            origin=origin,
                                            poly=MultiPolygon([newRect]))

        self.update_needed()

    def F_add_poly(self, points, gid = None, useWindowCoordinates = False):

        if useWindowCoordinates:
            scaledPoints = np.array(points)/self.currentScale
            newPoly = Polygon(scaledPoints)
        else:
            newPoly = Polygon(points)

        if newPoly.is_valid:
            if gid:
                gid_to_use = gid
            else:
                gid_to_use = self.nextGID
                self.nextGID += 1
            bbox = newPoly.bounds
            origin=(bbox[0],bbox[1])
            self.env.factory.machine_dict[gid_to_use] = FactoryObject(gid=str(gid_to_use), 
                                                name="M_" + str(gid_to_use),
                                                origin=origin,
                                                poly=MultiPolygon([newPoly]))
            self.update_needed()

    def F_delete_item(self, index):
            self.env.factory.machine_dict.pop(index)
            indexNames = self.env.factory.dfMF[ (self.env.factory.dfMF['source'] == index) | (self.env.factory.dfMF['target'] == index) ].index
            self.env.factory.dfMF = self.env.factory.dfMF.drop(indexNames).reset_index(drop=True)
            self.update_needed()
            
    def create_factory(self, ifcPath=None):
        if self.EVALUATION:
            env_config = self.f_config['evaluation_config']["env_config"].copy()
        else:
            env_config = self.f_config['env_config'].copy()
        if ifcPath:
            env_config['inputfile'] = ifcPath
            env_config["createMachines"] = False
            env_config["randomSeed"] = 42
            env_config["maxMF_Elements"] = None
        self.env = FactorySimEnv( env_config = env_config)
        self.env.reset()


        self.future = self.executor.submit(self.env.factory.evaluate)
        _, _ , self.rating, _ = self.future.result()

    def set_factoryScale(self):
        self.currentScale = self.env.factory.creator.suggest_factory_view_scale(self.window_size[0],self.window_size[1])
        self.recreateCairoContext()


    def agentInference(self):
        if self.selected:
            obs = self.env._get_obs(highlight=self.selected)
            action = self.Agent.compute_single_action(obs)[0]
            #action = self.env.action_space.sample()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            self.env.factory.update(self.selected, action[0], action[1], action[2], 0)
            self.update_needed()
            





# MQTT Stuff ----------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        client.subscribe("EDF/BP/#")

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
            if topic == "EDF/BP/bg":
                if payload == b"True":
                    self.is_darkmode = True
                elif payload == b"False":
                    self.is_darkmode = False
                else:
                    print("Unknown payload for EDF/BP/bg: " + payload)
            if topic.startswith("EDF/BP/machines/"):
                if topic.endswith("/pos"):
                    self.handleMQTT_Position(topic, payload)
                if topic.endswith("/geom"):
                    self.handleMQTT_Geometry(topic, payload)


    def extractID(self, topic):
        index = topic.split("/")
        #safeguard against misformed topics
        
        if len(index) >=3:
            if index[3].isnumeric():
                return int(index[3].strip())
            else:
                return index[3].strip()
   

    def handleMQTT_Position(self, topic, payload):
        pp = json.loads(payload)
        index = str(self.extractID(topic))

        if index in self.env.factory.machine_dict and "x" in pp and "y" in pp:        
            #Calculate distance of change
            machine = self.env.factory.machine_dict[index]
            maxDelta = max(abs(pp["x"] - machine.origin[0]), abs(pp[ "y"] - machine.origin[1]))
            #if change is larger than 2% of factory size, update
            if maxDelta > max(*self.env.factory.FACTORYDIMENSIONS)*0.02:
                self.env.factory.machine_dict[index].translate_Item(pp["x"],pp[ "y"])
                self.update_needed()

        elif index in self.env.factory.machine_dict and "u" in pp and "v" in pp:
            #input 0-1 -> scale to window coordinates -> scale to current zoom
            scaled_u = np.around(pp["u"],2) * self.window_size[0] / self.currentScale + self.env.factory.DRAWINGORIGIN[0]
            scaled_v = np.around(pp["v"],2) * self.window_size[1] / self.currentScale + self.env.factory.DRAWINGORIGIN[1]
            #Grab machine center instead of origin
            scaled_u = scaled_u - self.env.factory.machine_dict[index].width/2
            scaled_v = scaled_v - self.env.factory.machine_dict[index].height/2
            #Calculate distance of change
            machine = self.env.factory.machine_dict[index]
            maxDelta = max(abs(scaled_u - machine.origin[0]),abs(scaled_v - machine.origin[1]))
            #if change is larger than 2% of factory size, update
            if maxDelta > max(*self.env.factory.FACTORYDIMENSIONS)*0.02:
                self.env.factory.machine_dict[index].translate_Item(scaled_u,scaled_v)
                self.update_needed()

        elif index in self.mobile_dict and "x" in pp and "y" in pp:
            self.mobile_dict[index].translate_Item(pp["x"],pp["y"])

        elif index in self.mobile_dict and "u" in pp and "v" in pp:
            #input 0-1 -> scale to window coordinates -> scale to current zoom
            scaled_u = np.around(pp["u"],2) * self.window_size[0] / self.currentScale + self.env.factory.DRAWINGORIGIN[0]
            scaled_v = np.around(pp["v"],2) * self.window_size[1] / self.currentScale + self.env.factory.DRAWINGORIGIN[1]
            self.mobile_dict[index].translate_Item(scaled_u,scaled_v)
        else:
            print("MQTT message malformed. Needs JSON Payload containing x and y coordinates (u, v coordinates) and valid machine index\n",index)
    
    def handleMQTT_Geometry(self, topic, payload):
        pp = json.loads(payload)
        index = self.extractID(topic)
        if index is not None and "topleft_x5" in pp and "topleft_y" in pp and "bottomright_x" in pp and "bottomright_y" in pp:
            self.F_add_rect((pp["topleft_x"],pp["topleft_y"]),(pp["bottomright_x"],pp["bottomright_y"]), gid=index)
        elif index is not None and "points" in pp:
            self.F_add_poly(pp["points"], gid=index)
        elif index is not None and index in self.env.factory.machine_dict and pp == []:
            self.env.factory.machine_dict.pop(index)
            self.update_needed()
        else:
            print("MQTT message malformed. Needs JSON Payload containing topleft_x, topleft_y, bottomright_x, bottomright_y of rectangle or coordinates of a polygon")



if __name__ == "__main__":
    factorySimLive.run()
