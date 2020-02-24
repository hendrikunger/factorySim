#!/usr/bin/env python3

import os
import math
import random
from itertools import combinations
from time import time

import ifcopenshell
import cairo
import pandas as pd

from Helpers.MFO import MFO 
from Polygon import Polygon as Poly
import Polygon.IO


class FactorySim:
 #------------------------------------------------------------------------------------------------------------
 # Loading
 #------------------------------------------------------------------------------------------------------------
    def __init__(self, path_to_ifc_file, width=1000, heigth=1000, randseed = "Kekse", path_to_materialflow_file = None, outputfile = "Out", randomMF = False):
        self.WIDTH = width
        self.HEIGHT = heigth

        self.RANDSEED = randseed
        random.seed(randseed)
        self.timezero = time()
        self.lasttime = 0
        
        self.outputfile = outputfile

        #ifc_file = ifcopenshell.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"Input","EP_v23_S1_clean.ifc"))
        self.ifc_file = ifcopenshell.open(path_to_ifc_file)

        self.printTime("Datei geladen")
        #Importing Machines
        self.machine_list = self.importIFC_Data(self.ifc_file, "IFCBUILDINGELEMENTPROXY")
        self.wall_list = self.importIFC_Data(self.ifc_file, "IFCWALL") 
        self.machineCollisionList = []
        self.wallCollisionList = []

        self.currentRating = 0 # Holds the Rating of the current state of the Layout 
        
        allElements = Poly()
        
        for wall in self.wall_list:
            for polygon in wall.polylist:
                for loop in polygon:
                    allElements.addContour(loop)
        for machine in self.machine_list:
            for polygon in machine.polylist:
                for loop in polygon:
                    allElements.addContour(loop)

        #Polygon.IO.writeSVG('test.svg', allElements, width=1000, height=1000)

        #Shifting and Scaling to fit into target Output Size
        boundingBox = allElements.boundingBox()      
        #boundingBox[0]     min_value_x
        #boundingBox[1]     max_value_x
        #boundingBox[2]     min_value_y
        #boundingBox[3]     max_value_y
        self.printTime("Boundingbox erstellt")
        scale_x = self.WIDTH / (boundingBox[1] - boundingBox[0])
        scale_y = self.HEIGHT / (boundingBox[3] - boundingBox[2])
        scale = min(scale_x, scale_y)

        if((boundingBox[1] > self.WIDTH) or (boundingBox[3] > self.HEIGHT)):
            for machine in self.machine_list:
                machine.scale_Points(scale, scale, -boundingBox[0], -boundingBox[2])
            for wall in self.wall_list:
                wall.scale_Points(scale, scale, -boundingBox[0], -boundingBox[2])
        self.printTime("Skaliert")
        
        #Finding Centers and merging internal polygons
        for machine in self.machine_list:      
            machine.finish()
        for wall in self.wall_list:      
            wall.finish()
        self.printTime("Mitten gefunden und finalisiert")
        
        #Find Collisions
        self.findCollisions()
        
        #Import Materialflow from Excel
        if path_to_materialflow_file is None and randomMF is True:
            path_to_materialflow_file = self.random_Material_Flow()
        if path_to_materialflow_file is not None:

            self.materialflow_file = pd.read_csv(path_to_materialflow_file, skipinitialspace=True, encoding= "utf-8")
            #Rename Colums
            indexes = self.materialflow_file.columns.tolist()
            self.materialflow_file.rename(columns={indexes[0]:'from', indexes[1]:'to', indexes[2]:'intensity'}, inplace=True)
            #Group by from and two, add up intensity of all duplicates in intensity_sum
            self.materialflow_file['intensity_sum'] = self.materialflow_file.groupby(by=['from', 'to'])['intensity'].transform('sum')
            #drop the duplicates and refresh index
            self.materialflow_file = self.materialflow_file.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)
        self.printTime("Materialfluss geladen")

        #Evaluate current State
        self.evaluate()
        
      
  #------------------------------------------------------------------------------------------------------------
    def importIFC_Data(self, ifc_file, elementName):
        elementlist = []
        elements = ifc_file.by_type(elementName)
        for element in elements:
            #get origin
            origin = element.ObjectPlacement.RelativePlacement.Location.Coordinates
            #element.ObjectPlacement.RelativePlacement.Axis.DirectionRatios[0]
            #element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]

            #get rotation
            x = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]
            y = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[1]
            rotation = math.atan2(y,x)

            #create MFO Object
            mfo_object = MFO(gid=element.GlobalId, 
                name=element.Name, 
                origin_x=origin[0],
                origin_y=origin[1],
                origin_z=origin[2],
                rotation=rotation)

            #points = element.Representation.Representations[0].Items[0].Outer.CfsFaces[0].Bounds[0].Bound.Polygon
            #Always choose Representation 0
            items = element.Representation.Representations[0].Items

            #Parse imported data into MFO Object (Machine)
            for item in items:
                loops = item.Outer.CfsFaces
                for loop in loops:
                    bounds = loop.Bounds
                    for bound in bounds:
                        type = bound.get_info()['type']
                        points = bound.Bound.Polygon
                        #Remove z Dimension of Polygon Coordinates
                        pointlist = [[point.Coordinates[0], point.Coordinates[1]] for point in points ]
                        #Check if face is a surface or a hole
                        if (type == "IfcFaceBound"):
                            #case Hole
                            mfo_object.add_Loop(pointlist, isHole=True)
                        else:
                            #Case surface
                            mfo_object.add_Loop(pointlist)
                            
                    
                mfo_object.close_Item()
            mfo_object.rotate_translate_Item() 
            elementlist.append(mfo_object)
            
        self.printTime(f"{elementName} geparsed")
        return elementlist
    
 #------------------------------------------------------------------------------------------------------------
 # Evaluation
 #------------------------------------------------------------------------------------------------------------
    def evaluate(self):
        self.currentRating = self.evaluateMF()                 
        self.printTime(f"Bewertung des Layouts abgeschlossen - {self.currentRating}")

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF(self):
        machine_dict = {machine.name: machine for machine in self.machine_list}

        for index, row in self.materialflow_file.iterrows():
            x1 = machine_dict[row['from']].center.x
            y1 = machine_dict[row['from']].center.y
            x2 = machine_dict[row['to']].center.x
            y2 = machine_dict[row['to']].center.y
            self.materialflow_file.loc[index , 'distance'] = math.sqrt(math.pow(x1-x2,2) + math.pow(y1-y2,2))
        
        print(self.materialflow_file)

        self.printTime("Bewertung des Materialfluss abgeschlossen")
        return 1

 #------------------------------------------------------------------------------------------------------------
 # Collision Detection
 #------------------------------------------------------------------------------------------------------------
    def findCollisions(self):
        #Machines with Machines
        self.machineCollisionList = []       
        for a,b in combinations(self.machine_list, 2):
            if a.hull.overlaps(b.hull):
                print(f"Kollision zwischen {a.gid} und {b.gid} gefunden.")
                self.machineCollisionList.append(a.hull & b.hull)
        #Machines with Walls     
        self.wallCollisionList = []
        for a in self.wall_list:
            for b in self.machine_list:
                if a.poly.overlaps(b.hull):
                    print(f"Kollision Wand {a.gid} und Maschine {b.gid} gefunden.")
                    self.wallCollisionList.append(a.poly & b.hull)
                    
        self.printTime("Kollisionen berechnen abgeschlossen")

 #------------------------------------------------------------------------------------------------------------
 # Drawing
 #------------------------------------------------------------------------------------------------------------
    def drawPositions(self, drawMaterialflow = True, drawMachineCenter = False, drawWalls = True):   
        #Drawing
        #Machine Positions
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        ctx = cairo.Context(surface)
        ctx.scale(1.0, -1.0)
        ctx.translate(0.0,-self.HEIGHT)
        ctx.rectangle(0, 0, self.WIDTH, self.HEIGHT)  
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.fill()

      #Walls
        if drawWalls:
            ctx.set_fill_rule(cairo.FillRule.EVEN_ODD)
            for wall in self.wall_list:
                ctx.set_source_rgb(0.5, 0.5, 0.5)
                #draw all walls
                for i, loop in enumerate(wall.poly):
                    if(wall.poly.isHole(i) is False):
                        if(len(loop) > 0):
                            ctx.move_to(loop[0][0], loop[0][1])
                            for point in loop:
                                ctx.line_to(point[0], point[1])
                            ctx.close_path()
                            ctx.fill()
                #draw all holes
                ctx.set_source_rgb(1, 1, 1)
                for i, loop in enumerate(wall.poly):
                    if(wall.poly.isHole(i)):
                        if(len(loop) > 0):
                            ctx.move_to(loop[0][0], loop[0][1])
                            for point in loop:
                                ctx.line_to(point[0], point[1])
                            ctx.close_path()
                            ctx.fill()
                            
        #draw machine positions
        ctx.set_fill_rule(cairo.FillRule.WINDING)
        for machine in self.machine_list:
            ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
            for loop in machine.hull:
                if(len(loop) > 0):
                    ctx.move_to(loop[0][0], loop[0][1])
                    for point in loop:
                        ctx.line_to(point[0], point[1])
                        #print(F"{machine.gid}, X:{point.x}, Y:{point.y}")
                    ctx.close_path()
                    ctx.fill()
      
        #Machine Centers
            if (machine.center is not None and drawMachineCenter):
                ctx.set_source_rgb(0, 0, 0)
                ctx.arc(machine.center.x, machine.center.y, 5, 0, 2*math.pi)
                ctx.fill()

        #Material Flow
        if drawMaterialflow:
            machine_dict = {machine.name: machine for machine in self.machine_list}

            mf_max = self.materialflow_file.max()["intensity_sum"]
            mf_min = self.materialflow_file.min()["intensity_sum"]

            for index, row in self.materialflow_file.iterrows():
                try:
                    #print(F"Draw Line from {machine_dict[row[0]].name} to {machine_dict[row[1]].name} with Intensity {row[2]}")
                    ctx.set_source_rgb(machine_dict[row[0]].color[0], machine_dict[row[0]].color[1], machine_dict[row[0]].color[2])
                    ctx.move_to(machine_dict[row[0]].center.x, machine_dict[row[0]].center.y)
                    ctx.line_to(machine_dict[row[1]].center.x, machine_dict[row[1]].center.y)
                    ctx.set_line_width(row["intensity_sum"]/(mf_max - mf_min) * 20)
                    ctx.stroke()   
                except KeyError:
                    print(f"Error in Material Flow Drawing - Machine {row[0]} or {row[1]} not defined")
                    continue
        self.printTime("Machinenpositionen gezeichnet")
    
        return surface
    
    
#------------------------------------------------------------------------------------------------------------  
    def drawDetailedMachines(self, randomcolors = False):   
        #Drawing
        #Machine Positions
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        ctx = cairo.Context(surface)
        ctx.scale(1.0, -1.0)
        ctx.translate(0.0,-self.HEIGHT)
        ctx.rectangle(0, 0, self.WIDTH, self.HEIGHT)  
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.fill()
        
        #draw machine positions
        for machine in self.machine_list:
            
            for i, polygon in enumerate(machine.polylist):
                if (i == 0):
                    ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                elif randomcolors:
                        ctx.set_source_rgb(random.random(), random.random(), random.random())
                else:
                    ctx.set_source_rgb(0.4, 0.4, 0.4)
                                     
                #draw all outer contours
                for i, loop in enumerate(polygon):
                    if(polygon.isHole(i) is False):
                        if(len(loop) > 0):
                            ctx.move_to(loop[0][0], loop[0][1])
                            for point in loop:
                                ctx.line_to(point[0], point[1])
                            ctx.close_path()
                            ctx.fill()
                #draw all holes
                if randomcolors:
                    ctx.set_source_rgb(random.random(), random.random(), random.random())
                else:
                    ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                    
                for i, loop in enumerate(polygon):
                    if(polygon.isHole(i)):
                        if(len(loop) > 0):
                            ctx.move_to(loop[0][0], loop[0][1])
                            for point in loop:
                                ctx.line_to(point[0], point[1])
                            ctx.close_path()
                            ctx.fill()

        self.printTime("Detailierte Machinenpositionen gezeichnet")
        return surface

 #------------------------------------------------------------------------------------------------------------
    def drawCollisions(self, drawWalls=True):

        #Machine Collisions
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        ctx = cairo.Context(surface)
        ctx.scale(1.0, -1.0)
        ctx.translate(0.0,-self.HEIGHT)
        ctx.rectangle(0, 0, self.WIDTH, self.HEIGHT)  
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.fill()

        #Drawing collisions between machines
        for polygon in self.machineCollisionList:
            for loop in polygon:
                ctx.set_source_rgb(1, 0 , 0)
                if(len(loop) > 0):
                    ctx.move_to(loop[0][0], loop[0][1])
                    for point in loop:
                        ctx.line_to(point[0], point[1])
                        #print(F"{machine.gid}, X:{point.x}, Y:{point.y}")
                    ctx.close_path()
                    ctx.fill()
                    
        #Drawing collisions between machines and walls
        if(drawWalls):
            for polygon in self.wallCollisionList:
                for loop in polygon:
                    ctx.set_source_rgb(0, 0 , 1)
                    if(len(loop) > 0):
                        ctx.move_to(loop[0][0], loop[0][1])
                        for point in loop:
                            ctx.line_to(point[0], point[1])
                            #print(F"{machine.gid}, X:{point.x}, Y:{point.y}")
                        ctx.close_path()
                        ctx.fill()            
                    
        self.printTime("Kollisionen gezeichnet")
        return surface
 #------------------------------------------------------------------------------------------------------------
 # Helpers  
 #------------------------------------------------------------------------------------------------------------
    def random_Material_Flow(self):
        random.seed()
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Input","Random_Materialflow.csv")

        with open(path, 'w') as file:
            file.write(f"From, To, Intensity\n")
            for _ in range(0,30):
                samples = random.sample(self.machine_list, k=2)
                file.write(f"{samples[0].name}, {samples[1].name},{random.randint(1,100)}\n")
        random.seed(self.RANDSEED)
        self.printTime("Zufälligen Materialfluss erstellt")
        return path
    
    
#------------------------------------------------------------------------------------------------------------
    def printTime(self, text):
        self.lasttime = (time() - self.timezero - self.lasttime)
        print(round(self.lasttime * 1000, 3) , " - " + text)



#------------------------------------------------------------------------------------------------------------
def main():
           
    outputfile ="Out"

    
    #filename = "Overlapp"
    #filename = "EP_v23_S1_clean"
    filename = "Simple"
    
    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input",  filename + ".ifc")


    materialflowpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", filename + "_Materialflow_doubles.csv")
    #materialflowpath = None
    
    demoFactory = FactorySim(ifcpath, path_to_materialflow_file = materialflowpath, randomMF = True)
 
    #Machine Positions Output to PNG
    machinePositions = demoFactory.drawPositions(drawMaterialflow = True, drawMachineCenter = True)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "Output", 
        F"{outputfile}_machines.png")
    machinePositions.write_to_png(path) 
 
    #detailed Machines Output to PNG
    detailedMachines = demoFactory.drawDetailedMachines(randomcolors = True)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "Output", 
        F"{outputfile}_detailed_machines.png")
    detailedMachines.write_to_png(path)
 
    #Machine Collisions Output to PNG
    Collisions = demoFactory.drawCollisions()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "Output", 
        F"{outputfile}_machine_collsions.png")
    Collisions.write_to_png(path) 
    
    print(f"Total runtime: {round((time() - demoFactory.timezero) * 1000, 3)} ")

if __name__ == "__main__":
    main()