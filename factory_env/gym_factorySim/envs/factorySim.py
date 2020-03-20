#!/usr/bin/env python3

import os
import math
import random
from itertools import combinations
from time import time

import ifcopenshell
import cairo
import pandas as pd
from fabulous.color import fg256, bg256, bold


from Polygon import Polygon as Poly
import Polygon.IO
import numpy as np


class FactorySim:
 #------------------------------------------------------------------------------------------------------------
 # Loading
 #------------------------------------------------------------------------------------------------------------
    def __init__(self, path_to_ifc_file, width=1000, heigth=1000, randseed = None, path_to_materialflow_file = None, outputfile = "Out", randomMF = False, verboseOutput = 0):
        self.WIDTH = width
        self.HEIGHT = heigth

        self.verboseOutput = verboseOutput
        self.RANDSEED = randseed
        random.seed(randseed)
            
        self.timezero = time()
        self.lasttime = 0
        
        self.outputfile = outputfile

        #ifc_file = ifcopenshell.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"Input","EP_v23_S1_clean.ifc"))
        self.ifc_file = ifcopenshell.open(path_to_ifc_file)

        if(self.verboseOutput >= 3):
            self.printTime("Datei geladen")
        #Importing Machines
        self.machine_list = self.importIFC_Data(self.ifc_file, "IFCBUILDINGELEMENTPROXY")
        self.wall_list = self.importIFC_Data(self.ifc_file, "IFCWALL") 
        self.machineCollisionList = []
        self.wallCollisionList = []

        self.currentRating    = 0 # Holds the Rating of the current state of the Layout 
        
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
        self.min_value_x = boundingBox[0]     
        self.max_value_x = boundingBox[1]     
        self.min_value_y = boundingBox[2]     
        self.max_value_y = boundingBox[3]     
        if(self.verboseOutput >= 3):
            self.printTime("Boundingbox erstellt")
 
        if((self.max_value_x > self.WIDTH) or (self.max_value_y > self.HEIGHT)):
            #Calculate new scale
            scale_x = self.WIDTH / (self.max_value_x - self.min_value_x)
            scale_y = self.HEIGHT / (self.max_value_y - self.min_value_y)
            scale = min(scale_x, scale_y)

            for machine in self.machine_list:
                machine.scale_Points(scale, scale, -self.min_value_x, -self.min_value_y)
            for wall in self.wall_list:
                wall.scale_Points(scale, scale, -self.min_value_x, -self.min_value_y)
            self.min_value_x = (self.min_value_x - self.min_value_x) * scale   
            self.max_value_x = (self.max_value_x - self.min_value_x) * scale   
            self.min_value_y = (self.min_value_y - self.min_value_y) * scale   
            self.max_value_y = (self.max_value_y - self.min_value_y) * scale 
        if(self.verboseOutput >= 3):
            self.printTime("Skaliert")
        
        #Finding Centers and merging internal polygons
        for machine in self.machine_list:      
            machine.finish()
        for wall in self.wall_list:      
            wall.finish()
        if(self.verboseOutput >= 3):
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
            #normalise intensity sum 
            self.materialflow_file['intensity_sum_norm'] = self.materialflow_file['intensity_sum'] / self.materialflow_file.max()["intensity_sum"]
            #use machine index as sink and source for materialflow
            machine_dict = {machine.name: index for index, machine in enumerate(self.machine_list)}
            self.materialflow_file[['from','to']] = self.materialflow_file[['from','to']].replace(machine_dict)
            #set initial values for costs
            self.materialflow_file['costs'] = 0
    
        if(self.verboseOutput >= 3):
            self.printTime("Materialfluss geladen")

        
      
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
            mfo_object.updatePosition() 
            elementlist.append(mfo_object)
            
        if(self.verboseOutput >= 3):
            self.printTime(f"{elementName} geparsed")
        return elementlist

 #------------------------------------------------------------------------------------------------------------
 # Update Machines
 #------------------------------------------------------------------------------------------------------------
    def update(self, machineIndex, xPosition = 0, yPosition = 0, rotation = None):
        
        if(self.verboseOutput >= 2):
            print(f"Update: {self.machine_list[machineIndex].name} - X: {xPosition:1.1f} Y: {yPosition:1.1f} R: {rotation:1.2f} ")

        if (rotation is not None):
            mappedRot = self.mapRange(rotation, (0,1), (0, 2*math.pi))
            self.machine_list[machineIndex].rotate_Item(mappedRot)

        #Max Value should move machine to the rightmost or topmost position without moving out of the image
        mappedXPos = self.mapRange(xPosition, (0,1), (0,self.WIDTH - self.machine_list[machineIndex].width))
        mappedYPos = self.mapRange(yPosition, (0,1), (0,self.HEIGHT - self.machine_list[machineIndex].height))

        self.machine_list[machineIndex].translate_Item(mappedXPos, mappedYPos)
        self.findCollisions()
        if(self.verboseOutput >= 3):
            self.printTime(f"{self.machine_list[machineIndex].name} geupdated")

    
 #------------------------------------------------------------------------------------------------------------
 # Evaluation
 #------------------------------------------------------------------------------------------------------------
    def evaluate(self):
        ratingMF = self.evaluateMF()          
        if(self.verboseOutput >= 3):
            self.printTime("Bewertung des Materialfluss abgeschlossen")
        ratingCollision = self.evaluateCollision()          
        if(self.verboseOutput >= 3):
            self.printTime("Kollisionsbewertung abgeschlossen")

        self.currentRating = ratingMF + ratingCollision
        if(self.verboseOutput >= 3):
            self.printTime("Bewertung des Layouts abgeschlossen")
        if(self.verboseOutput >= 1):
            print("Total Rating " + bg256("blue", fg256("red" ,f"{self.currentRating:1.2f}")) + ", ",
                "MaterialFlow " + bg256("blue", f"{ratingMF:1.2f}") + ", ",
                "Kollisionen " + bg256("blue", f"{ratingCollision:1.2f}") + ", ",
                "Test " + bg256("blue", f"{ratingMF:1.2f}")+ ", ",
                "Test " + bg256("blue", f"{ratingMF:1.2f}"))
        
        return self.currentRating

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF_Helper(self, source, sink): 
        x1 = self.machine_list[int(source)].center.x
        y1 = self.machine_list[int(source)].center.y
        x2 = self.machine_list[int(sink)].center.x
        y2 = self.machine_list[int(sink)].center.y
        return math.sqrt(math.pow(x1-x2,2) + math.pow(y1-y2,2))

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF(self):
        self.materialflow_file['distance'] = self.materialflow_file.apply(lambda row: self.evaluateMF_Helper(row['from'], row['to']), axis=1)

        #sum of all costs /  maximum intensity (intensity sum norm * 1) 
        maxDistance = max(self.max_value_x,  self.max_value_y)
        self.materialflow_file['distance_norm'] = self.materialflow_file['distance'] / maxDistance
        self.materialflow_file['costs'] = self.materialflow_file['distance_norm'] * self.materialflow_file['intensity_sum_norm'] 
        output = self.materialflow_file['costs'].sum() / self.materialflow_file['intensity_sum_norm'].sum()

        return output

 #------------------------------------------------------------------------------------------------------------
    def evaluateCollision(self):
        
        machineCollisionArea = 0
        wallCollisionArea = 0
        totalMachineArea = 0

        for collision in self.machineCollisionList:
            machineCollisionArea += collision.area()
        for collision in self.wallCollisionList:
           wallCollisionArea += collision.area() 
        for machine in self.machine_list:      
            totalMachineArea += machine.hull.area()

        #print(len(list(combinations(self.machine_list, 2))))
        nMachineCollisions = len(self.machineCollisionList)
        nWallCollosions = len(self.wallCollisionList)

        output = 1 - (0.25 * nMachineCollisions) - (0.5 * nWallCollosions)
  
        return output


 #------------------------------------------------------------------------------------------------------------
 # Collision Detection
 #------------------------------------------------------------------------------------------------------------
    def findCollisions(self):
        #Machines with Machines
        self.machineCollisionList = []       
        for a,b in combinations(self.machine_list, 2):
            if a.hull.overlaps(b.hull):
                if(self.verboseOutput >= 4):
                    print(fg256("red", f"Kollision zwischen {a.name} und {b.name} gefunden."))
                self.machineCollisionList.append(a.hull & b.hull)
        #Machines with Walls     
        self.wallCollisionList = []
        for a in self.wall_list:
            for b in self.machine_list:
                if a.poly.overlaps(b.hull):
                    if(self.verboseOutput >= 4):
                        print(fg256("red", f"Kollision Wand {a.gid} und Maschine {b.name} gefunden."))
                    self.wallCollisionList.append(a.poly & b.hull)
                    
        if(self.verboseOutput >= 3):
            self.printTime("Kollisionen berechnen abgeschlossen")

 #------------------------------------------------------------------------------------------------------------
 # Drawing
 #------------------------------------------------------------------------------------------------------------
    def drawPositions(self, surfaceIn=None, drawMaterialflow = True, drawMachineCenter = False, drawOrigin = True, drawMachineBaseOrigin = False, drawWalls = True, highlight = None):   
        #Drawing
        if(surfaceIn is None):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        else:
            surface = surfaceIn
        ctx = cairo.Context(surface)
        ctx.scale(1.0, -1.0)
        ctx.translate(0.0,-self.HEIGHT)
        if(surfaceIn is None):
            ctx.rectangle(0, 0, self.WIDTH, self.HEIGHT)  
            ctx.set_source_rgb(1.0, 1.0, 1.0)
            ctx.fill()

      #Walls
        if drawWalls:
            ctx.set_fill_rule(cairo.FillRule.EVEN_ODD)
            for wall in self.wall_list:
                ctx.set_source_rgb(0, 0, 0)
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
        ctx.set_line_width(4)
        for index, machine in enumerate(self.machine_list):
                
            for loop in machine.hull:
                if(len(loop) > 0):
                    ctx.move_to(loop[0][0], loop[0][1])
                    for point in loop:
                        ctx.line_to(point[0], point[1])
                        #print(F"{machine.gid}, X:{point.x}, Y:{point.y}")
                    ctx.close_path()

                    #no highlights
                    if(highlight is  None):
                        ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                    #highlighted machine
                    elif(index == highlight):
                        ctx.set_source_rgb(1, 0.4, 0)
                    #other machines
                    else:
                        ctx.set_source_rgb(0.4, 0.4, 0.4)

                    ctx.fill_preserve()
                    ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                    ctx.stroke()
      
        #Machine Centers
            if (machine.center is not None and drawMachineCenter):
                ctx.set_source_rgb(0, 0, 0)
                ctx.arc(machine.center.x, machine.center.y, 5, 0, 2*math.pi)
                ctx.fill()

        #Machine Origin 
            if (machine.origin is not None and drawOrigin):
                ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                ctx.arc(machine.origin.x, machine.origin.y, 5, 0, 2*math.pi)
                ctx.fill()

        #Machine Base Origin
            if (machine.baseOrigin is not None and drawMachineBaseOrigin):
                ctx.set_source_rgb(1,0,0)
                ctx.arc(machine.baseOrigin.x, machine.baseOrigin.y, 3, 0, 2*math.pi)
                ctx.fill()

        #Material Flow
        if drawMaterialflow:

            mf_max = self.materialflow_file.max()["intensity_sum"]
            mf_min = self.materialflow_file.min()["intensity_sum"]

            for index, row in self.materialflow_file.iterrows():
                try:
                    ctx.set_source_rgb(self.machine_list[int(row['from'])].color[0], self.machine_list[int(row['from'])].color[1], self.machine_list[int(row['from'])].color[2])
                    ctx.move_to(self.machine_list[int(row['from'])].center.x, self.machine_list[int(row['from'])].center.y)
                    ctx.line_to(self.machine_list[int(row['to'])].center.x, self.machine_list[int(row['to'])].center.y)
                    ctx.set_line_width(row["intensity_sum"]/(mf_max - mf_min) * 20)
                    ctx.stroke()   
                except KeyError:
                    print(f"Error in Material Flow Drawing - Machine {row[0]} or {row[1]} not defined")
                    continue
        if(self.verboseOutput >= 3):
            self.printTime("Machinenpositionen gezeichnet")
    
        return surface
    
    
#------------------------------------------------------------------------------------------------------------  
    def drawDetailedMachines(self, surfaceIn=None, randomcolors = False):   
        #Drawing
        #Machine Positions
        if(surfaceIn is None):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        else:
            surface = surfaceIn
        ctx = cairo.Context(surface)
        ctx.scale(1.0, -1.0)
        ctx.translate(0.0,-self.HEIGHT)
        if(surfaceIn is None):
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

        if(self.verboseOutput >= 3):
            self.printTime("Detailierte Machinenpositionen gezeichnet")
        return surface

 #------------------------------------------------------------------------------------------------------------
    def drawCollisions(self, surfaceIn=None, drawWalls=True):

        #Machine Collisions
        if(surfaceIn is None):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        else:
            surface = surfaceIn
        ctx = cairo.Context(surface)
        ctx.scale(1.0, -1.0)
        ctx.translate(0.0,-self.HEIGHT)
        if(surfaceIn is None):
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
                    
        if(self.verboseOutput >= 3):
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
        if(self.verboseOutput >= 3):
            self.printTime("Zufälligen Materialfluss erstellt")
        return path
    
    
#------------------------------------------------------------------------------------------------------------
    def printTime(self, text):
        number = (time() - self.timezero - self.lasttime)
        self.lasttime += number
        number = round(number * 1000, 2)
        print(bold(fg256("green", f'{number:6.2f}ms')) , "- " + text)

  #------------------------------------------------------------------------------------------------------------  
    def mapRange(self,s,  a, b):
	    (a1, a2), (b1, b2) = a, b
	    return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))



#------------------------------------------------------------------------------------------------------------
def main():
    outputfile ="Out"

    #filename = "Overlapp"
    filename = "Basic"
    #filename = "EP_v23_S1_clean"
    #filename = "Simple"
    #filename = "SimpleNoCollisions"

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "..",
        "Input",  
        filename + ".ifc")

    file_name, _ = os.path.splitext(ifcpath)
    materialflowpath = file_name + "_Materialflow.csv"
    #materialflowpath = None
    
    demoFactory = FactorySim(ifcpath, path_to_materialflow_file = materialflowpath, randomMF = True, verboseOutput=3)
 
    #Machine Positions Output to PNG
    #machinePositions = demoFactory.drawPositions(drawMaterialflow = True, drawMachineCenter = True)
    machinePositions = demoFactory.drawPositions(drawMaterialflow = True, drawMachineCenter = True, drawMachineBaseOrigin=True, highlight=0)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "Output", 
        F"{outputfile}_machines.png")
    machinePositions.write_to_png(path) 
 
 #------------------------------------------------------------------------------------------------------------------------------------------
    ##detailed Machines Output to PNG
    #detailedMachines = demoFactory.drawDetailedMachines(randomcolors = True)
    #path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #    "Output", 
    #    F"{outputfile}_detailed_machines.png")
    #detailedMachines.write_to_png(path)
 #------------------------------------------------------------------------------------------------------------------------------------------


    #Rate current Layout
    demoFactory.evaluate()

    #Change machine
    #demoFactory.update(0,demoFactory.machine_list[0].origin.x,demoFactory.machine_list[0].origin.y, math.pi/2)
    demoFactory.update(0,1,1, 1)
    #demoFactory.update(0,100,300, math.pi/2)
    #demoFactory.update(0,200,200, math.pi/2)
    #demoFactory.update(4,150,100,0)
    #demoFactory.update(1,-10,200,0)
    #demoFactory.update(0, 50,150,0)

    machinePositions = demoFactory.drawPositions(drawMaterialflow = True, drawMachineCenter = True, drawMachineBaseOrigin=True, highlight=0)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "Output", 
        F"{outputfile}_machines_update.png")
    machinePositions.write_to_png(path) 

    #Machine Collisions Output to PNG
    Collisions = demoFactory.drawCollisions(surfaceIn=machinePositions)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "Output", 
        F"{outputfile}_machine_collsions.png")
    Collisions.write_to_png(path) 

    ##Rate current Layout
    demoFactory.evaluate()

    print("Total runtime: " + bold(fg256("green", bg256("yellow", round((time() - demoFactory.timezero) * 1000, 2)))))

    
if __name__ == "__main__":
    from Helpers.MFO import MFO 
    main()
else:
    from gym_factorySim.envs.Helpers.MFO import MFO
    from gym_factorySim.envs.Helpers.Point3D import Point3D