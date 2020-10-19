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
    def __init__(self, path_to_ifc_file, width=1000, heigth=1000, randseed = None, path_to_materialflow_file = None, outputfile = "Out", randomPos = False, randomMF = False, verboseOutput = 0, maxMF_Elements = None, objectScaling = 1.0):
        self.WIDTH = width
        self.HEIGHT = heigth
        self.MAXMF_ELEMENTS = maxMF_Elements
        self.MAX_STROKE_WIDTH = min(width,heigth) * 0.04 * objectScaling
        self.DOT_RADIUS = self.MAX_STROKE_WIDTH / 4

        self.verboseOutput = verboseOutput
        self.RANDSEED = randseed
        random.seed(randseed)
            
        self.timezero = time()
        self.lasttime = 0
        
        self.outputfile = outputfile

        if(os.path.isdir(path_to_ifc_file)):
            
            chosen_ifc_file = random.choice([x for x in os.listdir(path_to_ifc_file) if ".ifc" in x and "LIB" not in x])
            chosen_ifc_file = os.path.join(path_to_ifc_file, chosen_ifc_file)
        else:
            chosen_ifc_file = path_to_ifc_file
            
        if(self.verboseOutput >= 2):
            print("Lade: ", chosen_ifc_file)

        self.ifc_file = ifcopenshell.open(chosen_ifc_file)

        if(self.verboseOutput >= 3):
            self.printTime("Datei geladen")
        #Importing Machines

        self.machine_list = []
        if(self.MAXMF_ELEMENTS):

            path = os.path.join(os.path.dirname(chosen_ifc_file), "MFO_LIB.ifc")
            if(os.path.exists(path)):
                mfo_ifc_file_path = path
            else:
                mfo_ifc_file_path = chosen_ifc_file

            if(self.verboseOutput >= 2):
                print(f"Lade: Demomaterialflussobjekte. Maximal {self.MAXMF_ELEMENTS} werden aus {mfo_ifc_file_path} geladen.")
            #2 bis MAXMF_ELEMENTS aus der Datei mit Demomaterialflussobjekten laden.
            self.machine_list = self.importIFC_Data(ifcopenshell.open(mfo_ifc_file_path), "IFCBUILDINGELEMENTPROXY", randomMF=self.MAXMF_ELEMENTS)
        else:
            if(self.verboseOutput >= 2):
                print("Nutze alle MF Objekte in der IFC Datei")
            self.machine_list = self.importIFC_Data(self.ifc_file, "IFCBUILDINGELEMENTPROXY")

        #Importing Walls
        self.wall_list = self.importIFC_Data(self.ifc_file, "IFCWALL") 
        self.machineCollisionList = []
        self.wallCollisionList = []

        self.lastRating = 0
        self.currentRating    = 0 # Holds the Rating of the current state of the Layout 
        self.currentMappedRating    = 0 # Holds the normalized Rating of the current state of the Layout 

        self.lastUpdatedMachine = None #Hold uid of last updated machine for collision checking
        self.collisionAfterLastUpdate = False # True if latest update leads to new collsions

        self.episodeCounter = 0
        
        allElements = Poly()
        
        for wall in self.wall_list:
            for polygon in wall.polylist:
                for loop in polygon:
                    allElements.addContour(loop)
        #for machine in self.machine_list:
        #    for polygon in machine.polylist:
        #        for loop in polygon:
        #            allElements.addContour(loop)

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
                machine.scale_Points(scale * objectScaling, scale * objectScaling, -self.min_value_x, -self.min_value_y)
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
        
        #Creating random positions
        if randomPos:
            for index, _ in enumerate(self.machine_list):
                self.update(index,
                    xPosition = random.uniform(-1,1),
                    yPosition = random.uniform(-1,1),
                    rotation = random.uniform(-1,1),
                    massUpdate = True)

        #Find Collisions
        self.findCollisions()
        
        #Import Materialflow from Excel
        if randomMF is True:
            names = []
            for _ in range(0,len(self.machine_list) * 2):
                samples = random.sample(self.machine_list, k=2)
                names.append([samples[0].name, samples[1].name]) 
            self.materialflow_file = pd.DataFrame(data=names, columns=["from", "to"])
            self.materialflow_file['intensity'] = np.random.randint(1,100, size=len(self.materialflow_file))

        elif path_to_materialflow_file is not None:

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
    def importIFC_Data(self, ifc_file, elementName, randomMF=None):
        elementlist = []
        elements = []
        if(randomMF):
            elements = random.choices(ifc_file.by_type(elementName), k=random.randint(2, randomMF))
        else:
            elements = ifc_file.by_type(elementName)
        for index, element in enumerate(elements):
            #get origin
            origin = element.ObjectPlacement.RelativePlacement.Location.Coordinates
            #element.ObjectPlacement.RelativePlacement.Axis.DirectionRatios[0]
            #element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]

            #get rotation
            x = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]
            y = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[1]
            rotation = math.atan2(y,x)
            my_uuid = ""
            if(randomMF):
                my_uuid= "_" + str(index)

            #create MFO Object
            mfo_object = MFO(gid=element.GlobalId, 
                name=element.Name + my_uuid,
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
            
        if(self.verboseOutput >= 1):
            self.printTime(f"{elementName} geparsed")
        return elementlist

 #------------------------------------------------------------------------------------------------------------
 # Update Machines
 #------------------------------------------------------------------------------------------------------------
    def update(self, machineIndex, xPosition = 0, yPosition = 0, rotation = None, skip = 0, massUpdate = False):
        if not massUpdate: self.episodeCounter += 1
        if(skip < 0.8):
            self.lastUpdatedMachine = self.machine_list[machineIndex].gid

            if(self.verboseOutput >= 2):
                print(f"Update: {self.machine_list[machineIndex].name} - X: {xPosition:1.1f} Y: {yPosition:1.1f} R: {rotation:1.2f} ")

            if (rotation is not None):
                mappedRot = self.mapRange(rotation, (-1,1), (0, 2*math.pi))
                self.machine_list[machineIndex].rotate_Item(mappedRot)

            #Max Value should move machine to the rightmost or topmost position without moving out of the image
            mappedXPos = self.mapRange(xPosition, (-1,1), (0,self.WIDTH - self.machine_list[machineIndex].width))
            mappedYPos = self.mapRange(yPosition, (-1,1), (0,self.HEIGHT - self.machine_list[machineIndex].height))
            #mappedXPos = self.mapRange(xPosition, (-1,1), (0,self.WIDTH))
            #mappedYPos = self.mapRange(yPosition, (-1,1), (0,self.HEIGHT))

            self.machine_list[machineIndex].translate_Item(mappedXPos, mappedYPos)
            if not massUpdate:
                self.findCollisions()
            if(self.verboseOutput >= 3):
                self.printTime(f"{self.machine_list[machineIndex].name} geupdated")
        else:
             if(self.verboseOutput >= 2):
                print(f"Update: {self.machine_list[machineIndex].name} - Skipped Update")


    
 #------------------------------------------------------------------------------------------------------------
 # Evaluation
 #------------------------------------------------------------------------------------------------------------
    def evaluate(self):
        output = {}
        output["ratingMF"] = self.evaluateMF()          
        if(self.verboseOutput >= 3):
            self.printTime("Bewertung des Materialfluss abgeschlossen")
        output["ratingCollision"] = self.evaluateCollision()          
        if(self.verboseOutput >= 3):
            self.printTime("Kollisionsbewertung abgeschlossen")


       
        #if(self.episodeCounter < len(self.machine_list)):
        #    self.currentRating = 0
        #el

        if(output["ratingCollision"] == 1):
            self.currentRating = math.pow(output["ratingMF"],3)
        else: 
            self.currentRating = -1
            #if(output["ratingCollision"] >= 0.5):
            #    self.currentRating = 0.1
            #else:
            #    self.currentRating = -1


        #if(self.collisionAfterLastUpdate):
        #    self.currentRating = -0.8
        #elif(output["ratingCollision"] < 1):
        #    self.currentRating = -0.5
        #else:
        #    self.currentRating = self.mapRange(output["ratingMF"],(-2,1),(-1,1))




        ##Normalize
        #if(self.episodeCounter % len(self.machine_list) == 0 ):
        #    self.currentMappedRating = self.mapRange(self.currentRating,(-2,2),(-1,1))
        #
        #elif(self.currentRating > self.lastRating):
        #    self.currentMappedRating = 0.1
        #else:
        #    self.currentMappedRating = -0.2
        #
        #self.lastRating = self.currentRating



        #self.currentMappedRating = self.mapRange(self.currentRating,(-2,2),(-1,1))
        self.currentMappedRating = output["TotalRating"]= self.currentRating

        #print(f"walls: {len(self.wallCollisionList)}, machines: {len(self.machineCollisionList)}, count m: {len(self.machine_list)}")
        #if(len(self.wallCollisionList) + len(self.machineCollisionList) >=len(self.machine_list)):
        #    done = True
        #else:
        #    done = False


        #if(self.episodeCounter >= 3 * len(self.machine_list)):
        if(self.episodeCounter > len(self.machine_list)+1):
            done = True
            output["done"] = True
        else:
            done = False     
        #done = False      


        if(self.verboseOutput >= 3):
            self.printTime("Bewertung des Layouts abgeschlossen")
        if(self.verboseOutput >= 1):
            print("Total Rating " + bg256("blue", fg256("red" ,f"{self.currentMappedRating:1.2f}")) + ", ",
                "Raw Rating " + bg256("yellow", fg256("black" ,f"{self.currentRating:1.2f}")) + ", ",
                "MaterialFlow " + bg256("blue", f"{output['ratingMF']:1.2f}") + ", ",
                "Kollisionen " + bg256("blue", f"{output['ratingCollision']:1.2f}"))

        return self.currentMappedRating, self.currentRating, output, done

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF_Helper(self, source, sink): 
        source_center = self.machine_list[int(source)].center
        sink_center = self.machine_list[int(sink)].center
        return math.sqrt(math.pow(source_center.x-sink_center.x,2) + math.pow(source_center.y-sink_center.y,2))

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF(self):
        self.materialflow_file['distance'] = self.materialflow_file.apply(lambda row: self.evaluateMF_Helper(row['from'], row['to']), axis=1)
        #sum of all costs /  maximum intensity (intensity sum norm * 1) 
        maxDistance = max(self.max_value_x,  self.max_value_y)
        self.materialflow_file['distance_norm'] = self.materialflow_file['distance'] / maxDistance
        self.materialflow_file['costs'] = self.materialflow_file['distance_norm'] * self.materialflow_file['intensity_sum_norm']
        output = 1 - (math.pow(self.materialflow_file['costs'].sum(),2) / self.materialflow_file['intensity_sum_norm'].sum())
        if(output < 0): output = 0

        return output

 #------------------------------------------------------------------------------------------------------------
    def evaluateCollision(self):
        
        #machineCollisionArea = 0
        #wallCollisionArea = 0
        #totalMachineArea = 0
        # 
        #for collision in self.machineCollisionList:
        #    machineCollisionArea += collision.area()
        #for collision in self.wallCollisionList:
        #   wallCollisionArea += collision.area() 
        #for machine in self.machine_list:      
        #    totalMachineArea += machine.hull.area()

        #print(len(list(combinations(self.machine_list, 2))))
        nMachineCollisions = len(self.machineCollisionList)
        nWallCollosions = len(self.wallCollisionList)

        #If latest update leads to collision give worst rating.
        #if(self.collisionAfterLastUpdate):
        #    output = -3
        #else:
        #    output = 1 - (0.5 * nMachineCollisions) - (0.5 * nWallCollosions)
    
        output = 1 - (0.5 * nMachineCollisions) - (0.5 * nWallCollosions)
        
        return output


 #------------------------------------------------------------------------------------------------------------
 # Collision Detection
 #------------------------------------------------------------------------------------------------------------
    def findCollisions(self):
        self.collisionAfterLastUpdate = False
        #Machines with Machines
        self.machineCollisionList = []       
        for a,b in combinations(self.machine_list, 2):
            if a.hull.overlaps(b.hull):
                if(self.verboseOutput >= 4):
                    print(fg256("red", f"Kollision zwischen {a.name} und {b.name} gefunden."))
                self.machineCollisionList.append(a.hull & b.hull)
                if(a.gid == self.lastUpdatedMachine or b.gid == self.lastUpdatedMachine): self.collisionAfterLastUpdate = True
        #Machines with Walls     
        self.wallCollisionList = []
        for a in self.wall_list:
            for b in self.machine_list:
                if a.poly.overlaps(b.hull):
                    if(self.verboseOutput >= 4):
                        print(fg256("red", f"Kollision Wand {a.name} und Maschine {b.name} gefunden."))
                    self.wallCollisionList.append(a.poly & b.hull)
                    if(b.gid == self.lastUpdatedMachine): self.collisionAfterLastUpdate = True
                    
        if(self.verboseOutput >= 3):
            self.printTime("Kollisionen berechnen abgeschlossen")

 #------------------------------------------------------------------------------------------------------------
 # Drawing
 #------------------------------------------------------------------------------------------------------------
    def drawPositions(self, surfaceIn=None, scale = 1, drawColors = True, drawMachines = True, drawMaterialflow = True, drawMachineCenter = False, drawOrigin = True, drawMachineBaseOrigin = False, drawWalls = True, highlight = None):   
        #Drawing
        if(surfaceIn is None):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH * scale, self.HEIGHT * scale)
        else:
            surface = surfaceIn
        ctx = cairo.Context(surface)
        ctx.scale(1.0 * scale, -1.0 * scale)
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
        if drawMachines:
            ctx.set_fill_rule(cairo.FillRule.WINDING)
            ctx.set_line_width(self.DOT_RADIUS)
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
                            ctx.set_source_rgb(0.9, 0.9, 0.9)
                        #other machines
                        else:
                            ctx.set_source_rgb(0.4, 0.4, 0.4)

                        ctx.fill_preserve()
                        if(drawColors):
                            ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                        else:
                            ctx.set_source_rgb(0.5, 0.5, 0.5)
                        ctx.stroke()

            #Machine Centers
                if (machine.center is not None and drawMachineCenter):
                    ctx.set_source_rgb(0, 0, 0)
                    ctx.arc(machine.center.x, machine.center.y, self.DOT_RADIUS, 0, 2*math.pi)
                    ctx.fill()

            #Machine Origin 
                if (machine.origin is not None and drawOrigin):
                    ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                    ctx.arc(machine.origin.x, machine.origin.y, self.DOT_RADIUS, 0, 2*math.pi)
                    ctx.fill()

            #Machine Base Origin
                if (machine.baseOrigin is not None and drawMachineBaseOrigin):
                    ctx.set_source_rgb(1,0,0)
                    ctx.arc(machine.baseOrigin.x, machine.baseOrigin.y, self.DOT_RADIUS, 0, 2*math.pi)
                    ctx.fill()

        #Material Flow
        if drawMaterialflow:

            for index, row in self.materialflow_file.iterrows():
                try:
                    if(drawColors):
                        ctx.set_source_rgb(self.machine_list[int(row['from'])].color[0], self.machine_list[int(row['from'])].color[1], self.machine_list[int(row['from'])].color[2])
                    else:
                        ctx.set_source_rgb(0.6, 0.6, 0.6)

                    ctx.move_to(self.machine_list[int(row['from'])].center.x, self.machine_list[int(row['from'])].center.y)
                    ctx.line_to(self.machine_list[int(row['to'])].center.x, self.machine_list[int(row['to'])].center.y)
                    ctx.set_line_width(row["intensity_sum_norm"] * self.MAX_STROKE_WIDTH)
                    ctx.stroke()   
                except KeyError:
                    print(f"Error in Material Flow Drawing - Machine {row[0]} or {row[1]} not defined")
                    continue
        if(self.verboseOutput >= 3):
            self.printTime("Maschinenpositionen gezeichnet")
    
        return surface
    
    
#------------------------------------------------------------------------------------------------------------  
    def drawDetailedMachines(self, surfaceIn=None, scale = 1, randomcolors = False):   
        #Drawing
        #Machine Positions
        if(surfaceIn is None):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH * scale, self.HEIGHT * scale)
        else:
            surface = surfaceIn
        ctx = cairo.Context(surface)
        ctx.scale(1.0 * scale, -1.0 * scale)
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
    def drawCollisions(self, surfaceIn=None, drawColors = True, scale = 1, drawWalls=True):

        #Machine Collisions
        if(surfaceIn is None):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.WIDTH * scale, self.HEIGHT * scale)
        else:
            surface = surfaceIn
        ctx = cairo.Context(surface)
        ctx.scale(1.0 * scale, -1.0 * scale)
        ctx.translate(0.0,-self.HEIGHT)
        if(surfaceIn is None):
            ctx.rectangle(0, 0, self.WIDTH, self.HEIGHT)  
            ctx.set_source_rgb(1.0, 1.0, 1.0)
            ctx.fill()

        #Drawing collisions between machines
        for polygon in self.machineCollisionList:
            for loop in polygon:
                if(drawColors):
                    ctx.set_source_rgb(1.0, 0.3, 0.0)
                else:
                    ctx.set_source_rgb(0.7, 0.7, 0.7)
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
                    if(drawColors):
                        ctx.set_source_rgb(1.0, 0.3, 0.0)
                    else:
                        ctx.set_source_rgb(0.7, 0.7, 0.7)
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
    def random_Material_Flow_File(self):
        random.seed()
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Input","Random_Materialflow.csv")

        with open(path, 'w') as file:
            file.write(f"From, To, Intensity\n")
            for _ in range(0,len(self.machine_list) * 2):
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
    def mapRange(self,s , a, b):
        (a1, a2), (b1, b2) = a, b
        if(s < a1): s = a1
        if(s > a2): s = a2
        return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))



#------------------------------------------------------------------------------------------------------------
def main():
    outputfile ="Out"

    #filename = "Overlapp"
    filename = "Basic"
    #filename = "Round_Walls"
    #filename = "EP_v23_S1_clean"
    #filename = "Simple"
    #filename = "SimpleNoCollisions"

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "..",
        "Input",
        "1",  
        filename + ".ifc")


    file_name, _ = os.path.splitext(ifcpath)
    materialflowpath = file_name + "_Materialflow.csv"
    #materialflowpath = None
    
    demoFactory = FactorySim(ifcpath,
    path_to_materialflow_file = materialflowpath,
    randomMF = True,
    randomPos = True,
    verboseOutput=3,
    maxMF_Elements = 5,
    objectScaling=1.0)
 
    #Machine Positions Output to PNG
    #machinePositions = demoFactory.drawPositions(drawMaterialflow = True, drawMachineCenter = True)
    machinePositions = demoFactory.drawPositions(scale = 1, drawMaterialflow = True, drawMachineCenter = True, drawMachineBaseOrigin=True, highlight=1)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "Output", 
        F"{outputfile}_machines.png")
    machinePositions.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")
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
    demoFactory.update(0,0.8 ,-0.2 , 1)
    demoFactory.evaluate()
    demoFactory.update(1,1 ,-1 , 1)
    demoFactory.evaluate()
    demoFactory.update(1,0.1 ,-0.8 , 1, 0.8)

    machinePositions = demoFactory.drawPositions(scale = 1, drawColors = False, drawMachines=True, drawMaterialflow = True, drawMachineCenter = True, drawMachineBaseOrigin=True, highlight=1)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "Output", 
        F"{outputfile}_machines_update.png")
    machinePositions.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")

    #Machine Collisions Output to PNG
    Collisions = demoFactory.drawCollisions(scale = 1, drawColors = False, surfaceIn=machinePositions)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "Output", 
        F"{outputfile}_machine_collsions.png")
    Collisions.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")

    toutput = demoFactory.drawPositions(drawMaterialflow = True, drawColors = False, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False, highlight=1)
    toutput = demoFactory.drawCollisions(surfaceIn = toutput, drawColors = False)
    toutput = demoFactory.drawPositions(drawMaterialflow = True, drawMachines = False, drawColors = False, drawMachineCenter = True, drawOrigin = False, drawMachineBaseOrigin=False)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "Output", 
        F"{outputfile}test.png")
    toutput.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")

    ##Rate current Layout
    demoFactory.evaluate()

    print("Total runtime: " + bold(fg256("green", bg256("yellow", round((time() - demoFactory.timezero) * 1000, 2)))))

    
if __name__ == "__main__":
    from Helpers.MFO import MFO 
    main()
else:
    from gym_factorySim.envs.Helpers.MFO import MFO
    from gym_factorySim.envs.Helpers.Point3D import Point3D