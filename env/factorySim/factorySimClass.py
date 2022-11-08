#!/usr/bin/env python3

import os
import math
import random

from time import time

import numpy as np

import cairo
import pandas as pd

from factorySim.creation import FactoryCreator
import factorySim.baseConfigs as baseConfigs
from factorySim.rendering import  draw_BG, drawFactory, drawCollisions
from factorySim.kpi import FactoryRating
from factorySim.routing import FactoryPath


class FactorySim:
 #------------------------------------------------------------------------------------------------------------
 # Loading
 #------------------------------------------------------------------------------------------------------------
    def __init__(self, path_to_ifc_file=None, path_to_materialflow_file = None, factoryConfig=baseConfigs.SMALLSQUARE, randseed = None, randomPos = False, createMachines = False, verboseOutput = 0, maxMF_Elements = None):
        self.FACTORYDIMENSIONS = (factoryConfig.WIDTH, factoryConfig.HEIGHT) # if something is read from file this is overwritten
        self.MAXMF_ELEMENTS = maxMF_Elements
        self.factoryCreator = FactoryCreator(self.FACTORYDIMENSIONS,
            factoryConfig.MAXSHAPEWIDTH,
            factoryConfig.MAXSHAPEHEIGHT, 
            math.floor(0.8*self.MAXMF_ELEMENTS) if maxMF_Elements else factoryConfig.AMOUNTRECT, 
            math.ceil(0.2*self.MAXMF_ELEMENTS) if maxMF_Elements else factoryConfig.AMOUNTPOLY, 
            factoryConfig.MAXCORNERS
            )
        self.verboseOutput = verboseOutput
        self.RANDSEED = randseed
        random.seed(randseed)
            
        self.timezero = time()
        self.lasttime = 0        
        self.RatingDict = {}

        #Importing Walls
        if path_to_ifc_file:
            if(os.path.isdir(path_to_ifc_file)):
                
                self.ifc_file = random.choice([x for x in os.listdir(path_to_ifc_file) if ".ifc" in x and "LIB" not in x])
                self.ifc_file = os.path.join(path_to_ifc_file, self.ifc_file)
            else:
                self.ifc_file = path_to_ifc_file

            if(self.verboseOutput >= 2):
                print("Lade: ", self.ifc_file)
                
            self.wall_dict = self.factoryCreator.load_ifc_factory(self.ifc_file, "IFCWALL", recalculate_bb=True)

        else:
            self.wall_dict = {}

        if(self.verboseOutput >= 1):
            self.printTime("IFCWALL geparsed")

        #Create Random Machines
        if createMachines:
            self.machine_dict = self.factoryCreator.create_factory()
        #Import Machines from IFC File
        else:
            #Import up to MAXMF_ELEMENTS from File
            if(self.MAXMF_ELEMENTS):
                path = os.path.join(os.path.dirname(self.ifc_file), "MFO_LIB.ifc")
                if(os.path.exists(path)):
                    mfo_ifc_file_path = path
                else:
                    mfo_ifc_file_path = self.ifc_file

                if(self.verboseOutput >= 2):
                    print(f"Lade: Demomaterialflussobjekte. Maximal {self.MAXMF_ELEMENTS} werden aus {mfo_ifc_file_path} geladen.")
                #2 bis MAXMF_ELEMENTS aus der Datei mit Demomaterialflussobjekten laden.
                self.machine_dict = self.factoryCreator.load_ifc_factory(mfo_ifc_file_path, "IFCBUILDINGELEMENTPROXY", randomMF=self.MAXMF_ELEMENTS)
            else:
                #Import full file
                if(self.verboseOutput >= 2):
                    print("Nutze alle MF Objekte in der IFC Datei")
                self.machine_dict = self.factoryCreator.load_ifc_factory(self.ifc_file, "IFCBUILDINGELEMENTPROXY")


            if(self.verboseOutput >= 3):
                self.printTime("Datei geladen")


        if(self.verboseOutput >= 1):
            self.printTime("IFCBUILDINGELEMENTPROXY geparsed")

        self.factoryPath=FactoryPath(factoryConfig.BOUNDARYSPACING, 
            factoryConfig.MINDEADENDLENGTH,
            factoryConfig.MINPATHWIDTH,
            factoryConfig.MINTWOWAYPATHWIDTH,
            factoryConfig.SIMPLIFICATIONANGLE)

        self.machineCollisionList = []
        self.wallCollisionList = []

        self.lastRating = 0
        self.currentRating    = 0 # Holds the Rating of the current state of the Layout 
        self.currentMappedRating    = 0 # Holds the normalized Rating of the current state of the Layout 

        self.lastUpdatedMachine = None #Hold uid of last updated machine for collision checking
        self.collisionAfterLastUpdate = False # True if latest update leads to new collsions

        self.episodeCounter = 0
        self.scale = 1 #Saves the scaling factor of provided factories for external access

        #Creating random positions
        if randomPos:
            for key in self.machine_dict:
                self.update(key,
                    xPosition = random.uniform(-1,1),
                    yPosition = random.uniform(-1,1),
                    rotation = random.uniform(-1,1),
                    massUpdate = True)

        #Find Collisions
        self.findCollisions()
        
        #Import Materialflow from Excel
        if path_to_materialflow_file:
            self.dfMF = pd.read_csv(path_to_materialflow_file, skipinitialspace=True, encoding= "utf-8")
            #Rename Colums
            indexes = self.dfMF.columns.tolist()
            self.dfMF.rename(columns={indexes[0]:'from', indexes[1]:'to', indexes[2]:'intensity'}, inplace=True)
        else:
            #Create Random Materialflow
            names = []
            for start in self.machine_dict.values():
                sample = random.choice(list(self.machine_dict.values()))
                names.append([start.name, sample.name]) 
                if random.random() >= 0.9:
                    sample = random.choice(list(self.machine_dict.values()))
                    names.append([start.name, sample.name])                 
            self.dfMF = pd.DataFrame(data=names, columns=["from", "to"])
            self.dfMF['intensity'] = np.random.randint(1,100, size=len(self.dfMF))

        #Group by from and two, add up intensity of all duplicates in intensity_sum
        self.dfMF['intensity_sum'] = self.dfMF.groupby(by=['from', 'to'])['intensity'].transform('sum')
        #drop the duplicates and refresh index
        self.dfMF = self.dfMF.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)
        #normalise intensity sum 
        self.dfMF['intensity_sum_norm'] = self.dfMF['intensity_sum'] / self.dfMF.max()["intensity_sum"]
        #use machine index as sink and source for materialflow
        #Replace Machine Names in Material flow (From Sketchup Import) with machine dict key
        machine_dict = {machine.name: key for key, machine in self.machine_dict.items()}
        self.dfMF[['from','to']] = self.dfMF[['from','to']].replace(machine_dict)
        #set initial values for costs
        self.dfMF['costs'] = 0

    
        if(self.verboseOutput >= 3):
            self.printTime("Materialfluss geladen")


        
 #------------------------------------------------------------------------------------------------------------
 # Update Machines
 #------------------------------------------------------------------------------------------------------------
    def update(self, machineIndex, xPosition = 0, yPosition = 0, rotation = None, skip = 0, massUpdate = False):
        if type(machineIndex) == int:
            if machineIndex< len(self.machine_dict):
                machineIndex = list(self.machine_dict)[machineIndex]
            else:
                print("Machine Index not found")
                return

        if not massUpdate: self.episodeCounter += 1
        if(skip < 0.8):
            self.lastUpdatedMachine = self.machine_dict[machineIndex].gid

            if(self.verboseOutput >= 2):
                print(f"Update: {self.machine_dict[machineIndex].name} - X: {xPosition:1.1f} Y: {yPosition:1.1f} R: {rotation:1.2f} ")

            if (rotation is not None):
                mappedRot = self.mapRange(np.clip(rotation,-1.0, 1.0), (-1,1), (0, 2*math.pi))
                self.machine_dict[machineIndex].rotate_Item(mappedRot)

            bbox = self.factoryCreator.bb.bounds #bbox is a tuple of (xmin, ymin, xmax, ymax)
            #Max Value should move machine to the rightmost or topmost position without moving out of the image
            mappedXPos = self.mapRange(np.clip(xPosition,-1.0, 1.0), (-1,1), (0,bbox[2] - self.machine_dict[machineIndex].width))
            mappedYPos = self.mapRange(np.clip(yPosition,-1.0, 1.0), (-1,1), (0,bbox[3] - self.machine_dict[machineIndex].height))
            #mappedXPos = self.mapRange(xPosition, (-1,1), (0,bbox[2]))
            #mappedYPos = self.mapRange(yPosition, (-1,1), (0,bbox[3]))

            self.machine_dict[machineIndex].translate_Item(mappedXPos, mappedYPos)
            if not massUpdate:
                self.findCollisions()
            if(self.verboseOutput >= 3):
                self.printTime(f"{self.machine_dict[machineIndex].name} geupdated")
        else:
             if(self.verboseOutput >= 2):
                print(f"Update: {self.machine_dict[machineIndex].name} - Skipped Update")


    
 #------------------------------------------------------------------------------------------------------------
 # Evaluation
 #------------------------------------------------------------------------------------------------------------
    def evaluate(self):
        self.RatingDict["ratingMF"] = self.evaluateMF()          
        if(self.verboseOutput >= 3):
            self.printTime("Bewertung des Materialfluss abgeschlossen")
        self.RatingDict["ratingCollision"] = self.evaluateCollision()          
        if(self.verboseOutput >= 3):
            self.printTime("Kollisionsbewertung abgeschlossen")
        self.fullPathGraph, self.reducedPathGraph = self.factoryPath.calculateAll(self.machine_dict, self.factoryCreator.bb)
        if(self.verboseOutput >= 3):
            self.printTime("Pfadbewertung abgeschlossen")


       
        #if(self.episodeCounter < len(self.machine_list)):
        #    self.currentRating = 0
        #el

        if(self.RatingDict["ratingCollision"] == 1):
            self.currentRating = math.pow(self.RatingDict["ratingMF"],3)
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
        self.currentMappedRating = self.RatingDict["TotalRating"]= self.currentRating

        #print(f"walls: {len(self.wallCollisionList)}, machines: {len(self.machineCollisionList)}, count m: {len(self.machine_list)}")
        #if(len(self.wallCollisionList) + len(self.machineCollisionList) >=len(self.machine_list)):
        #    done = True
        #else:
        #    done = False


        #if(self.episodeCounter >= 3 * len(self.machine_list)):
        if(self.episodeCounter >= len(self.machine_dict)+1):
            done = True
            self.RatingDict["done"] = True
        else:
            done = False     
        #done = False      


        if(self.verboseOutput >= 3):
            self.printTime("Bewertung des Layouts abgeschlossen")
        if(self.verboseOutput >= 1):
            print(f"Total Rating: {self.currentMappedRating:1.2f}\n"
                f"Raw Rating: {self.currentRating:1.2f}\n"
                f"MaterialFlow: {self.RatingDict['ratingMF']:1.2f}\n"
                f"Kollisionen: {self.RatingDict['ratingCollision']:1.2f}")

        return self.currentMappedRating, self.currentRating, self.RatingDict, done

    def generateRatingText(self):
        return f"Reward: {self.RatingDict['TotalRating']:1.2f} |  MF: {self.RatingDict['ratingMF']:1.2f}  |  COLL: {self.RatingDict['ratingCollision']:1.2f}"

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF_Helper(self, source, sink): 
        source_center = self.machine_dict[source].center
        sink_center = self.machine_dict[sink].center
        return math.sqrt(math.pow(source_center.x-sink_center.x,2) + math.pow(source_center.y-sink_center.y,2))

 #------------------------------------------------------------------------------------------------------------
    def evaluateMF(self):
        self.dfMF['distance'] = self.dfMF.apply(lambda row: self.evaluateMF_Helper(row['from'], row['to']), axis=1)
        #sum of all costs /  maximum intensity (intensity sum norm * 1) 
        #find longest distance possible in factory
        maxDistance = max(self.factoryCreator.bb.bounds[2],  self.factoryCreator.bb.bounds[3])
        self.dfMF['distance_norm'] = self.dfMF['distance'] / maxDistance
        self.dfMF['costs'] = self.dfMF['distance_norm'] * self.dfMF['intensity_sum_norm']
        output = 1 - (math.pow(self.dfMF['costs'].sum(),2) / self.dfMF['intensity_sum_norm'].sum())
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
        #for machine in self.machine_list.values():      
        #    totalMachineArea += machine.hull.area()

        #print(len(list(combinations(self.machine_list.values(), 2))))
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
        factoryRating = FactoryRating(self.machine_dict, self.wall_dict)
        self.collisionAfterLastUpdate = factoryRating.findCollisions(self.lastUpdatedMachine)
        self.machineCollisionList = factoryRating.machineCollisionList
        self.wallCollisionList = factoryRating.wallCollisionList                    
        if(self.verboseOutput >= 3):
            self.printTime("Kollisionen berechnen abgeschlossen")

 #------------------------------------------------------------------------------------------------------------
 # Drawing
 #------------------------------------------------------------------------------------------------------------
    def provideCairoDrawingData(self, width, height):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        self.scale = self.factoryCreator.suggest_factory_view_scale(width, height)
        ctx.scale(self.scale, self.scale)
        return surface, ctx


    
#------------------------------------------------------------------------------------------------------------  
    def drawDetailedMachines(self, surfaceIn=None, scale = 1, randomcolors = False, highlight = None):   
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
        for i, machine in enumerate(self.machine_dict.values()):
            for poly in machine.poly.geoms:
                if (((i == highlight) or machine.gid==highlight) and (highlight is not None)):
                    ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                elif randomcolors:
                        ctx.set_source_rgb(random.random(), random.random(), random.random())
                else:
                    ctx.set_source_rgb(0.4, 0.4, 0.4)
                #draw all outer contours
                for point in poly.exterior.coords:  
                    ctx.line_to(point[0], point[1])
                ctx.close_path()
                ctx.fill()
                #draw all holes
                if randomcolors:
                    ctx.set_source_rgb(random.random(), random.random(), random.random())
                else:
                    ctx.set_source_rgb(machine.color[0], machine.color[1], machine.color[2])
                for loop in poly.interiors:
                    for point in loop.coords:
                        ctx.line_to(point[0], point[1])
                    ctx.close_path()
                ctx.fill()

                
        if(self.verboseOutput >= 3):
            self.printTime("Detailierte Machinenpositionen gezeichnet")
        return surface



 #------------------------------------------------------------------------------------------------------------
 # Helpers  
 #------------------------------------------------------------------------------------------------------------
    
    def printTime(self, text):
        number = (time() - self.timezero - self.lasttime)
        self.lasttime += number
        number = round(number * 1000, 2)
        print(f"{number:6.2f} - {text}")

  #------------------------------------------------------------------------------------------------------------  
    def mapRange(self,s , a, b):
        (a1, a2), (b1, b2) = a, b
        if(s < a1): s = a1
        if(s > a2): s = a2
        return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))

#------------------------------------------------------------------------------------------------------------
def main():

    img_resolution = (500, 500)
    outputfile ="Out"

    #filename = "Long"
    #filename = "Basic"
    filename = "Simple"
    #filename = "SimpleNoCollisions"

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "Input",
        "2",  
        filename + ".ifc")


    file_name, _ = os.path.splitext(ifcpath)
    #materialflowpath = file_name + "_Materialflow.csv"
    materialflowpath = None
    demoFactory = FactorySim(ifcpath,
        path_to_materialflow_file = materialflowpath,
        factoryConfig=baseConfigs.SMALLSQUARE,
        randomPos=False,
        createMachines=True,
        verboseOutput=4,
        maxMF_Elements = None)
    
    surface, ctx = demoFactory.provideCairoDrawingData(*img_resolution)
    #Machine Positions Output to PNG
    draw_BG(ctx, *img_resolution)
    drawFactory(ctx, demoFactory.machine_dict, demoFactory.wall_dict, demoFactory.dfMF, drawNames=False, drawColors = True, drawOrigin = True, drawMachineCenter = True, highlight=0)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "Output", 
        F"{outputfile}_machines.png")
    surface.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")
    
 #------------------------------------------------------------------------------------------------------------------------------------------
    #detailed Machines Output to PNG
    # detailedMachines = demoFactory.drawDetailedMachines(surfaceIn=machinePositions, randomcolors = False, highlight=None)
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #     "..",
    #     "..",
    #    "Output", 
    #    F"{outputfile}_detailed_machines.png")
    # detailedMachines.write_to_png(path)
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

    draw_BG(ctx, *img_resolution)
    drawFactory(ctx, demoFactory.machine_dict, demoFactory.wall_dict, demoFactory.dfMF, drawNames=False, drawOrigin = True, drawMachineCenter = True, highlight=0)
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "Output", 
        F"{outputfile}_machines_update.png")
    surface.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")
    
    #Machine Collisions Output to PNG
    drawCollisions(ctx, demoFactory.machineCollisionList, demoFactory.wallCollisionList)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "Output", 
        F"{outputfile}_machine_collsions.png")
    surface.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")

    
    ##Rate current Layout
    demoFactory.evaluate()

    print(f"Total runtime: {round((time() - demoFactory.timezero) * 1000, 2)}")

    
if __name__ == "__main__":
    main()

    