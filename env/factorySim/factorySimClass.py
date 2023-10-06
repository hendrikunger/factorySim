#!/usr/bin/env python3

import os
import random
import logging

from time import time

import numpy as np
import cairo
import pandas as pd

from factorySim.creation import FactoryCreator
import factorySim.baseConfigs as baseConfigs
from factorySim.rendering import  draw_BG, drawFactory, drawCollisions
from factorySim.kpi import FactoryRating
from factorySim.routing import FactoryPath
from shapely.ops import unary_union, snap
from shapely.geometry import MultiPolygon, Polygon

class FactorySim:
 #------------------------------------------------------------------------------------------------------------
 # Loading
 #------------------------------------------------------------------------------------------------------------
    def __init__(self, path_to_ifc_file=None, path_to_materialflow_file = None, factoryConfig=baseConfigs.SMALLSQUARE, randseed = None, randomPos = False, createMachines = False, verboseOutput = 0, maxMF_Elements = None):
        self.FACTORYDIMENSIONS = (factoryConfig.WIDTH, factoryConfig.HEIGHT) # if something is read from file this is overwritten
        self.DRAWINGORIGIN = (0,0)
        self.MAXMF_ELEMENTS = maxMF_Elements
        self.factoryCreator = FactoryCreator(self.FACTORYDIMENSIONS,
            factoryConfig.MAXSHAPEWIDTH,
            factoryConfig.MAXSHAPEHEIGHT,
            int(np.floor(0.8*self.MAXMF_ELEMENTS)) if maxMF_Elements else factoryConfig.AMOUNTRECT, 
            int(np.ceil(0.2*self.MAXMF_ELEMENTS)) if maxMF_Elements else factoryConfig.AMOUNTPOLY, 
            factoryConfig.MAXCORNERS
            )
        self.verboseOutput = verboseOutput
        self.RANDSEED = randseed
        self.pathPolygon = None
        self.MFIntersectionPoints = None
        random.seed(randseed)
            
        self.timezero = time()
        self.lasttime = 0        
        self.RatingDict = {}
        self.MachinesFarFromPath = set()
        self.machine_dict = None
        self.wall_dict = None
        self.machineCollisionList = []
        self.wallCollisionList = []
        self.outsiderList = []

        #Importing Walls
        if path_to_ifc_file:
            if(os.path.isdir(path_to_ifc_file)):
                
                self.ifc_file = random.choice([x for x in os.listdir(path_to_ifc_file) if ".ifc" in x and "LIB" not in x])
                self.ifc_file = os.path.join(path_to_ifc_file, self.ifc_file)
            else:
                self.ifc_file = path_to_ifc_file

            logging.info(f"Lade: {self.ifc_file}")
                
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

                if(self.verboseOutput >= 2):
                    print(f"Lade: Demomaterialflussobjekte. Maximal {self.MAXMF_ELEMENTS} werden aus {self.ifc_file} geladen.")
                #2 bis MAXMF_ELEMENTS aus der Datei mit Demomaterialflussobjekten laden.
                self.machine_dict = self.factoryCreator.load_ifc_factory(self.ifc_file, "IFCBUILDINGELEMENTPROXY", randomMF=self.MAXMF_ELEMENTS)
            else:
                #Import full file
                if(self.verboseOutput >= 2):
                    print("Nutze alle MF Objekte in der IFC Datei")
                self.machine_dict = self.factoryCreator.load_ifc_factory(self.ifc_file, "IFCBUILDINGELEMENTPROXY")


            if(self.verboseOutput >= 3):
                self.printTime("Datei geladen")


        if(self.verboseOutput >= 1):
            self.printTime("IFCBUILDINGELEMENTPROXY geparsed")

        #Update Dimensions after Loading
        self.FACTORYDIMENSIONS = (self.factoryCreator.factoryWidth, self.factoryCreator.factoryHeight)
        self.DRAWINGORIGIN = (self.factoryCreator.bb.bounds[0], self.factoryCreator.bb.bounds[1])

        self.factoryPath=FactoryPath(factoryConfig.BOUNDARYSPACING, 
            factoryConfig.MINDEADENDLENGTH,
            factoryConfig.MINPATHWIDTH,
            factoryConfig.MAXPATHWIDTH,
            factoryConfig.MINTWOWAYPATHWIDTH,
            factoryConfig.SIMPLIFICATIONANGLE)


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
                    rotation = random.uniform(-1,1))
        
        #Import Materialflow from Excel
        if path_to_materialflow_file:
            self.dfMF = pd.read_csv(path_to_materialflow_file, skipinitialspace=True, encoding= "utf-8")
            #Rename Colums
            indexes = self.dfMF.columns.tolist()
            self.dfMF.rename(columns={indexes[0]:'source', indexes[1]:'target', indexes[2]:'intensity'}, inplace=True)
        else:
            #Create Random Materialflow
            self.dfMF = self.factoryCreator.createRandomMaterialFlow()
    
        self.cleanMaterialFLow()
    
        if(self.verboseOutput >= 3):
            self.printTime("Materialfluss geladen")
 #------------------------------------------------------------------------------------------------------------
 # Update Materialflow
 #------------------------------------------------------------------------------------------------------------
    def addMaterialFlow(self, fromMachine, toMachine, intensity):
        #Add new Materialflow
        newDF = pd.DataFrame({'source': [fromMachine], 'target': [toMachine], 'intensity': [intensity]})
        self.dfMF = pd.concat([self.dfMF, newDF], ignore_index=True)
        self.cleanMaterialFLow()

    def cleanMaterialFLow(self):
        #Group by from and two, add up intensity of all duplicates in intensity_sum
        self.dfMF['intensity_sum'] = self.dfMF.groupby(by=['source', 'target'])['intensity'].transform('sum')
        #drop the duplicates and refresh index
        self.dfMF = self.dfMF.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
        #normalise intensity sum 
        self.dfMF['intensity_sum_norm'] = self.dfMF['intensity_sum'] / self.dfMF.max()["intensity_sum"]
        #use machine index as sink and source for materialflow
        #Replace Machine Names in Material flow (From Sketchup Import) with machine dict key
        machine_dict = {machine.name: key for key, machine in self.machine_dict.items()}
        self.dfMF[['source','target']] = self.dfMF[['source','target']].replace(machine_dict)
        #set initial values for costs
        self.dfMF['costs'] = 0
        self.dfMF['trueCosts'] = 0  # using the true distances
        
 #------------------------------------------------------------------------------------------------------------
 # Update Machines
 #------------------------------------------------------------------------------------------------------------
    def update(self, machineIndex, xPosition = 0, yPosition = 0, rotation = None, skip = 0):
        if type(machineIndex) == int:
            if machineIndex< len(self.machine_dict):
                machineIndex = list(self.machine_dict)[machineIndex]
            else:
                print("Machine Index not found")
                return

        self.episodeCounter += 1
        if(skip < 0.8):
            self.lastUpdatedMachine = self.machine_dict[machineIndex].gid

            if(self.verboseOutput >= 2):
                print(f"Update: {self.machine_dict[machineIndex].name} - X: {xPosition:1.1f} Y: {yPosition:1.1f} R: {rotation:1.2f} ")

            if (rotation is not None):
                mappedRot = np.interp(rotation, (-1.0, 1.0), (0, 2*np.pi))
                self.machine_dict[machineIndex].rotate_Item(mappedRot)

            bbox = self.factoryCreator.bb.bounds #bbox is a tuple of (xmin, ymin, xmax, ymax)
            #Max Value should move machine to the rightmost or topmost position without moving out of the image
            #np.interp also Clips Position to Output Range
                    
            mappedXPos = np.interp(xPosition, (-1.0, 1.0), (0, bbox[2] - self.machine_dict[machineIndex].width))  
            mappedYPos = np.interp(yPosition, (-1.0, 1.0), (0, bbox[3] - self.machine_dict[machineIndex].height)) 


            self.machine_dict[machineIndex].translate_Item(mappedXPos, mappedYPos)

            if(self.verboseOutput >= 3):
                self.printTime(f"{self.machine_dict[machineIndex].name} geupdated")
        else:
             if(self.verboseOutput >= 2):
                print(f"Update: {self.machine_dict[machineIndex].name} - Skipped Update")


    
 #------------------------------------------------------------------------------------------------------------
 # Evaluation
 #------------------------------------------------------------------------------------------------------------
    def evaluate(self):

        self.RatingDict = {}
        
        self.fullPathGraph, self.reducedPathGraph, self.walkableArea = self.factoryPath.calculateAll(self.machine_dict, self.wall_dict, self.factoryCreator.bb)
        if self.fullPathGraph and self.reducedPathGraph and self.walkableArea is not None:
            self.dfMF = self.factoryPath.calculateRoutes(self.dfMF)
            

            if(self.verboseOutput >= 3):
                self.printTime("Pfadbewertung abgeschlossen")

            self.factoryRating = FactoryRating(machine_dict=self.machine_dict, wall_dict=self.wall_dict, fullPathGraph=self.fullPathGraph, reducedPathGraph=self.reducedPathGraph, prepped_bb=self.factoryCreator.prep_bb, dfMF=self.dfMF)

            self.RatingDict["ratingCollision"] = self.evaluateCollision()          
            if(self.verboseOutput >= 3):
                self.printTime("Kollisionsbewertung abgeschlossen")

            self.RatingDict["ratingMF"] = self.factoryRating.evaluateMF(self.factoryCreator.bb) 
            self.RatingDict["ratingTrueMF"] = self.factoryRating.evaluateTrueMF(self.factoryCreator.bb)
            #sort MF Dict for Rendering
            self.dfMF.sort_values(by=['intensity_sum_norm'], inplace=True, ascending=False) 
            self.RatingDict["MFIntersection"], self.MFIntersectionPoints = self.factoryRating.evaluateMFIntersection()        
            if(self.verboseOutput >= 3):
                self.printTime("Bewertung des Materialfluss abgeschlossen")

            self.pathPolygon, self.extendedPathPolygon = self.factoryRating.PathPolygon()
            self.MachinesFarFromPath = self.factoryRating.getMachinesFarFromPath(self.extendedPathPolygon)
            self.RatingDict["routeAccess"] = self.factoryRating.evaluateRouteAccess(self.MachinesFarFromPath)
            self.RatingDict["pathEfficiency"] = self.factoryRating.PathEfficiency()
            #20 % of the maximum dimension of the factory as grouping threshold
            self.usedSpacePolygonDict, self.machine_dict = self.factoryRating.UsedSpacePolygon(max(self.FACTORYDIMENSIONS) * 0.2)
            self.freeSpacePolygon, self.growingSpacePolygon = self.factoryRating.FreeSpacePolygon(self.pathPolygon, self.walkableArea, self.usedSpacePolygonDict)
            self.RatingDict["areaUtilisation"] = self.factoryRating.evaluateAreaUtilisation(self.walkableArea, self.freeSpacePolygon)
            self.RatingDict["Scalability"] = self.factoryRating.evaluateScalability(self.growingSpacePolygon)
            #Currently not used, does not make relevant difference
            self.factoryRating.evaluateCompactness(self.usedSpacePolygonDict)
            self.freespaceAlongRoutesPolygon = self.factoryRating.FreeSpaceRoutesPolygon(self.pathPolygon)
            self.RatingDict["routeContinuity"] = self.factoryRating.evaluateRouteContinuity()
            self.RatingDict["routeWidthVariance"] =self.factoryRating.PathWidthVariance()
            self.RatingDict["Deadends"] =self.factoryRating.evaluateDeadends()
            




    ## Total Rating Calculation 

        partialRatings = np.array([v for k, v in self.RatingDict.items() if k != "TotalRating" and k != "terminated" and k != "ratingCollision"])
        weights = np.ones_like(partialRatings)

        if(self.RatingDict.get("ratingCollision", -1) == 1):
            self.currentRating = np.average(partialRatings, weights=weights)
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




        #if(self.episodeCounter >= 3 * len(self.machine_list)):
        if(self.episodeCounter >= len(self.machine_dict)+1):
            self.RatingDict["terminated"] = True
        else:
            self.RatingDict["terminated"] = False   
        #done = False      


        if(self.verboseOutput >= 3):
            self.printTime("Bewertung des Layouts abgeschlossen")
        if(self.verboseOutput >= 1):
            print(self.generateRatingText(multiline=True))

        return self.currentMappedRating, self.currentRating, self.RatingDict, self.RatingDict["terminated"]

    def generateRatingText(self, multiline=False):
        if(multiline):
            con = "\n"
        else:
            con = " | "
        return (f"REWARD              : {self.RatingDict.get('TotalRating', 0): 1.2f}{con}"
                f"Material Flow       : {self.RatingDict.get('ratingMF', 0): 1.2f}{con}"
                f"True Material Flow  : {self.RatingDict.get('ratingTrueMF', 0): 1.2f}{con}"
                f"MF Intersections    : {self.RatingDict.get('MFIntersection', 0): 1.2f}{con}"
                f"Collisions          : {self.RatingDict.get('ratingCollision', 0): 1.2f}{con}"
                f"Route Continuity    : {self.RatingDict.get('routeContinuity', 0): 1.2f}{con}"
                f"Route Width Variance: {self.RatingDict.get('routeWidthVariance', 0): 1.2f}{con}"
                f"Dead Ends           : {self.RatingDict.get('Deadends', 0): 1.2f}{con}"
                f"Route Access        : {self.RatingDict.get('routeAccess', 0): 1.2f}{con}"
                f"Path Efficiency     : {self.RatingDict.get('pathEfficiency', 0): 1.2f}{con}"
                f"Area Utilization    : {self.RatingDict.get('areaUtilisation', 0): 1.2f}{con}"
                f"Scalability         : {self.RatingDict.get('Scalability', 0): 1.2f}{con}"

                )

 #------------------------------------------------------------------------------------------------------------
    def evaluateCollision(self):
        
        self.collisionAfterLastUpdate = self.factoryRating.findCollisions(self.lastUpdatedMachine)                 
        if(self.verboseOutput >= 3):
            self.printTime("Kollisionen berechnen abgeschlossen")      

        self.machineCollisionList = self.factoryRating.machineCollisionList
        self.wallCollisionList = self.factoryRating.wallCollisionList
        self.outsiderList = self.factoryRating.outsiderList

        #print(len(list(combinations(self.machine_list.values(), 2))))
        nMachineCollisions = len(self.factoryRating.machineCollisionList)
        nWallCollosions = len(self.factoryRating.wallCollisionList)
        nOutsiders = len(self.factoryRating.outsiderList)
        


        #If latest update leads to collision give worst rating.
        #if(self.collisionAfterLastUpdate):
        #    output = -3
        #else:
        #    output = 1 - (0.5 * nMachineCollisions) - (0.5 * nWallCollosions)
    
        output = 1 - (0.5 * nMachineCollisions) - (0.5 * nWallCollosions) - (0.5 * nOutsiders)
        
        return output



 #------------------------------------------------------------------------------------------------------------
 # Drawing
 #------------------------------------------------------------------------------------------------------------
    def provideCairoDrawingData(self, width, height, scale=None):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        if scale:
            self.scale = scale
        else:
            self.scale = self.factoryCreator.suggest_factory_view_scale(width, height)

        ctx.scale(self.scale, self.scale)
        ctx.translate(-self.factoryCreator.bb.bounds[0], -self.factoryCreator.bb.bounds[1])
        
        
        return surface, ctx


 #------------------------------------------------------------------------------------------------------------
 # Helpers  
 #------------------------------------------------------------------------------------------------------------
    
    def printTime(self, text):
        number = (time() - self.timezero - self.lasttime)
        self.lasttime += number
        number = round(number * 1000, 2)
        print(f"{number:6.2f} - {text}")


#------------------------------------------------------------------------------------------------------------
def main():

    img_resolution = (500, 500)
    outputfile ="Out"

    filename = "Long"
    #filename = "Basic"
    #filename = "Simple"
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
        factoryConfig=baseConfigs.SMALL,
        randomPos=False,
        createMachines=True,
        verboseOutput=4,
        maxMF_Elements = 3)
    
    surface, ctx = demoFactory.provideCairoDrawingData(*img_resolution)
    #Machine Positions Output to PNG
    draw_BG(ctx, demoFactory.DRAWINGORIGIN, *img_resolution)
    drawFactory(ctx, demoFactory.machine_dict, demoFactory.wall_dict, demoFactory.dfMF, drawNames=False, drawOrigin = True, drawMachineCenter = True, highlight=0)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "Output", 
        F"{outputfile}_machines.png")
    surface.write_to_png(path) 
    demoFactory.printTime("PNG schreiben")
    


    #Rate current Layout
    demoFactory.evaluate()

    #Change machine
    #demoFactory.update(0,demoFactory.machine_list[0].origin.x,demoFactory.machine_list[0].origin.y, np.pi/2)
    #demoFactory.update(0,0.8 ,-0.2 , 1)
    #demoFactory.evaluate()
    #demoFactory.update(1,0.1 ,-0.8 , 1, 0.8)
    #demoFactory.evaluate()
    demoFactory.update(1,-1 ,-1 , 0.2)
    ##Rate current Layout
    demoFactory.evaluate()

    draw_BG(ctx, demoFactory.DRAWINGORIGIN, *img_resolution)
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

    


    print(f"Total runtime: {round((time() - demoFactory.timezero) * 1000, 2)}")

    
if __name__ == "__main__":
    main()

    