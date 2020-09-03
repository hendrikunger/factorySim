import os
import random
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import cairo
from tqdm import tqdm
from gym_factorySim.envs.factorySim import FactorySim



 
class FactorySimEnv(gym.Env):  
    metadata = {'render.modes': ['human', 'rgb_array']}

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, inputfile = 'None', obs_type='image', uid=0, Loglevel=0, width = 1000, heigth = 1000, maxMF_Elements = None, outputScale = 1, objectScaling = 1.0):
        super()
        self.stepCount = 0
        self._obs_type = obs_type
        self.Loglevel = Loglevel
        self.uid = uid
        self.width = width
        self.heigth = heigth
        self.maxMF_Elements = maxMF_Elements
        self.scale = outputScale
        self.objectScaling = objectScaling 
        if inputfile is not None:
            file_name, _ = os.path.splitext(inputfile)
        else:
            exit("No inputfile given.")
        self.inputfile = inputfile
        self.materialflowpath = file_name + "_Materialflow.csv"
    
        self.factory = FactorySim(self.inputfile, 
                        path_to_materialflow_file = self.materialflowpath, 
                        width=self.width, heigth=self.heigth,
                        randomMF = True,
                        randomPos = True,
                        maxMF_Elements = self.maxMF_Elements,
                        objectScaling = self.objectScaling,
                        verboseOutput = self.Loglevel)

        self.machineCount = len(self.factory.machine_list)
        self.currentMachine = 0
        self.currentReward = 0
        self.currentMappedReward = 0
        self.info = {}
        self.output = None
        if(os.path.isdir(inputfile)):
            self.output_path = os.path.join(os.path.dirname(os.path.realpath(inputfile)),
            "..", 
            "Output")
        else:
            self.output_path = os.path.join(os.path.dirname(os.path.realpath(inputfile)), 
            "..",
            "Output")
    

        # Actions of the format MoveX, MoveY, Rotate, (Skip) 
        #self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0]), high=np.array([1,1,1,1]), dtype=np.float32)
        #Skipping disabled
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1,1,1]), dtype=np.float32)



        if self._obs_type == 'image':
            #self.observation_space = spaces.Box(low=0, high=256**4 -1, shape=(self.width *self.heigth,))
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.heigth, 2), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))
 
    def step(self, action):
       
        #self.factory.update(self.currentMachine, action[0], action[1], action[2], action[3])
        self.factory.update(self.currentMachine, action[0], action[1], action[2], 0)
        self.currentMappedReward, self.currentReward, self.info, done = self.factory.evaluate()
        self.stepCount += 1
        self.currentMachine += 1
        
        if(self.currentMachine >= self.machineCount):
            self.currentMachine = 0

    
        return self._get_obs(), self.currentMappedReward, done, self.info
        
 
    def reset(self):
        #print("\nReset")
        self.factory = FactorySim(self.inputfile,
        path_to_materialflow_file = self.materialflowpath,
        width=self.width,
        heigth=self.heigth,
        randomMF = True,
        randomPos = True,
        maxMF_Elements = self.maxMF_Elements,
        objectScaling = self.objectScaling,
        verboseOutput = self.Loglevel)

        self.machineCount = len(self.factory.machine_list)
        self.stepCount = 0
        self.currentMachine = 0
        self.currentReward = 0
        self.currentMappedReward = 0
        self.output = None
        

        return self._get_obs()
 
    def render(self, mode='human', prefix = ""):
        #print(f"Render ----{prefix}_{self.stepCount:04d}-------------------------")
        if mode == 'rgb_array':
            return self._get_np_array_render()
        elif mode == 'human':
            return self._get_image(prefix)

    def _get_obs(self):
        if self._obs_type == 'image':
            img = self._get_np_array()
        return img
         
    #Rendering for AI
    def _get_np_array(self):
        #old colorimage
        #bgra to rgb
        #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]


        #new Version greyscale

        self.output = self.factory.drawPositions(drawMaterialflow = False, drawColors = False, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False, highlight=self.currentMachine)
        self.output = self.factory.drawCollisions(surfaceIn = self.output, drawColors = False)
        buf = self.output.get_data()
        machines_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]

        #separate Image for Materialflow
        materialflow = self.factory.drawPositions(drawMaterialflow = True, drawMachines = False, drawColors = False, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False)
        buf = materialflow.get_data()
        materialflow_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]

        out = np.concatenate((machines_greyscale, materialflow_greyscale), axis=2)
        return out

    #Rendering for human as png
    def _get_image(self, prefix=None):
        outputPath = os.path.join(self.output_path, f"{prefix:02d}_{self.stepCount:04d}.png")

        #Add Materialflow
        self.output = self.factory.drawPositions(surfaceIn = self.output, drawMaterialflow = True, drawMachines = False, drawWalls = False, drawColors = False, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False)
        self.output =  self._addText(self.output, f"{self.uid:02d}.{self.stepCount:02d} | {self.currentMappedReward:1.2f} | {self.currentReward:1.2f} | {self.info.get('ratingMF', -100):1.2f} | {self.info.get('ratingCollision', -100):1.2f}")
        self.output.write_to_png(outputPath)
        buf = self.output.get_data()
        #bgra to rgb
        #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0,3]]
        rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]
        return rgb


    #Rendering for human with parallel environments
    def _get_np_array_render(self):
        self.output = self.factory.drawPositions(scale = self.scale, drawMaterialflow = True, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False, highlight=self.currentMachine)
        self.output = self.factory.drawCollisions(scale = self.scale, surfaceIn = self.output)
        buf = self._addText(self.output, f"{self.uid:02d}.{self.stepCount:02d} | {self.currentMappedReward:1.2f} | {self.currentReward:1.2f} | {self.info.get('ratingMF', -100):1.2f} | {self.info.get('ratingCollision', -100):1.2f}").get_data()

        #bgra to rgb
        #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0,3]]
        rgb = np.ndarray(shape=(self.width * self.scale, self.heigth * self.scale, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]
        return rgb

    def _addText(self, surface, text):
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1, 0, 0)
        ctx.scale(self.scale, self.scale)
        #ctx.select_font_face("Purisa", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(6)
        ctx.move_to(5, 13)
        ctx.show_text(text)
        return surface




#------------------------------------------------------------------------------------------------------------
def main():

    #filename = "Overlapp"
    #filename = "Basic"
    #filename = "EP_v23_S1_clean"
    #filename = "Simple"
    #filename = "SimpleNoCollisions"
    filename = "LShape"

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "..",
        "Input",  
        filename + ".ifc")

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "..",
        "Input",
        "2")

        
    env = FactorySimEnv(inputfile = ifcpath, obs_type='image', objectScaling=0.5, maxMF_Elements = 5, Loglevel=1)
    env.reset()
    output = None
    prefix=0
    output = env.render(mode='human', prefix=prefix)
    for _ in tqdm(range(0,500)):
        observation, reward, done, info = env.step([random.uniform(-1,1),random.uniform(-1,1), random.uniform(-1, 1), random.uniform(0, 1)])    
        output = env.render(mode='human', prefix=prefix)
        if done:
            env.reset()
            prefix+=1
            output = env.render(mode='human', prefix=prefix)
        #output = env.render(mode='rgb_array')



    #np.savetxt('data.csv', output, delimiter=',')

        
    
if __name__ == "__main__":
    from factorySim import FactorySim
    main()