import os
import random
import math
import uuid

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import cairo
from tqdm import tqdm

from factorySim.factorySimClass import FactorySim
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import make_multi_agent

from PIL import Image
 
class FactorySimEnv(gym.Env):  
    metadata = {'render.modes': ['human', 'rgb']}

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, env_config: EnvContext):
        super()
        print(env_config)
        self.stepCount = 0
        self._obs_type = env_config["obs_type"]
        self.Loglevel = env_config["Loglevel"]
        self.uid = 0
        self.width = env_config["width"]
        self.heigth = env_config["heigth"]
        self.maxMF_Elements = env_config["maxMF_Elements"]
        self.scale = env_config["outputScale"]
        self.objectScaling = env_config["objectScaling"]
        if env_config["inputfile"] is not None:
            file_name, _ = os.path.splitext(env_config["inputfile"])
        else:
            exit("No inputfile given.")
        self.inputfile = env_config["inputfile"]
        self.materialflowpath = file_name + "_Materialflow.csv"
        self.rendermode= env_config["rendermode"]
        

        self.info = {}
        if(os.path.isdir(env_config["inputfile"])):
            self.output_path = os.path.join(os.path.dirname(os.path.realpath(env_config["inputfile"])),
            "..", 
            "Output")
        else:
            self.output_path = os.path.join(os.path.dirname(os.path.realpath(env_config["inputfile"])), 
            "..",
            "Output")

        # Actions of the format MoveX, MoveY, Rotate, (Skip) 
        #self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0]), high=np.array([1,1,1,1]), dtype=np.float32)
        #Skipping disabled
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1,1,1], dtype=np.float32))

        if self._obs_type == 'image':
            #self.observation_space = spaces.Box(low=0, high=256**4 -1, shape=(self.width *self.heigth,))
            self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(255), shape=(self.width, self.heigth, 2))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

        self.reset()

    def step(self, action):
       
        #self.factory.update(self.currentMachine, action[0], action[1], action[2], action[3])
        #print(F"Actions: 1 - {action[0]}      2 - {action[1]}       3 - {action[2]}")
        self.factory.update(self.currentMachine, action[0], action[1], action[2], 0)
        self.currentMappedReward, self.currentReward, self.info, done = self.factory.evaluate()
        #print(F"Reward: {self.currentMappedReward}")
        self.stepCount += 1
        self.currentMachine += 1
        
        if(self.currentMachine >= self.machineCount):
            self.currentMachine = 0

    
        return (self._get_obs(), self.currentMappedReward, done, self.info)
        
 
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
        self.uid +=1
        

        return self._get_obs()
 
    def render(self, mode='rgb', prefix = ""):

        output = self.factory.drawPositions(scale = self.scale, drawMaterialflow = False, drawMachines = True, drawWalls = True, drawColors = True, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False)
        output = self.factory.drawCollisions(surfaceIn = output, scale = self.scale, drawColors = True)
        output = self.factory.drawPositions(surfaceIn = output, scale = self.scale, drawMaterialflow = True, drawMachines = False, drawWalls = False, drawColors = True, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False)
        output =  self._addText(output, f"{self.uid:02d}.{self.stepCount:02d} | {self.currentMappedReward:1.2f} | {self.currentReward:1.2f} | {self.info.get('ratingMF', -100):1.2f} | {self.info.get('ratingCollision', -100):1.2f}")

        if mode == 'human' or self.rendermode == 'human':
            outputPath = os.path.join(self.output_path, f"{self.uid}_{self.stepCount:04d}.png")
            output.write_to_png(outputPath)
            return True
        elif mode == 'rgb':
            buf = output.get_data()
            #bgra to rgb
            #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0,3]]
            rgb = np.ndarray(shape=(self.width * self.scale, self.heigth * self.scale, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]
            return rgb
        elif mode == None or self.rendermode == None:
            return
        else:
            print(F"Error -  Unkonwn Render Mode: {mode}")
            return -1


    def _get_obs(self):

        #old colorimage
        #bgra to rgb
        #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]

        #new Version greyscale

        output = self.factory.drawPositions(drawMaterialflow = False, drawColors = True, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False, highlight=self.currentMachine)
        output = self.factory.drawCollisions(surfaceIn = output, drawColors = True)
        buf = output.get_data()
        machines_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]
        
        #separate Image for Materialflow
        materialflow = self.factory.drawPositions(drawMaterialflow = True, drawMachines = False, drawColors = True, drawMachineCenter = False, drawOrigin = False, drawMachineBaseOrigin=False)
        buf = materialflow.get_data()
        materialflow_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]

        out = np.concatenate((machines_greyscale, materialflow_greyscale), axis=2)
        #Normalize to Range 0-1
        out = out / 255

        return out
         


    def _addText(self, surface, text):
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1, 0, 0)
        ctx.scale(self.scale, self.scale)
        #ctx.select_font_face("Purisa", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(6)
        ctx.move_to(5, 13)
        ctx.show_text(text)
        return surface


MultiFactorySimEnv = make_multi_agent(lambda config: FactorySimEnv(config))

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
        "Input",  
        filename + ".ifc")

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "Input",
        "2")

    env_config = {
        "inputfile" : ifcpath,
        "obs_type" : 'image',
        "Loglevel" : 1,
        "width" : 84,
        "heigth" : 84,
        "maxMF_Elements" : 5,
        "outputScale" : 4,
        "objectScaling" : 1.0,
        "rendermode": "human",
            }
        
    env = FactorySimEnv( env_config = env_config)
    env.reset()
    output = None
    prefix=0
    output = env.render(mode='human', prefix=prefix)
    for _ in tqdm(range(0,10)):
        observation, reward, done, info = env.step([random.uniform(-1,1),random.uniform(-1,1), random.uniform(-1, 1), random.uniform(0, 1)])    
        output = env.render(mode='human', prefix=prefix)
        if done:
            env.reset()
            prefix+=1
            output = env.render(mode='human', prefix=prefix)
        #output = env.render(mode='rgb_array')



    #np.savetxt('data.csv', output, delimiter=',')

        
    
if __name__ == "__main__":
    main()
