import os
import random


import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import cairo
from tqdm import tqdm

from factorySim.factorySimClass import FactorySim
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import make_multi_agent


import factorySim.baseConfigs as baseConfigs
from factorySim.rendering import  draw_BG, drawFactory, drawCollisions, draw_detail_paths, draw_text_topleft, drawMaterialFlow


 
class FactorySimEnv(gym.Env):  
    metadata = {'render.modes': ['human', 'rgb_array']}

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, env_config: EnvContext):
        super()
        print(env_config)
        self.factory = None
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
        self.materialflowpath = None #file_name + "_Materialflow.csv"
        self.rendermode= env_config["rendermode"]
        

        self.info = {}
        if(os.path.isdir(env_config["inputfile"])):
            self.output_path = os.path.join(os.path.dirname(os.path.realpath(env_config["inputfile"])),
            "..", 
            "Output")
        else:
            self.output_path = os.path.join(os.path.dirname(os.path.realpath(env_config["inputfile"])), 
            "..",
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
        del(self.factory)
        self.factory = FactorySim(self.inputfile,
        path_to_materialflow_file = self.materialflowpath,
        factoryConfig=baseConfigs.SMALLSQUARE,
        randomPos=False,
        createMachines=True,
        verboseOutput=self.Loglevel,
        maxMF_Elements = self.maxMF_Elements)
        
        self.surface, self.ctx = self.factory.provideCairoDrawingData(self.width, self.heigth)
        self.rsurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width * self.scale, self.heigth*self.scale)
        self.rctx = cairo.Context(self.rsurface)

        self.rctx.scale(self.scale*self.factory.scale, self.scale*self.factory.scale)

        self.machineCount = len(self.factory.machine_dict)
        self.stepCount = 0
        self.currentMachine = 0
        self.currentReward = 0
        self.currentMappedReward = 0
        self.uid +=1

        self.factory.evaluate()
        return self._get_obs()

    def render(self, mode='rgb_array', prefix = ""):
        draw_BG(self.rctx, self.width, self.heigth, darkmode=False)
        drawFactory(self.rctx, self.factory.machine_dict, self.factory.wall_dict, None, drawNames=False, highlight=self.currentMachine)
        #draw_detail_paths(self.ctx, self.factory.fullPathGraph, self.factory.ReducedPathGraph)
        drawCollisions(self.rctx, self.factory.machineCollisionList, self.factory.wallCollisionList)
        drawMaterialFlow(self.rctx, self.factory.machine_dict, self.factory.dfMF, drawColors=True)
        draw_text_topleft(self.rctx, f"{self.uid:02d}.{self.stepCount:02d}       {self.currentMappedReward:1.2f} | {self.currentReward:1.2f} | {self.info.get('ratingMF', -100):1.2f} | {self.info.get('ratingCollision', -100):1.2f}",(1,0,0))
        
        if mode == 'human' or self.rendermode == 'human':
            outputPath = os.path.join(self.output_path, f"{prefix}_{self.uid}_{self.stepCount:04d}.png")
            self.rsurface.write_to_png(outputPath)
            return True
        elif mode == 'rgb_array':
            buf = self.rsurface.get_data()
            #bgra to rgb
            #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0,3]]
            rgb = np.ndarray(shape=(self.width * self.scale, self.heigth * self.scale, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]
            return rgb
        elif mode == None or self.rendermode == None:
            return
        else:
            print(F"Error -  Unkown Render Mode: {mode}")
            return -1


    def _get_obs(self):

        #old colorimage
        #bgra to rgb
        #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]

        #new Version greyscale

        draw_BG(self.ctx, self.width, self.heigth, darkmode=False)
        drawFactory(self.ctx, self.factory.machine_dict, self.factory.wall_dict, None, drawColors = False, drawNames=False, highlight=self.currentMachine)
        drawCollisions(self.ctx, self.factory.machineCollisionList, self.factory.wallCollisionList)

        buf = self.surface.get_data()
        machines_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]

        #separate Image for Materialflow
        draw_BG(self.ctx, self.width, self.heigth, darkmode=False)
        draw_detail_paths(self.ctx, self.factory.fullPathGraph, self.factory.reducedPathGraph)
        drawFactory(self.ctx, self.factory.machine_dict, None, self.factory.dfMF, drawColors = False, drawNames=False, highlight=self.currentMachine)
        
        buf = self.surface.get_data()
        materialflow_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]

        return np.concatenate((machines_greyscale, materialflow_greyscale), axis=2) 

    def close(self):
        self.surface.finish()
        del(self.surface)
        del(self.ctx)
        self.rsurface.finish()
        del(self.rsurface)
        del(self.rctx)

MultiFactorySimEnv = make_multi_agent(lambda config: FactorySimEnv(config))

#------------------------------------------------------------------------------------------------------------

def main():

    #filename = "Long"
    filename = "Basic"
    #filename = "Simple"
    #filename = "SimpleNoCollisions"

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "Input",
        "1",  
        filename + ".ifc")

    env_config = {
        "inputfile" : ifcpath,
        "obs_type" : 'image',
        "Loglevel" : 1,
        "width" : 84,
        "heigth" : 84,
        "maxMF_Elements" : 10,
        "outputScale" : 10,
        "objectScaling" : 1.0,
        "rendermode": "human",
            }
        
    env = FactorySimEnv( env_config = env_config)

    prefix="test"

    for _ in tqdm(range(0,50)):
        observation, reward, done, info = env.step([random.uniform(-1,1),random.uniform(-1,1), random.uniform(-1, 1), random.uniform(0, 1)])    
        env.render(mode='human', prefix=prefix)
        if done:
            env.reset()
            env.render(mode='human', prefix=prefix)



        
    
if __name__ == "__main__":
    main()
