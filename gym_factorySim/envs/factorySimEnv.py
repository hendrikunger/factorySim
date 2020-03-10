import os
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import cairo
from tqdm import tqdm

from factorySim import FactorySim
 
class FactorySimEnv(gym.Env):  
    metadata = {'render.modes': ['human', 'rgb_array']}

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, inputfile, obs_type='image', Loglevel=0):
        super()
        self.stepCount = 0
        self._obs_type = obs_type
        file_name, _ = os.path.splitext(inputfile)
        materialflowpath = file_name + "_Materialflow.csv"
    
        self.factory = FactorySim(inputfile, path_to_materialflow_file = materialflowpath, verboseOutput = Loglevel)
        self.machineCount = len(self.factory.machine_list)
        self.currentMachine = 0
        self.lastMachine = None
        self.output = None
 
    def step(self, action):

        #Index, xPos, yPos, Rotation
        self.factory.update(self.currentMachine, random.randint(0, 1000),random.randint(0, 1000), 0)
        reward = self.factory.evaluate()
        self.stepCount += 1
        self.lastMachine = self.currentMachine
        self.currentMachine += 1
        
        if(self.currentMachine >= self.machineCount):
            self.currentMachine = 0

        self.output = self.factory.drawPositions(drawMaterialflow = True, drawMachineCenter = False, highlight=self.currentMachine)
        self.output = self.factory.drawCollisions(surfaceIn = self.output)

        done = False
        info = {}
    
        return self._get_obs(), reward, done, info
        
 
    def reset(self):
        self.stepCount = 0
 
    def render(self, mode='human', close=False):

        if mode == 'rgb_array':
            return self._get_np_array()
        elif mode == 'human':
            #add rendering to window
            return self._get_image()
            


    def _get_image(self):
        outputPath = "/workspace/factorySim/Output/" + f"state_{self.stepCount:04d}.png" 
        return self.output.write_to_png(outputPath)

    def _get_np_array(self):
        buf = self.output.get_data()
        return np.ndarray(shape=(self.factory.WIDTH, self.factory.HEIGHT), dtype=np.uint32, buffer=buf)

    def _get_obs(self):
        if self._obs_type == 'image':
            img = self._get_np_array()
        return img


#------------------------------------------------------------------------------------------------------------
def main():
    env = FactorySimEnv("/workspace/factorySim/Input/Simple.ifc", obs_type='image', Loglevel=0)    

    for _ in tqdm(range(0,100)):
        observation, reward, done, info = env.step(None)    
        #env.render(mode='human')
        print(reward)

    
    
if __name__ == "__main__":
    main()