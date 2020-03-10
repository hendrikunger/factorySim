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
    metadata = {'render.modes': ['human']}  

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, inputfile, Loglevel):
        super()
        self.stepCount = 0
        file_name, _ = os.path.splitext(inputfile)
        materialflowpath = file_name + "_Materialflow.csv"
    
        self.factory = FactorySim(inputfile, path_to_materialflow_file = materialflowpath, verboseOutput = Loglevel)
        self.machineCount = len(self.factory.machine_list)
        self.currentMachine = 0
        self.lastMachine = None
 
    def step(self, action):

        #Index, xPos, yPos, Rotation
        self.factory.update(self.currentMachine, random.randint(0, 1000),random.randint(0, 1000), 0)
        self.factory.evaluate()
        self.stepCount += 1
        self.lastMachine = self.currentMachine
        self.currentMachine += 1
        
        if(self.currentMachine >= self.machineCount):
            self.currentMachine = 0

        
 
    def reset(self):
        self.stepCount = 0
 
    def render(self, mode='human', close=False):
        output = self.factory.drawPositions(drawMaterialflow = True, drawMachineCenter = False, highlight=self.currentMachine)
        output = self.factory.drawCollisions(surfaceIn = output)



        if mode == 'rgb_array':
            buf = output.get_data()
            data = np.ndarray(shape=(self.factory.WIDTH, self.factory.HEIGHT), dtype=np.uint32, buffer=buf)
            return data
        elif mode == 'human':
            outputPath = "/workspace/factorySim/Output/" + f"state_{self.stepCount:04d}.png" 
            output.write_to_png(outputPath) 





#------------------------------------------------------------------------------------------------------------
def main():
    env = FactorySimEnv("/workspace/factorySim/Input/Simple.ifc", Loglevel=0)    
    env.render()

    for _ in tqdm(range(0,100)):
        env.step(None)    
        env.render(mode='rgb_array')
    print(env.factory)

    
    
if __name__ == "__main__":
    main()