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
    metadata = {'render.modes': ['human', 'rgb_array', 'imageseries']}

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, inputfile = 'None', obs_type='image', Loglevel=0):
        super()
        self.stepCount = 0
        self._obs_type = obs_type
        if inputfile is not None:
            file_name, _ = os.path.splitext(inputfile)
        else:
            exit("No inputfile given.")
        materialflowpath = file_name + "_Materialflow.csv"
    
        self.factory = FactorySim(inputfile, path_to_materialflow_file = materialflowpath, verboseOutput = Loglevel)
        self.machineCount = len(self.factory.machine_list)
        self.currentMachine = 0
        self.lastMachine = None
        self.output = None

        # Actions of the format MoveX, MoveY, Rotate 
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1,1,1]))

        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.factory.WIDTH, self.factory.HEIGHT), dtype=np.uint32)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))
 
    def step(self, action):
       
        self.factory.update(self.currentMachine, action[0], action[1], action[2])
        reward = self.factory.evaluate()
        self.stepCount += 1
        self.lastMachine = self.currentMachine
        self.currentMachine += 1
        
        if(self.currentMachine >= self.machineCount):
            self.currentMachine = 0

        done = False
        info = {}
    
        return self._get_obs(), reward, done, info
        
 
    def reset(self):
        self.stepCount = 0
        self.currentMachine = 0
        self.lastMachine = None
        self.output = None

        return self._get_obs()
 
    def render(self, mode='human', close=False):

        if mode == 'rgb_array':
            return self._get_np_array()
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'imageseries':
            return self._get_image()

    def _get_obs(self):

        self.output = self.factory.drawPositions(drawMaterialflow = True, drawMachineCenter = False, drawMachineBaseOrigin=True, highlight=self.currentMachine)
        self.output = self.factory.drawCollisions(surfaceIn = self.output)
        if self._obs_type == 'image':
            img = self._get_np_array()
        return img

    def _get_image(self):
        outputPath = "/workspace/factorySim/Output/" + f"state_{self.stepCount:04d}.png" 
        return self.output.write_to_png(outputPath)

    def _get_np_array(self):
        buf = self.output.get_data()
        return np.ndarray(shape=(self.factory.WIDTH, self.factory.HEIGHT), dtype=np.uint32, buffer=buf)




#------------------------------------------------------------------------------------------------------------
def main():

    #filename = "Overlapp"
    #filename = "EP_v23_S1_clean"
    filename = "Simple"
    #filename = "SimpleNoCollisions"

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "..",
        "Input",  
        filename + ".ifc")

    env = FactorySimEnv(inputfile = ifcpath, obs_type='image', Loglevel=2)    
    output = None
    for _ in tqdm(range(0,100)):
        observation, reward, done, info = env.step([random.uniform(0,1),random.uniform(0,1), random.uniform(0, 1)])    
        output = env.render(mode='rgb_array')

    print(output.ndim)
    print(output.shape)
    #np.savetxt('data.csv', output, delimiter=',')

        
    
if __name__ == "__main__":
    from factorySim import FactorySim
    main()