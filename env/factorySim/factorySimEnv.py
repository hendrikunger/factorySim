import os
import random
import yaml

import gymnasium as gym
from gymnasium import error, spaces


import numpy as np
import cairo
from tqdm import tqdm

from factorySim.factorySimClass import FactorySim
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import make_multi_agent


import factorySim.baseConfigs as baseConfigs
from factorySim.rendering import  draw_BG, drawFactory, drawCollisions, draw_detail_paths, draw_text, drawMaterialFlow

 
class FactorySimEnv(gym.Env):  
    metadata = {'render_modes': ['human', 'rgb_array']}

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, env_config: EnvContext, render_mode=None):
        super().__init__()
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
        if env_config["inputfile"] is not None:
            file_name, _ = os.path.splitext(env_config["inputfile"])
        else:
            exit("No inputfile given.")
        self.inputfile = env_config["inputfile"]
        self.materialflowpath = None #file_name + "_Materialflow.csv"
        self.factoryConfig = baseConfigs.BaseFactoryConf.byStringName(env_config["factoryconfig"])
        self.render_mode = render_mode if not render_mode is None else env_config.get("render_mode", None)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        
        self.surface = None
        self.rsurface = None
        self.prefix = env_config.get("prefix", "0")
        

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
        self.currentMappedReward, self.currentReward, self.info, terminated = self.factory.evaluate()
        #print(F"Reward: {self.currentMappedReward}")
        self.stepCount += 1
        self.currentMachine += 1
        
        if(self.currentMachine >= self.machineCount):
            self.currentMachine = 0

        if self.render_mode == "human":
            self._render_frame()
        return (self._get_obs(), self.currentMappedReward, terminated, False, self.info)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #print("\nReset")
        del(self.factory)
        self.factory = FactorySim(self.inputfile,
        path_to_materialflow_file = self.materialflowpath,
        factoryConfig=self.factoryConfig,
        randomPos=False,
        createMachines=True,
        randseed = seed,
        verboseOutput=self.Loglevel,
        maxMF_Elements = self.maxMF_Elements)
        if self.surface:
            self.surface.finish()
            del(self.surface)
        if self.rsurface:
            self.rsurface.finish()
            del(self.rsurface)
        self.surface, self.ctx = self.factory.provideCairoDrawingData(self.width, self.heigth)
        self.rsurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width * self.scale, self.heigth*self.scale)
        self.rctx = cairo.Context(self.rsurface)

        self.rctx.scale(self.scale*self.factory.scale, self.scale*self.factory.scale)
        self.rctx.translate(-self.factory.factoryCreator.bb.bounds[0], -self.factory.factoryCreator.bb.bounds[1])

        self.machineCount = len(self.factory.machine_dict)
        self.stepCount = 0
        self.currentMachine = 0
        self.currentReward = 0
        self.currentMappedReward = 0
        self.uid +=1

        self.factory.evaluate()
        if self.render_mode == "human":
            self._render_frame()
        return (self._get_obs(), self.info)
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        
        draw_BG(self.rctx, self.factory.DRAWINGORIGIN,*self.factory.FACTORYDIMENSIONS, darkmode=False)
        drawFactory(self.rctx, self.factory, drawColors=True, drawNames=False, highlight=self.currentMachine)
        
        draw_detail_paths(self.rctx, self.factory.fullPathGraph, self.factory.reducedPathGraph, asStreets=True)
        drawCollisions(self.rctx, self.factory.machineCollisionList, wallCollisionList=self.factory.wallCollisionList, outsiderList=self.factory.outsiderList)
        drawMaterialFlow(self.rctx, self.factory.machine_dict, self.factory.dfMF, drawColors=True)
        draw_text(self.rctx, 
                  f"{self.uid:02d}.{self.stepCount:02d}       Mapped Reward: {self.currentMappedReward:1.2f} | Reward: {self.currentReward:1.2f}",
                  (1,0,0),
                  (20, 20),
                  factoryCoordinates=False)
        
        if self.render_mode == 'human':
            outputPath = os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}.png")
            self.rsurface.write_to_png(outputPath)
        elif self.render_mode == 'rgb_array':
            buf = self.rsurface.get_data()
            #bgra to rgb
            #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0,3]]
            rgb = np.ndarray(shape=(self.width * self.scale, self.heigth * self.scale, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]
            return rgb
        

    def _get_obs(self):

        #old colorimage
        #bgra to rgb
        #rgb = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]

        #new Version greyscale

        draw_BG(self.ctx, self.factory.DRAWINGORIGIN, *self.factory.FACTORYDIMENSIONS, darkmode=False)
        drawFactory(self.ctx, self.factory, None, drawColors = False, drawNames=False, highlight=self.currentMachine, isObs=True, wallInteriorColor = (1, 1, 1),)
        drawCollisions(self.ctx, self.factory.machineCollisionList, self.factory.wallCollisionList, outsiderList=self.factory.outsiderList)

        buf = self.surface.get_data()
        machines_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]
        #self.surface.write_to_png(os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}_agent_1_collision.png"))

        #separate Image for Materialflow
        draw_BG(self.ctx, self.factory.DRAWINGORIGIN, *self.factory.FACTORYDIMENSIONS, darkmode=False)
        draw_detail_paths(self.ctx, self.factory.fullPathGraph, self.factory.reducedPathGraph)
        drawFactory(self.ctx, self.factory, self.factory.dfMF, drawWalls=False, drawColors = False, drawNames=False, highlight=self.currentMachine, isObs=True)
        
        buf = self.surface.get_data()
        materialflow_greyscale = np.ndarray(shape=(self.width, self.heigth, 4), dtype=np.uint8, buffer=buf)[...,[2]]
        #self.surface.write_to_png(os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}_agent_2_materialflow.png"))

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

    
    import wandb
    import datetime

    def logging(env):
        image = wandb.Image(env.render(), caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}")
        wandb.log({"output": image})


    #filename = "Long"
    #filename = "Basic"
    #filename = "Simple"
    filename = "EDF"
    #filename = "SimpleNoCollisions"

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "Input",
        "2",  
        filename + ".ifc")

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
    "..",
    "..",
    "Input",
    "2")

    
    configpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
        "..",
        "..",
        "config.yaml")

    with open(configpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['env_config']['inputfile'] = ifcpath
    config['env_config']['Loglevel'] = 0
    config['env_config']['render_mode'] = "rgb_array"
    

    run = wandb.init(
        project="factorySim",
        name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=config,
        save_code=True,
    )

            
    env = FactorySimEnv( env_config = config['env_config'])
    env.prefix="test"
    
    ratingkeys = ['TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends','terminated',]
    tbl = wandb.Table(columns=["image"] + ratingkeys)
    for key in ratingkeys:
        wandb.define_metric(key, summary="mean")
 
    for _ in tqdm(range(0,50)):
        observation, reward, terminated, truncated, info = env.step([random.uniform(-1,1),random.uniform(-1,1), random.uniform(-1, 1), random.uniform(0, 1)])    
        image = wandb.Image(env.render(), caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}")
        tbl.add_data(image, *[info[key] for key in ratingkeys])
        if terminated:
            wandb.log(info)
            env.reset()

    run.log({'results': tbl})
    run.finish()


    
if __name__ == "__main__":
    main()
