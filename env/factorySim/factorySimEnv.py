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
from factorySim.rendering import draw_obs_layer_A, draw_obs_layer_B




class FactorySimEnv(gym.Env):  
    metadata = {'render_modes': ['human', 'rgb_array']}

    #Expects input ifc file. Other datafiles have to have the same path and filename. 
    def __init__(self, env_config: EnvContext, render_mode=None):
        super().__init__()
        print(env_config)
        self.evaluationMode = env_config["evaluation"]
        self.factory = None
        self.stepCount = 0
        self._obs_type = env_config["obs_type"]
        self.Loglevel = env_config["Loglevel"]
        self.uid = -1
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.reward_function = env_config["reward_function"]
        self.maxMF_Elements = env_config["maxMF_Elements"]
        self.createMachines = env_config["createMachines"]
        self.scale = env_config["outputScale"]
        self.evalFiles = [None]
        self.currentEvalEnv = None
        self.seed = env_config["randomSeed"]
        if env_config["inputfile"] is not None:
            file_name, _ = os.path.splitext(env_config["inputfile"])
        else:
            exit("No inputfile given.")
        self.inputfile = env_config["inputfile"]

        if(os.path.isdir(env_config["inputfile"])):
            self.base_path = os.path.join(os.path.dirname(os.path.realpath(self.inputfile)),
            "..")
        else:
            self.base_path = os.path.join(os.path.dirname(os.path.realpath(self.inputfile)), 
            "..",
            "..")
        self.output_path = os.path.join(self.base_path, "Output")
        self.evalPath= os.path.join(self.base_path, "Evaluation")

        if self.evaluationMode:  
            print("\n\n-----------------------------------------------Evaluation Mode-----------------------------------------------\n\n")
            self.evalFiles = [x for x in os.listdir(self.evalPath) if ".ifc" in x]
            self.createMachines = False

        self.materialflowpath = None #file_name + "_Materialflow.csv"
        self.factoryConfig = baseConfigs.BaseFactoryConf.byStringName(env_config["factoryconfig"])
        self.render_mode = render_mode if not render_mode is None else env_config.get("render_mode", None)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        
        self.surface = None
        self.rsurface = None
        self.prefix = env_config.get("prefix", "0") 

        self.info = {}
        self.terminated = False


        # Actions of the format MoveX, MoveY, Rotate, (Skip) 
        #self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0]), high=np.array([1,1,1,1]), dtype=np.float32)
        #Skipping disabled
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64)

        if self._obs_type == 'image':
            #self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 2), dtype=np.uint8)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.width, self.height, 2), dtype=np.float64)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))




    def step(self, action):
        if not np.isnan(action[0]) :
            self.factory.update(self.currentMachine, action[0], action[1], action[2], 0)

        self.tryEvaluate()

        self.stepCount += 1
        self.currentMachine += 1
        
        if(self.currentMachine >= self.machineCount):
            self.currentMachine = 0

        if self.evaluationMode:
            self.info["Evaluation"] = True
            self.info["Image"] = self.render()
            self.info["Step"] = self.stepCount
            self.info["evalEnvID"] = self.currentEvalEnv
     
        return (self._get_obs(), self.currentMappedReward, self.terminated, False, self.info)

    def reset(self, seed=None, options={}):
        super().reset(seed=self.seed)
        del(self.factory)
        self.uid +=1 
        if self.evaluationMode:
            self.currentEvalEnv = self.uid % len(self.evalFiles)  
            self.inputfile = os.path.join(self.evalPath, self.evalFiles[self.currentEvalEnv])
            print(self.inputfile)
        self.factory = FactorySim(self.inputfile,
        path_to_materialflow_file = self.materialflowpath,
        factoryConfig=self.factoryConfig,
        randomPos=False,
        createMachines=self.createMachines,
        randSeed = self.seed,
        verboseOutput=self.Loglevel,
        maxMF_Elements = self.maxMF_Elements)
        self.info = {}
        if self.surface:
            self.surface.finish()
            del(self.surface)
        if self.rsurface:
            self.rsurface.finish()
            del(self.rsurface)
        self.surface, self.ctx = self.factory.provideCairoDrawingData(self.width, self.height)
        self.rsurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width * self.scale, self.height*self.scale)
        self.rctx = cairo.Context(self.rsurface)

        self.rctx.scale(self.scale*self.factory.scale, self.scale*self.factory.scale)
        self.rctx.translate(-self.factory.creator.bb.bounds[0], -self.factory.creator.bb.bounds[1])

        self.machineCount = len(self.factory.machine_dict)
        self.stepCount = 0
        self.currentMachine = 0
        self.currentReward = 0
        self.currentMappedReward = 0
        

        self.tryEvaluate()

        if self.evaluationMode:
            self.info["Evaluation"] = True
            self.info["Image"] = self.render()
            self.info["Step"] = self.stepCount
            self.info["evalEnvID"] = self.currentEvalEnv

        if (self.render_mode == "human" and options.get("RenderInitFrame", True)):
            self._render_frame() 

        return (self._get_obs(), self.info)

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            return self._render_frame()
        else:
            raise error.Error('Unrecognized render mode: {}'.format(self.render_mode))


    def _render_frame(self):
        
        draw_BG(self.rctx, self.factory.DRAWINGORIGIN,*self.factory.FACTORYDIMENSIONS, darkmode=False)
        drawFactory(self.rctx, self.factory, drawColors=True, darkmode=False, drawNames=False, highlight=str(self.currentMachine))
        
        draw_detail_paths(self.rctx, self.factory.fullPathGraph, self.factory.reducedPathGraph, asStreets=True)
        drawCollisions(self.rctx, self.factory.machineCollisionList, wallCollisionList=self.factory.wallCollisionList, outsiderList=self.factory.outsiderList)
        drawMaterialFlow(self.rctx, self.factory.machine_dict, self.factory.dfMF, drawColors=True)
        # draw_text(self.rctx, 
        #           f"{self.uid:02d}.{self.stepCount:02d}       Mapped Reward: {self.currentMappedReward:1.2f} | Reward: {self.currentReward:1.2f}",
        #           (1,0,0),
        #           (20, 20),
        #           factoryCoordinates=False)
        
        if self.render_mode == 'human':
            outputPath = os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}.png")
            self.rsurface.write_to_png(outputPath)
            return np.array([])
        elif self.render_mode == 'rgb_array':
            buf = self.rsurface.get_data()
            #bgra to rgb
            #rgb = np.ndarray(shape=(self.width, self.height, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0,3]]
            rgb = np.ndarray(shape=(self.width * self.scale, self.height * self.scale, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]
            return rgb
        

    def _get_obs(self, highlight=None):

        #old colorimage
        #bgra to rgb
        #rgb = np.ndarray(shape=(self.width, self.height, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]

        #new Version greyscale
        machineToHighlight = highlight if not highlight is None else str(self.currentMachine)

        draw_obs_layer_A(self.ctx, self.factory, highlight=machineToHighlight)

        buf = self.surface.get_data()

        machines_greyscale = np.ndarray(shape=(self.width, self.height, 4), dtype=np.uint8, buffer=buf)[...,[2]]
        #self.surface.write_to_png(os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}_agent_1_collision.png"))

        #separate Image for Materialflow
        draw_obs_layer_B(self.ctx, self.factory, highlight=machineToHighlight)

        buf = self.surface.get_data()

        materialflow_greyscale = np.ndarray(shape=(self.width, self.height, 4), dtype=np.uint8, buffer=buf)[...,[2]]
        #self.surface.write_to_png(os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}_agent_2_materialflow.png"))
        #Format (width, height, 2)
        output = np.concatenate((machines_greyscale, materialflow_greyscale), axis=2)
        
        return output/ 255.0
    
      

    def close(self):
        if self.surface:
            self.surface.finish()
            del(self.surface)
            del(self.ctx)
        if self.rsurface:
            self.rsurface.finish()
            del(self.rsurface)
            del(self.rctx)

    def tryEvaluate(self):
        try:
            self.currentMappedReward, self.currentReward, self.info, self.terminated = self.factory.evaluate(self.reward_function)
        except Exception as e:
            print(e)
            print("Error in evaluate")
            self.currentMappedReward = -10
            self.currentReward = -10
            self.info = {}
            self.terminated = True
        
        

MultiFactorySimEnv = make_multi_agent(lambda config: FactorySimEnv(config))


#------------------------------------------------------------------------------------------------------------




def main():

    
    import wandb
    import datetime


    #filename = "Long"
    #filename = "Basic"
    filename = "Simple"
    #filename = "EDF"
    #filename = "SimpleNoCollisions"

    basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","..")
    
    ifcPath = os.path.join(basePath, "Input", "2", f"{filename}.ifc")
    #ifcPath = os.path.join(basePath, "Input", "2")
    configpath = os.path.join(basePath,"config.yaml")
    

    with open(configpath, 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)

    f_config['env_config']['render_mode'] = "rgb_array"
    f_config['env_config']['inputfile'] = ifcPath
    f_config['env_config']['evaluation'] = True
    f_config['env_config']['randomSeed'] = 42

    run = wandb.init(
        project="factorySim_ENVTEST",
        name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=f_config,
        save_code=True,
        mode="offline",
    )

    env = FactorySimEnv( env_config = f_config['env_config'])
    env.prefix="test"
    
    
    ratingkeys = ['TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends','terminated',]
    tbl = wandb.Table(columns=["evalFile", "evalFile.Step", "image"] + ratingkeys)

    

    for key in ratingkeys:
        wandb.define_metric(key, summary="mean")
    obs, info = env.reset()
    image = wandb.Image(env.info["Image"], caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}")
    tbl.add_data(0, f"{0}.{env.stepCount}", image, *[info.get(key, -1) for key in ratingkeys])
 
    for index in tqdm(range(0,60)):

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) 
        if env.render_mode == "rgb_array":   
            image = wandb.Image(env.info["Image"], caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}")
        else:
            image = None
            env.render()


        tbl.add_data(env.currentEvalEnv, f"{env.currentEvalEnv}.{env.stepCount}", image, *[info.get(key, -1) for key in ratingkeys])
        if terminated:
            wandb.log(info)
            obs, info = env.reset()
            image = wandb.Image(env.render(), caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}") 
            tbl.add_data(env.currentEvalEnv, f"{env.currentEvalEnv}_{env.stepCount}", image, *[info.get(key, -1) for key in ratingkeys])

    env.close()

    

    run.log({'results': tbl})
    run.finish()


    
if __name__ == "__main__":
    main()
