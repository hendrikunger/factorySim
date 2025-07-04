import os
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
from factorySim.rendering import draw_obs_layer_A, draw_obs_layer_B, draw_obs_layer_C




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
        self.logLevel = env_config["logLevel"]
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
        self.useCoordinateChannels = env_config.get("coordinateChannels", False)
        if env_config.get("inputfile", None) is not None:
            file_name, _ = os.path.splitext(env_config["inputfile"])
        else:
            print("No inputfile given.")
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
        self.materialflowpath = None #file_name + "_Materialflow.csv"

        if self.evaluationMode:  
            print("\n\n-----------------------------------------------Evaluation Mode-----------------------------------------------\n\n")
            if(os.path.isdir(env_config["inputfile"])):
                self.evalFiles = [x for x in os.listdir(self.evalPath) if ".ifc" in x]
            else:
                self.evalFiles = [self.inputfile]
            self.evalFiles.sort()
            self.createMachines = False

        
        self.factoryConfig = baseConfigs.BaseFactoryConf.byStringName(env_config["factoryconfig"])
        self.render_mode = render_mode if not render_mode is None else env_config.get("render_mode", None)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        
        self.surface = None
        self.rsurface = None
        self.ctx = None
        self.rctx = None
        self.prefix = env_config.get("prefix", "0") 

        self.info = {}
        self.terminated = False


        # Actions of the format MoveX, MoveY, Rotate, (Skip) 
        #self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0]), high=np.array([1,1,1,1]), dtype=np.float32)
        #Skipping disabled
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        if self._obs_type == 'image':
            #self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 2), dtype=np.uint8)
            dimensions =  5 if env_config.get("coordinateChannels", False) else 3
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, dimensions), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))
        
        if self.useCoordinateChannels:
            # Precalculate the coordinates of the factory as channels
            x = np.linspace(0, 255, self.width, dtype=np.uint8)
            y = np.linspace(0, 255, self.height, dtype=np.uint8)
            self.y_coordChannel, self.x_coordChannel = np.meshgrid(y, x, indexing='ij')

            #Add third axis for the coordinate channels
            self.x_coordChannel = np.expand_dims(self.x_coordChannel, axis=2)
            self.y_coordChannel = np.expand_dims(self.y_coordChannel, axis=2)






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
     
        return (self._get_obs(), self.currentReward, self.terminated, False, self.info)

    def reset(self, seed=None, options={}):
        if seed is not None:
            self.seed = seed
            print(f"Seed set to {self.seed}")
        super().reset(seed=self.seed)
        del(self.factory)
        self.uid +=1 
        if self.evaluationMode:
            self.currentEvalEnv = self.uid % len(self.evalFiles)  
            self.inputfile = os.path.join(self.evalPath, self.evalFiles[self.currentEvalEnv])
            self.materialflowpath = self.inputfile.replace(".ifc", "_mf.csv")
            if os.path.exists(self.materialflowpath):
                print(f"Loading materialflow from {self.materialflowpath}", flush=True)
            else:
                print(f"Would load materialflow from {self.materialflowpath}, but file does not exist", flush=True)
                self.materialflowpath = None

        self.factory = FactorySim(self.inputfile,
        path_to_materialflow_file = self.materialflowpath,
        factoryConfig=self.factoryConfig,
        randomPos=False,
        createMachines=self.createMachines,
        randSeed = self.seed,
        logLevel=self.logLevel,
        maxMF_Elements = self.maxMF_Elements)
        self.info = {}
        if self.surface:
            self.surface.finish()
            del(self.surface)
        if self.ctx:
            del(self.ctx)
        if self.rsurface:
            self.rsurface.finish()
            del(self.rsurface)
        if self.rctx:
            del(self.rctx)
        self.surface, self.ctx = self.factory.provideCairoDrawingData(self.width, self.height)
        self.rsurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width * self.scale, self.height*self.scale)
        self.rctx = cairo.Context(self.rsurface)

        self.rctx.scale(self.scale*self.factory.scale, self.scale*self.factory.scale)
        self.rctx.translate(-self.factory.creator.bb.bounds[0], -self.factory.creator.bb.bounds[1])

        self.machineCount = len(self.factory.machine_dict)
        self.stepCount = 0
        self.currentMachine = 0
        self.currentReward = 0
        self.materialflowpath = None
        

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


    def _render_frame(self, path=None):
        
        draw_BG(self.rctx, self.factory.DRAWINGORIGIN,*self.factory.FACTORYDIMENSIONS, darkmode=False)
        drawFactory(self.rctx, self.factory, drawColors=True, darkmode=False, drawNames=False, highlight=str(self.currentMachine))
        
        draw_detail_paths(self.rctx, self.factory.fullPathGraph, self.factory.reducedPathGraph, asStreets=True)
        drawCollisions(self.rctx, self.factory.machineCollisionList, wallCollisionList=self.factory.wallCollisionList, outsiderList=self.factory.outsiderList)
        drawMaterialFlow(self.rctx, self.factory.machine_dict, self.factory.dfMF, drawColors=True)

        
        if self.render_mode == 'human':
            if path is None:
                outputPath = os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}.png")
            else:
                outputPath = path + ".png"
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
        machines_greyscale = self._surface_to_grayscale(self.surface)
        #self.surface.write_to_png(os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}_agent_1_collision.png"))

        #separate Image for Materialflow
        draw_obs_layer_B(self.ctx, self.factory, highlight=machineToHighlight)
        materialflow_greyscale = self._surface_to_grayscale(self.surface)

        #separate Image for Collisions
        draw_obs_layer_C(self.ctx, self.factory, highlight=machineToHighlight)
        collisions_greyscale = self._surface_to_grayscale(self.surface)

        #self.surface.write_to_png(os.path.join(self.output_path, f"{self.prefix}_{self.uid}_{self.stepCount:04d}_agent_2_materialflow.png"))
        
        
        #Format (width, height, 2)
        if self.useCoordinateChannels:
            output = np.concatenate((machines_greyscale, materialflow_greyscale, collisions_greyscale, self.x_coordChannel, self.y_coordChannel), axis=2, dtype=np.uint8)
        else:
            output = np.concatenate((machines_greyscale, materialflow_greyscale, collisions_greyscale), axis=2, dtype=np.uint8)
        
        return output
    
    def _surface_to_grayscale(self, surface):
        buf = surface.get_data()
        image = np.ndarray(shape=(self.width, self.height, 4), dtype=np.uint8, buffer=buf)[...,[2,1,0]]
        del(buf)

    # Check if image has 3 channels (RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            # Apply the standard formula: 0.299*R + 0.587*G + 0.114*B
            grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
            # Convert back to uint8 but add axis to keep the shape consistent  (H, W , 1)
            return grayscale[..., np.newaxis].astype(np.uint8)
        else:
            raise ValueError("Input image must be an RGB image with shape (H, W, 3)")
    
      

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
            self.currentReward, self.info, self.terminated = self.factory.evaluate(self.reward_function)
        except Exception as e:
            print(e)
            print("Error in evaluate")
            self.currentReward = -10
            self.info = {}
            self.terminated = True
        
    def __str__():
        return "FactorySimEnv"

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
    f_config['env_config']['logLevel'] = 0

 

    run = wandb.init(
        project="factorySim_ENVTEST",
        name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=f_config,
        save_code=True,
        mode="online",
    )

    env = FactorySimEnv( env_config = f_config['env_config'])
    env.prefix="test"
    
    
    ratingkeys = ['Reward', 'TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends','terminated',]
    tbl = wandb.Table(columns=["evalFile", "evalFile.Step", "image"] + ratingkeys)

    

    for key in ratingkeys:
        wandb.define_metric(key, summary="mean")
    obs, info = env.reset()
    image = wandb.Image(env.info["Image"], caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}")
    tbl.add_data(0, f"{0}.{env.stepCount}", image, *[info.get(key, -1) for key in ratingkeys])
 
    for index in tqdm(range(0,10)):

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
