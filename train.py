#!/usr/bin/env python3
import os

from factorySim.factorySimEnv import FactorySimEnv, MultiFactorySimEnv

import ray


from ray.tune.logger import pretty_print
from ray.rllib.agents import ppo
from ray import air, tune


from ray.rllib.models import ModelCatalog
from factorySim.customModels import MyXceptionModel

import wandb
import yaml

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 


#filename = "Overlapp"
filename = "Basic"
#filename = "EP_v23_S1_clean"
#filename = "Simple"
#filename = "SimpleNoCollisions"
#filename = "LShape"

#ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1", filename + ".ifc")
ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "2")

#Import Custom Models
ModelCatalog.register_custom_model("my_model", MyXceptionModel)

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config['env'] = MultiFactorySimEnv
#config['callbacks'] = TraceMallocCallback
config['callbacks'] = None
config['env_config']['inputfile'] = ifcpath




if __name__ == "__main__":
    ray.init(num_gpus=1, local_mode=False, include_dashboard=False) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))





    stop = {
    "training_iteration": 500,
    "timesteps_total": 50000,
    "episode_reward_mean": 3,
    }

    tuner = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=air.RunConfig(stop=stop, checkpoint_config=2),
        )
    results = tuner.fit()
    ray.shutdown()


    # for _ in range(500): #args.stop_iters,
    #     result = trainer.train()
    #     print(pretty_print(result))
    #     # stop training of the target train steps or reward are reached
    #     if (
    #         result["timesteps_total"] >= 2000000#args.stop_timesteps
    #         or result["episode_reward_mean"] >= 50000 #args.stop_reward
    #     ):
    #         break
