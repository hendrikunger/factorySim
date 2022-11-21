#!/usr/bin/env python3
import os

from factorySim.factorySimEnv import FactorySimEnv, MultiFactorySimEnv

import ray


from ray.tune.logger import pretty_print
from ray import air, tune
import ray.rllib.algorithms.ppo as ppo

from ray.rllib.models import ModelCatalog
from factorySim.customModels import MyXceptionModel

import wandb
import yaml

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 


RESUME = False

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
    "training_iteration": 50000,
    "timesteps_total": 2000000,
    "episode_reward_mean": 5,
    }


    if not RESUME:
  

        tuner = tune.Tuner(ppo.PPO,
                param_space=config,
                run_config=air.RunConfig(stop=stop, checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=100, checkpoint_score_attribute="episode_reward_mean", num_to_keep=10 ))
                )
        results = tuner.fit()

    else:
    #Continuing training

        stop = {
        "training_iteration": 50000,
        "timesteps_total": 4000000,
        "episode_reward_mean": 5,
        }

        results = ray.tune.run(ppo.PPO, 
                    config=config, 
                    stop=stop, 
                    reuse_actors=True, 
                    checkpoint_freq=10,
                    keep_checkpoints_num=100,
                    checkpoint_score_attr="episode_reward_mean", 
                    restore="/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")


    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")



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
