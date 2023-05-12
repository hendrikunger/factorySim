#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from factorySim.factorySimEnv import FactorySimEnv, MultiFactorySimEnv

import ray

from ray import air, tune
from ray.tune.registry import get_trainable_cls
from ray.tune import Tuner
from ray.train.rl.rl_trainer import RLTrainer
from ray.air import Checkpoint
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.result import Result
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from factorySim.customModels import MyXceptionModel

import wandb
import yaml
import pprint 




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
    f_config = yaml.load(f, Loader=yaml.FullLoader)

f_config['env'] = FactorySimEnv
#file_config['callbacks'] = TraceMallocCallback
f_config['callbacks'] = None
f_config['env_config']['inputfile'] = ifcpath




if __name__ == "__main__":
    ray.init(num_gpus=1, local_mode=False, include_dashboard=False) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))


    stop = {
    "training_iteration": 50000,
    "timesteps_total": 5000,
    "episode_reward_mean": 5,
    }

    checkpoint_config = CheckpointConfig(checkpoint_at_end=True, 
                                         checkpoint_frequency=10, 
                                         checkpoint_score_order="max", 
                                         checkpoint_score_attribute="episode_reward_mean", 
                                         num_to_keep=10 
    )

    ppo_config = (
        get_trainable_cls("PPO")
        .get_default_config()
        # or "corridor" if registered above
        .environment(FactorySimEnv, env_config=f_config['env_config'])
        .framework("tf2")
        .rollouts(num_rollout_workers=1)
        .training(
            model={
                "custom_model": "my_model",
                
            }
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
    )



    trainer = RLTrainer(
        run_config=RunConfig(name="klaus",
                                         stop=stop,
                                         checkpoint_config=checkpoint_config
                            ),
        scaling_config=ScalingConfig(num_workers=f_config['num_workers'], 
                                     use_gpu=True,
                                    ),
        algorithm="PPO",
        config=ppo_config,

    )
    if Tuner.can_restore("/root/ray_results/PPO_2022-11-29_21-36-58/PPO_MultiEnv_f3c8f_00000_0_2022-11-29_21-36-58"):

        #Continuing training
        tuner = Tuner.restore("/root/ray_results/PPO_2022-11-29_21-36-58/PPO_MultiEnv_f3c8f_00000_0_2022-11-29_21-36-58", trainable=trainer)
        results = tuner.fit() 

    else:

        tuner = tune.Tuner(trainer)
        results = tuner.fit()

    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")



    ray.shutdown()

