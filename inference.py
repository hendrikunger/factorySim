from factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
import argparse
import yaml
import ray
import os

from gymnasium import spaces
import numpy as np
from tqdm import tqdm

import wandb
import datetime

from helpers.cli import get_args_inference


args = get_args_inference()

def inference():
 
    #filename = "Long"
    #filename = "Basic"
    filename = "Simple"
    #filename = "EDF"
    #filename = "SimpleNoCollisions"

    basePath = os.path.dirname(os.path.realpath(__file__))
    checkpointPath = os.path.join(basePath, "artifacts", "checkpoint_SAC")

    ifcPath = os.path.join(basePath, "Input", "2", f"{filename}.ifc")
    #ifcPath = os.path.join(basePath, "Input", "2")

    configpath = os.path.join(basePath,"config.yaml")



    with open(configpath, 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)
    f_config['env_config']['inputfile'] = ifcPath
    f_config['env_config']['Loglevel'] = 0
    #f_config['env_config']['render_mode'] = "rgb_array"
    f_config['env_config']['render_mode'] = "rgb_array"
    f_config['env_config']['evaluation'] = True    

    run = wandb.init(
        project="factorySim_ENVTEST",
        name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=f_config,
        save_code=True,
        mode="online",
    )

    env = FactorySimEnv( env_config = f_config['env_config'])
    env.prefix="test"

    if args.rollout: 
        print("Restore RL module from checkpoint ...")       
        rl_module  = RLModule.from_checkpoint(os.path.join( checkpointPath, "learner_group", "learner", "rl_module", DEFAULT_MODULE_ID, ))

        print("Restore env-to-module connector from checkpoint ...")
        env_to_module = EnvToModulePipeline.from_checkpoint(os.path.join( checkpointPath, COMPONENT_ENV_RUNNER, COMPONENT_ENV_TO_MODULE_CONNECTOR,))

        print("Restore module-to-env connector from checkpoint ...")
        module_to_env = ModuleToEnvPipeline.from_checkpoint(os.path.join( checkpointPath, COMPONENT_ENV_RUNNER, COMPONENT_MODULE_TO_ENV_CONNECTOR,))
    

    ratingkeys = ['TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends','terminated',]
    tbl = wandb.Table(columns=["image"] + ratingkeys)


    for key in ratingkeys:
        wandb.define_metric(key, summary="mean")

    obs, info = env.reset()
    episode = SingleAgentEpisode(
        observations=[obs],
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    to_env = None

    for index in tqdm(range(0,50)):

        shared_data = {}

        if args.rollout:
            input_dict = env_to_module(episodes=[episode], rl_module=rl_module, explore=False, shared_data=shared_data,)
            rl_module_out = rl_module.forward_inference(input_dict)
            to_env = module_to_env(batch=rl_module_out, episodes=[episode], rl_module=rl_module, explore=False, shared_data=shared_data,)
            action = to_env[Columns.ACTIONS][0]
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action) 
        
        episode.add_env_step(obs, action, reward, terminated=terminated, truncated=truncated,
                              extra_model_outputs={k: v[0] for k, v in to_env.items()} if to_env else None,
                              )

        if env.render_mode == "rgb_array":   
            image = wandb.Image(env.render(), caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}")
        else:
            image = None
            env.render()

        tbl.add_data(image, *[info.get(key, -1) for key in ratingkeys])

        if terminated:
            wandb.log(info)
            obs, info = env.reset()
            episode = SingleAgentEpisode(
                observations=[obs],
                observation_space=env.observation_space,
                action_space=env.action_space,
            )

    env.close()

    run.log({'results': tbl})
    run.finish()
    if args.rollout:
        ray.shutdown()


def loadTest():

    #ray.init(num_gpus=0, include_dashboard=False) 

    basePath = os.path.dirname(os.path.realpath(__file__))
    checkpointPath = os.path.join(basePath, "artifacts", "checkpoint_SAC")
    with open('config.yaml', 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)

    observation_space = spaces.Box(low=0.0, high=1.0, shape=(128, 128, 3), dtype=np.float64)
    obs = observation_space.sample()
    print("Loading RL Module:")
    rl_module  = RLModule.from_checkpoint(os.path.join(
            checkpointPath,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        ))

    input_dict = {Columns.OBS: np.expand_dims(obs, 0)}
    res = rl_module.forward_inference(obs)
    print(res)

    print("\n\n\n")


    action = rl_module.forward_inference(input_dict)
    print(action)


    #ray.shutdown()

def main():
    inference()
    #loadTest()


if __name__ == "__main__":
    main()