from factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.policy.policy import Policy
import yaml
import ray
import os

from gymnasium import spaces
import numpy as np
from tqdm import tqdm

import wandb
import datetime




def inference():
    ROLLOUT = True

    #filename = "Long"
    #filename = "Basic"
    filename = "Simple"
    #filename = "EDF"
    #filename = "SimpleNoCollisions"

    basePath = os.path.dirname(os.path.realpath(__file__))
    checkpointPath = os.path.join(basePath, "artifacts", "checkpoint_PPO_latest")

    ifcPath = os.path.join(basePath, "Input", "2", f"{filename}.ifc")
    #ifcPath = os.path.join(basePath, "Input", "2")

    configpath = os.path.join(basePath,"config.yaml")



    with open(configpath, 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)
    f_config['env_config']['inputfile'] = ifcPath
    f_config['env_config']['Loglevel'] = 0
    #f_config['env_config']['render_mode'] = "rgb_array"
    f_config['env_config']['render_mode'] = "human"
    f_config['env_config']['evaluation'] = False    

    run = wandb.init(
        project="factorySim_ENVTEST",
        name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=f_config,
        save_code=True,
        mode="offline",
    )

    env = FactorySimEnv( env_config = f_config['env_config'])
    env.prefix="test"

    if ROLLOUT: 
        restored_policy = Policy.from_checkpoint(checkpointPath)["default_policy"]


    ratingkeys = ['TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends','terminated',]
    tbl = wandb.Table(columns=["image"] + ratingkeys)


    for key in ratingkeys:
        wandb.define_metric(key, summary="mean")

    obs, info = env.reset()

    for index in tqdm(range(0,50)):

        if ROLLOUT:
            action = restored_policy.compute_single_action(obs)[0]
        else:
            action = env.action_space.sample()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action) 

        if env.render_mode == "rgb_array":   
            image = wandb.Image(env.render(), caption=f"{env.prefix}_{env.uid}_{env.stepCount:04d}")
        else:
            image = None
            env.render()
        tbl.add_data(image, *[info.get(key, -1) for key in ratingkeys])
        if terminated:
            wandb.log(info)
            env.reset()

    env.close()

    run.log({'results': tbl})
    run.finish()
    if ROLLOUT:
        ray.shutdown()


def loadTest():

    #ray.init(num_gpus=0, include_dashboard=False) 

    basePath = os.path.dirname(os.path.realpath(__file__))
    checkpointPath = os.path.join(basePath, "artifacts", "checkpoint_PPO_latest")
    with open('config.yaml', 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)

    observation_space = spaces.Box(low=0.0, high=1.0, shape=(84, 84, 2), dtype=np.float64)
    obs = observation_space.sample()

    my_restored_policy = Policy.from_checkpoint(checkpointPath)["default_policy"]

    print(my_restored_policy)
    res = my_restored_policy.compute_single_action(obs)
    print(res)

    print("\n\n\n")

    ppo_config = ppo.PPOConfig()
    ppo_config.environment(FactorySimEnv, env_config=f_config['env_config'], render_env=False)
    ppo_config.rollouts(num_rollout_workers=0)
    ppo_config.framework(framework="torch", eager_tracing=False,)


    # Build the Algorithm instance using the config.
    algorithm = ppo_config.build()
    # Restore the algo's state from the checkpoint.
    algorithm.restore(checkpointPath)
    action = algorithm.compute_single_action(obs)
    print(action)


    #ray.shutdown()

def main():
    inference()
    #loadTest()


if __name__ == "__main__":
    main()