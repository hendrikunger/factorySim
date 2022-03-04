import argparse
import os

from factorySim.factorySimEnv import FactorySimEnv

import ray
from ray import tune
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch

from ray.tune.logger import pretty_print
from ray.rllib.agents import ppo

# parser = argparse.ArgumentParser()
# parser.add_argument("--stop-iters", type=int, default=200)
# parser.add_argument("--num-cpus", type=int, default=10)

#filename = "Overlapp"
filename = "Basic"
#filename = "EP_v23_S1_clean"
#filename = "Simple"
#filename = "SimpleNoCollisions"
#filename = "LShape"

#ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1", filename + ".ifc")
ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1")

config = {
    "env": FactorySimEnv,  # or "corridor" if registered above
    "env_config": {
        "inputfile" : ifcpath,
        "obs_type" : 'image',
        "uid" : 0,
        "Loglevel" : 0,
        "width" : 84,
        "heigth" : 84,
        "maxMF_Elements" : 5,
        "outputScale" : 1,
        "objectScaling" : 1.0,
    },
    "log_level": "DEBUG",
    # Evaluate once per training iteration.
    "evaluation_interval": 1,
    # Run evaluation on (at least) ten episodes
    "evaluation_duration": 10,
    # ... using one evaluation worker (setting this to 0 will cause
    # evaluation to run on the local evaluation worker, blocking
    # training until evaluation is done).
    "evaluation_num_workers": 1,
    # Special evaluation config. Keys specified here will override
    # the same keys in the main config, but only for evaluation.
    "evaluation_config": {
        # Store videos in this relative directory here inside
        # the default output dir (~/ray_results/...).
        # Alternatively, you can specify an absolute path.
        # Set to True for using the default output dir (~/ray_results/...).
        # Set to False for not recording anything.
        "record_env": os.path.join(os.path.dirname(os.path.realpath(__file__)), "Output"),
        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
        # env and only the 1st sub-env in a vectorized env.
        "render_env": False,
    },
    #"tf", "tf2", "tfe", "torch"
    "framework": "torch",
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 1, #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    "num_workers": 1,  # parallelism
}

# stop = {
#     "training_iteration": args.stop_iters,
#     "timesteps_total": args.stop_timesteps,
#     "episode_reward_mean": args.stop_reward,
# }



if __name__ == "__main__":
    #args = parser.parse_args()
    ray.init(num_gpus=1) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)

    # use fixed learning rate instead of grid search (needs tune)
    ppo_config["lr"] = 1e-3
    trainer = ppo.PPOTrainer(config=ppo_config, env=FactorySimEnv)
    # run manual training loop and print results after each iteration

    for _ in range(5000): #args.stop_iters,
        result = trainer.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if (
            result["timesteps_total"] >= args.stop_timesteps
            or result["episode_reward_mean"] >= args.stop_reward
        ):
            break




