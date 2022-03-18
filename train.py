#!/usr/bin/env python3
import argparse
import os

from factorySim.factorySimEnv import FactorySimEnv, MultiFactorySimEnv

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
    "env": MultiFactorySimEnv,  
    "env_config": {
            "inputfile" : ifcpath,
            "obs_type" : 'image',
            "Loglevel" : 0,
            "width" : 84,
            "heigth" : 84,
            "maxMF_Elements" : 5,
            "outputScale" : 4,
            "objectScaling" : 0.5,
            "rendermode" : None,
        },
    "model": {
        "dim" : 256,
        # Change individual keys in that dict by overriding them, e.g.
        "conv_filters": None,
        "conv_activation": "relu",
        # == Attention Nets (experimental: torch-version is untested) ==
        # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
        # wrapper Model around the default Model.
        "use_attention": False,
        # The number of transformer units within GTrXL.
        # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
        # b) a position-wise MLP.
        "attention_num_transformer_units": 1,
        # The input and output size of each transformer unit.
        "attention_dim": 64,
        # The number of attention heads within the MultiHeadAttention units.
        "attention_num_heads": 1,
        # The dim of a single head (within the MultiHeadAttention units).
        "attention_head_dim": 32,
        # The memory sizes for inference and training.
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        # The output dim of the position-wise MLP.
        "attention_position_wise_mlp_dim": 32,
        # The initial bias values for the 2 GRU gates within a transformer unit.
        "attention_init_gru_gate_bias": 2.0,
        # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
        "attention_use_n_prev_actions": 0,
        # Whether to feed r_{t-n:t-1} to GTrXL.
        "attention_use_n_prev_rewards": 0,
    },
    # Should be one of DEBUG, INFO, WARN, or ERROR
    "log_level": "WARN",
    # Evaluate once per training iteration.
    "evaluation_interval": 20,
    # Run evaluation on (at least) ten episodes
    "evaluation_duration": 3,
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
        #"record_env": os.path.join(os.path.dirname(os.path.realpath(__file__)), "Output"),
        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
        # env and only the 1st sub-env in a vectorized env.
        "render_env": True,
        "env_config": {
            "rendermode" : "human",

        },
    },
    #"tf", "tf2", "tfe", "torch"
    "render_env": False,
    "framework": "tf2",
    "eager_tracing": True,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 1, #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    "num_workers": 12,  # parallelism
    "num_envs_per_worker": 4,
    "rollout_fragment_length" : 20,
    "train_batch_size": 4800,
    "sgd_minibatch_size": 100,
    "num_sgd_iter": 4,
    "gamma": 0.99,
    # The default learning rate.
    "lr": 0.00025, #0.00002,
    "lambda": 0.95,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "clip_param": 0.2,
    "vf_clip_param": 10.0,
    "kl_target": 0.01,
}

# stop = {
#     "training_iteration": args.stop_iters,
#     "timesteps_total": args.stop_timesteps,
#     "episode_reward_mean": args.stop_reward,
# }



if __name__ == "__main__":
    #args = parser.parse_args()
    ray.init(num_gpus=1, local_mode=False) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)



    # use fixed learning rate instead of grid search (needs tune)
    #ppo_config["lr"] = 1e-3
    trainer = ppo.PPOTrainer(config=ppo_config)
    # run manual training loop and print results after each iteration

    for _ in range(5000000): #args.stop_iters,
        result = trainer.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if (
            result["timesteps_total"] >= 2000000#args.stop_timesteps
            or result["episode_reward_mean"] >= 50000 #args.stop_reward
        ):
            break

    ray.shutdown()


