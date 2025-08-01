import gymnasium as gym

from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune.registry import register_env
from ray import tune
import ray

parser = add_rllib_example_script_args(
    default_reward=20.0,
    default_timesteps=1000000,
)
parser.set_defaults(
    enable_new_api_stack=False,
    env="ale_py:ALE/Pong-v5",
)
args = parser.parse_args()


def _make_env_to_module_connector(env, spaces, device):
    return FrameStackingEnvToModule(num_frames=4)


def _make_learner_connector(input_observation_space, input_action_space):
    return FrameStackingLearner(num_frames=4)


def _env_creator(cfg):
    return wrap_atari_for_new_api_stack(
        gym.make(args.env, **cfg, **{"render_mode": "rgb_array"}),
        dim=64,
        framestack=None,
    )


register_env("env", _env_creator)


config = (
    APPOConfig()
    .environment(
        "env",
        env_config={
            # Make analogous to old v4 + NoFrameskip.
            "frameskip": 1,
            "full_action_space": False,
            "repeat_action_probability": 0.0,
        },
        clip_rewards=True,
    )
    .env_runners(
        env_to_module_connector=_make_env_to_module_connector,
        num_envs_per_env_runner=2,
        num_env_runners=5,
       
    )
    .learners(
        num_learners=2,
        #num_aggregator_actors_per_learner=0,
        num_gpus_per_learner=0,
        num_cpus_per_learner=2
    )
    .training(
        learner_connector=_make_learner_connector,
        train_batch_size_per_learner=500,
        target_network_update_freq=2,
        lr=0.0005 * ((args.num_learners or 1) ** 0.5),
        vf_loss_coeff=tune.quniform(0.0, 1.0, 0.2),                  #1.0,
        entropy_coeff=[[0, 0.01], [3000000, 0.0]],  # <- crucial parameter to finetune
        # Only update connector states and model weights every n training_step calls.
        broadcast_interval=5,
        # learner_queue_size=1,
        circular_buffer_num_batches=4,
        circular_buffer_iterations_per_batch=2,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            vf_share_layers=True,
            conv_filters=[(16, 4, 2), (32, 4, 2), (64, 4, 2), (128, 4, 2)],
            conv_activation="relu",
            head_fcnet_hiddens=[256],
        )
    )
)



@ray.remote(num_gpus=1)
def check_env():
    import os
    return os.environ.get("NCCL_P2P_DISABLE")




if __name__ == "__main__":
    from ray.rllib.utils.test_utils import run_rllib_example_script_experiment

    print(ray.get(check_env.remote()))
    run_rllib_example_script_experiment(config, args)

#to run multiple parallel trials, use:
# ray.tune.run(
#     "APPO",
#     config=config.to_dict(),
#     stop={"training_iteration": 100},
#     num_samples=10,
#     local_dir="ray_results",
#     checkpoint_freq=10,
#     checkpoint_at_end=True,
#     resources_per_trial={"cpu": 1, "gpu": 1},
#     verbose=1,
#     name="pong_experiment",
#     keep_checkpoints_num=5,
#     checkpoint_score_attr="episode_reward_mean",
#     metric="episode_reward_mean",
#     mode="max",
#     trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
#     log_to_file=True,
#     progress_reporter=None,
#     reuse_actors=True,