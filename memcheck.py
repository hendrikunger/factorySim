import argparse
import os
import tracemalloc
from email.policy import Policy
from typing import Optional, Dict

import numpy as np
import psutil
import ray
from gym.spaces import Box
from ray import tune
from ray.rllib import BaseEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
#from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer


class TraceMallocCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()

        tracemalloc.start(10)

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, env_index: Optional[int] = None, **kwargs) -> None:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        for stat in top_stats[:5]:
            count = stat.count
            size = stat.size

            trace = str(stat.traceback)

            episode.custom_metrics[f'tracemalloc/{trace}/size'] = size
            episode.custom_metrics[f'tracemalloc/{trace}/count'] = count

        process = psutil.Process(os.getpid())
        worker_rss = process.memory_info().rss
        worker_data = process.memory_info().data
        worker_vms = process.memory_info().vms
        episode.custom_metrics[f'tracemalloc/worker/rss'] = worker_rss
        episode.custom_metrics[f'tracemalloc/worker/data'] = worker_data
        episode.custom_metrics[f'tracemalloc/worker/vms'] = worker_vms


def dim_to_gym_box(dim, val=np.inf):
    """Create gym.Box with specified dimension."""
    high = np.full((dim,), fill_value=val)
    return Box(low=-high, high=high)


class DummyMultiAgentEnv(MultiAgentEnv):
    """Return zero observations."""

    def __init__(self, config):
        del config  # Unused
        super(DummyMultiAgentEnv, self).__init__()
        self.config = dict(act_dim=17, obs_dim=380, n_players=2, n_steps=1000)
        self.players = ["player_%d" % p for p in range(self.config['n_players'])]
        self.current_step = 0
        self.observation_space = dim_to_gym_box(self.config['obs_dim'])
        self.action_space = dim_to_gym_box(self.config['act_dim'])

    def _obs(self):
        return np.zeros((self.config['obs_dim'],))

    def reset(self):
        self.current_step = 0
        return {p: self._obs() for p in self.players}

    def step(self, action_dict):
        done = self.current_step >= self.config['n_steps']
        self.current_step += 1

        obs = {p: self._obs() for p in self.players}
        rew = {p: np.random.random() for p in self.players}
        dones = {p: done for p in self.players + ["__all__"]}
        infos = {p: {'test_thing': 'wahoo'} for p in self.players}

        return obs, rew, dones, infos




def create_env(config):
    """Create the dummy environment."""
    return DummyMultiAgentEnv(config)


env_name = "DummyMultiAgentEnv"
register_env(env_name, create_env)


def get_trainer_config(env_config, train_policies, num_workers=9, framework="tf2"):
    """Build configuration for 1 run."""

    # trainer config
    config = {
        "env": env_name, "env_config": env_config, "num_workers": num_workers,
        # "multiagent": {"policy_mapping_fn": lambda x: x, "policies": policies,
        #               "policies_to_train": train_policies},
        "framework": framework,
        "train_batch_size": 512,

        'batch_mode': 'truncate_episodes',

        "callbacks": TraceMallocCallback,
        "lr": 0.0,
        "num_gpus": 1,
    }
    return config


def tune_run():
    parser = argparse.ArgumentParser(description='Run experiments')

    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--yaml-file', help='YAML file containing GDY for the game')
    parser.add_argument('--root-directory', default=os.path.expanduser("~/ray_results"))

    args = parser.parse_args()

    #wandbLoggerCallback = WandbLoggerCallback(
    #    project='ma_mem_leak_exp',
    #    api_key_file='~/.wandb_rc',
    #    dir=args.root_directory
    #)

    ray.init(ignore_reinit_error=True, num_gpus=1, include_dashboard=False)
    config = get_trainer_config(train_policies=['player_1', 'player_2'], env_config={})
    return tune.run(PPOTrainer,
                    config=config)
                    #callbacks=[wandbLoggerCallback])


if __name__ == '__main__':
    tune_run()