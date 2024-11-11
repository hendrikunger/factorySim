from pathlib import Path
import os
from env.factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv
import gymnasium as gym


from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.rl_module.rl_module import RLModule

from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union, Any

import wandb
import yaml

from datetime import datetime

class MyAlgoCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, Any] = None):
        #super().__init__()    
        self.ratingkeys = ['Reward', 'TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends',]

    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        episode.media["tabledata"] = {}
        episode.media["tabledata"]["ratings"] = []
        episode.media["tabledata"]["images"] = []
        episode.media["tabledata"]["captions"] = []
        episode.media["tabledata"]["currentStep"] = []
        episode.media["tabledata"]["evalEnvID"] = []

        #this should give us the initial info dict, but it is empty
        #info = episode._last_infos['agent0']

    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:

        if "Evaluation" in episode._last_infos['agent0']:
            info = episode._last_infos['agent0']
            episode.media["tabledata"]["captions"] += [f"{episode.episode_id}_{info.get('Step', 0):04d}"]
            episode.media["tabledata"]["images"] += [info.get("Image", None)]
            episode.media["tabledata"]["ratings"] += [[info.get(key, -1) for key in self.ratingkeys]]
            episode.media["tabledata"]["currentStep"] += [info.get('Step', -1)]
            episode.media["tabledata"]["evalEnvID"] += [f"{info.get('evalEnvID', 0)}_{info.get('Step', -1)}"]
      


    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        
        if "Evaluation" in episode._last_infos['agent0']:
            info = episode._last_infos['agent0']
            for key in self.ratingkeys:
                episode.custom_metrics[key] = info.get(key, -1)
        else:
            episode.media.pop("tabledata", None)



        
    def on_evaluate_start(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        print(f"--------------------------------------------EVAL START")



    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:


        print(f"--------------------------------------------EVAL END")

        data = evaluation_metrics["episode_media"].pop("tabledata", None)


        tbl = wandb.Table(columns=["id", "episode","evalEnvID", "image"] + self.ratingkeys)
        if data:
            for episode_id, episode in enumerate(data):
                for step, image, caption , rating, evalEnvID in zip(episode["currentStep"], episode["images"], episode["captions"], episode["ratings"], episode["evalEnvID"]):
                    logImage = wandb.Image(image, caption=caption, grouping=episode_id) 
                    tbl.add_data(f"{episode_id}_{step}", episode_id, evalEnvID, logImage, *rating)

            evaluation_metrics["episode_media"]["Eval_Table"] = tbl


       


config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1")
    .env_runners(num_env_runners=20)
    .resources(num_gpus=1)
)

algo = config.build()

for i in range(10):
    result = algo.train()
    result.pop("config")
    pprint(result)

    if i % 5 == 0:
        checkpoint_dir = algo.save_to_path()
        print(f"Checkpoint saved in directory {checkpoint_dir}")