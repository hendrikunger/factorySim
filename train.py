import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path

from env.factorySim.factorySimEnv import FactorySimEnv, MultiFactorySimEnv

import ray

from ray import air, tune
from ray.tune import Tuner
from ray.tune import Callback
from ray.train.rl.rl_trainer import RLTrainer
from ray.air import Checkpoint
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.result import Result
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from factorySim.customModels import MyXceptionModel
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict
from ray.rllib.algorithms.algorithm import Algorithm

import datetime
import yaml
from  typing import Any

#filename = "Overlapp"
filename = "Basic"
#filename = "EP_v23_S1_clean"
#filename = "Simple"
#filename = "SimpleNoCollisions"
#filename = "LShape"

#ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1", filename + ".ifc")
ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1")

#Import Custom Models
ModelCatalog.register_custom_model("my_model", MyXceptionModel)

class MyAlgoCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, Any] = None):
        super().__init__(legacy_callbacks_dict)    
        self.ratingkeys = ['TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends','terminated',]

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index:  None,
        **kwargs,
    ):  
        episode.media["tabledata"] = {}
        episode.media["tabledata"]["ratings"] = []
        episode.media["tabledata"]["images"] = []
        episode.media["tabledata"]["captions"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        if "Evaluation" in episode._last_infos['agent0']:
            info = episode._last_infos['agent0']
            episode.media["tabledata"]["captions"] += [f"{episode.episode_id}_{info.get('Step', 0):04d}"]
            episode.media["tabledata"]["images"] += [info.get("Image", None)]
            episode.media["tabledata"]["ratings"] += [[info.get(key, -1) for key in self.ratingkeys]]
            


    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
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
        **kwargs,
    ):
        print(f"--------------------------------------------EVAL START")



    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        evaluation_metrics: dict,
        **kwargs,
    ):


        print(f"--------------------------------------------EVAL END")

        data = evaluation_metrics["evaluation"]["episode_media"].pop("tabledata", None)
        tbl = wandb.Table(columns=["image"] + self.ratingkeys)
        images = []
        if data:
            for episode_id, episode in enumerate(data):
                for image, caption , rating in zip(episode["images"], episode["captions"], episode["ratings"]):
                    logImage = wandb.Image(image, caption=caption, grouping=episode_id) 
                    images += [logImage]
                    tbl.add_data(logImage, *rating)

            evaluation_metrics["evaluation"]["episode_media"]["Eval_Table"] = tbl
            evaluation_metrics["evaluation"]["episode_media"]["Eval_Images"] = images

            print(evaluation_metrics)
       

    


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print("\n\n")
        #print(f"Got result: {result}")
        print("\n\n")


with open('config.yaml', 'r') as f:
    f_config = yaml.load(f, Loader=yaml.FullLoader)

#f_config['env'] = FactorySimEnv
f_config['env_config']['inputfile'] = ifcpath



if __name__ == "__main__":
    ray.init(num_gpus=1, include_dashboard=False) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))


    stop = {
    "training_iteration": 10,
    #"timesteps_total": 5000000,
    #"episode_reward_mean": 5,
    }

    checkpoint_config = CheckpointConfig(checkpoint_at_end=True, 
                                         checkpoint_frequency=10, 
                                         checkpoint_score_order="max", 
                                         checkpoint_score_attribute="episode_reward_mean", 
                                         num_to_keep=10 
    )

    ppo_config = PPOConfig()
    ppo_config.environment(FactorySimEnv, env_config=f_config['env_config'])
    ppo_config.training(model={"custom_model": "my_model"})    
    ppo_config.update_from_dict(f_config)
    ppo_config.callbacks(MyAlgoCallback)


    trainer = RLTrainer(
        run_config=RunConfig(name="klaus",
                                         stop=stop,
                                         checkpoint_config=checkpoint_config,
                                         log_to_file=True,
                                         callbacks=[
                                                WandbLoggerCallback(project="factorySimTest_Train",
                                                                    log_config=True,
                                                                    upload_checkpoints=False,
                                                                    save_checkpoints=False,
                                                                    ),
                                                MyCallback(),
                                        ],
                            ),
        scaling_config=ScalingConfig(num_workers=f_config['num_workers'], 
                                     use_gpu=True,
                                    ),
        algorithm="PPO",
        config=ppo_config.to_dict(),

    )

    path = Path.home() /"ray_results"
    print(path)
    if Tuner.can_restore(path):

        #Continuing training
        tuner = Tuner.restore(path, trainable=trainer)
        results = tuner.fit() 

    else:

        tuner = tune.Tuner(trainer)
        results = tuner.fit()

    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")



    ray.shutdown()

