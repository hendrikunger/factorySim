
import pprint
import numpy as np
import sys

from pathlib import Path
import os
from env.factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv
import gymnasium as gym

import ray

from ray.tune import Tuner, Callback
from ray.air.config import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.core.rl_module.rl_module import RLModule, DefaultModelConfig
#from factorySim.customRLModulTorch import MyPPOTorchRLModule
#from factorySim.customRLModulTF import MyXceptionRLModule
#from factorySim.customModelsTorch import MyXceptionModel


from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict
from ray.rllib.evaluation.episode_v2 import EpisodeV2



from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.connectors import ObservationPreprocessor
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID

import wandb
import yaml
from  typing import Any
from datetime import datetime



#filename = "Overlapp"
filename = "Basic"
#filename = "EP_v23_S1_clean"
#filename = "Simple"
#filename = "SimpleNoCollisions"
#filename = "LShape"

#ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1", filename + ".ifc")
ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1")

#Import Custom Models
from ray.rllib.models import ModelCatalog
#ModelCatalog.register_custom_model("my_model", MyXceptionModel)

NO_TUNE = True


class MyAlgoCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, Any] = None):
        #super().__init__()    
        self.ratingkeys = ['Reward', 'TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends',]

    # def on_episode_start(
    #     self,
    #     *,
    #     episode: Union[EpisodeType, Episode, EpisodeV2],
    #     env_runner: Optional["EnvRunner"] = None,
    #     metrics_logger: Optional[MetricsLogger] = None,
    #     env: Optional[gym.Env] = None,
    #     env_index: int,
    #     rl_module: Optional[RLModule] = None,
    #     worker: Optional["EnvRunner"] = None,
    #     base_env: Optional[BaseEnv] = None,
    #     policies: Optional[Dict[PolicyID, Policy]] = None,
    #     **kwargs,
    # ) -> None: 
    #     pass
  

    # def on_episode_step(
    #     self,
    #     *,
    #     episode: Union[EpisodeType, Episode, EpisodeV2],
    #     env_runner: Optional["EnvRunner"] = None,
    #     metrics_logger: Optional[MetricsLogger] = None,
    #     env: Optional[gym.Env] = None,
    #     env_index: int,
    #     rl_module: Optional[RLModule] = None,
    #     worker: Optional["EnvRunner"] = None,
    #     base_env: Optional[BaseEnv] = None,
    #     policies: Optional[Dict[PolicyID, Policy]] = None,
    #     **kwargs,
    # ) -> None:
        
    #     if env_runner.config["env_config"]["evaluation"]:
    #         pass


    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        
        if env_runner.config["env_config"]["evaluation"]:
            infos = episode.get_infos()
            #Save as a dict with key "myData" and the evalEnvID as subkey, so different episodes can be parsed later
            metrics_logger.log_value(("myData",infos[0].get('evalEnvID', 0)+1), infos, reduce=None, clear_on_reduce=True)

        
    def on_evaluate_start(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        print(f"--------------------------------------------EVAL START--------------------------------------------")




    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:


        print(f"--------------------------------------------EVAL END--------------------------------------------")
        #This is a dict with the evalEnvID as key and the infos of all steps in array form as value
        data = evaluation_metrics["env_runners"]["myData"]

        tbl = wandb.Table(columns=["id", "episode","evalEnvID", "Image"] + self.ratingkeys)
        if data:
            #iterate over all eval episodes
            for episode_id, episode in data.items():
                #episode is a list of dicts, each dict is a step info
                for info in episode:
                    step = info.get("Step", -1)
                    image = info.get("Image", np.random.randint(low=0, high=255, size=(100, 100, 3)))
                    caption = f"{episode_id}_{step:04d}"
                    logImage = wandb.Image(image, caption=caption, grouping=int(episode_id))
                    rating = [info.get(key, -1) for key in self.ratingkeys]
                    tbl.add_data(f"{episode_id}_{step}", episode_id, info.get("evalEnvID", -1), logImage, *rating)
            evaluation_metrics["table"] = tbl



class NormalizeObservations(ObservationPreprocessor):
    def preprocess(self, observation: Dict[AgentID, Dict[str, np.ndarray]]) -> Dict[AgentID, Dict[str, np.ndarray]]:

        return observation / 255.0


       
def _env_to_module(env):
# Create the env-to-module connector pipeline.
    return NormalizeObservations()



with open('config.yaml', 'r') as f:
    f_config = yaml.load(f, Loader=yaml.FullLoader)

#f_config['env'] = FactorySimEnv
f_config['env_config']['inputfile'] = ifcpath

# myRLModule = SingleAgentRLModuleSpec(
#     module_class=MyPPOTorchRLModule,
#     model_config_dict={"model":"resnet34", "pretrained": False},
# )




def run():
    runtime_env = {
    "env_vars": {"PYTHONWARNINGS": "ignore::UserWarning"},
    "working_dir": os.path.join(os.path.dirname(os.path.realpath(__file__))),
    "excludes": ["/.git", "/.vscode", "/wandb", "/artifacts", "*.skp"]
    }
    NUMGPUS = int(os.getenv("$SLURM_GPUS", 0 if sys.platform == "darwin" else 1))
    ray.init(num_gpus=NUMGPUS, include_dashboard=False, runtime_env=runtime_env) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))

    stop = {
    "training_iteration": 20,
    #"num_env_steps_sampled_lifetime": 15000000,
    #"episode_reward_mean": 5,
    }

    checkpoint_config = CheckpointConfig(checkpoint_at_end=True, 
                                         checkpoint_frequency=50, 
                                         checkpoint_score_order="max", 
                                         checkpoint_score_attribute="episode_reward_mean", 
                                         num_to_keep=5 
    )

    ppo_config = PPOConfig()
    ppo_config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    ppo_config.training(
                    train_batch_size=f_config['train_batch_size_per_learner'],
                    minibatch_size=f_config['mini_batch_size_per_learner'],


    ) 
    #ppo_config.lr=0.00005
                 #0.003
                 #0.000005
    ppo_config.rl_module(model_config=DefaultModelConfig(),
                         model_config=DefaultModelConfig(
                                        #Input is 84x84x2 output needs to be [B, X, 1, 1] for PyTorch), where B=batch and X=last Conv2D layer's number of filters

                                        conv_filters= [
                                                        (32, 8, 4),  # Reduces spatial size from 84x84 -> 20x20
                                                        (64, 4, 2),  # Reduces spatial size from 20x20 -> 9x9
                                                        (128, 3, 1),  # Reduces spatial size from 9x9 -> 7x7
                                                        (256, 7, 1),  # Reduces spatial size from 7x7 -> 1x1
                                                    ],
                                        conv_activation="relu",
                                        post_fcnet_hiddens=[256],
                                        vf_share_layers=True,
                                    ),
                        #rl_module_spec=myRLModule,
                         )
        


    ppo_config.environment(FactorySimEnv, env_config=f_config['env_config'], render_env=False)

    ppo_config.callbacks(MyAlgoCallback)
    ppo_config.env_runners(#num_env_runners=int(os.getenv("SLURM_CPUS_PER_TASK", f_config['num_workers']))-1,  #f_config['num_workers'], 
                        num_env_runners=0,
                        num_envs_per_env_runner=1,  #2
                        enable_connectors=True,
                        env_to_module_connector=_env_to_module,
                        )
    #ppo_config.train_batch_size=256
    ppo_config.framework(framework="torch",
                         )

    eval_config = f_config['evaluation_config']["env_config"]
    eval_config['inputfile'] = ifcpath

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation") 

    eval_duration = len([x for x in os.listdir(path) if ".ifc" in x])
    print(f"---->Eval Duration: {eval_duration}")
    ppo_config.evaluation(evaluation_duration=eval_duration,
                          evaluation_duration_unit="episodes", 
                          evaluation_interval=f_config["evaluation_interval"],
                          evaluation_config={"env_config": eval_config},
                          evaluation_parallel_to_training=False,
                        )   
    ppo_config.learners( num_learners=NUMGPUS,
                         num_gpus_per_learner=0 if sys.platform == "darwin" else 1,
                         )


    

    name = os.getenv("SLURM_JOB_ID", f"D-{datetime.now().strftime('%Y%m%d_%H-%M-%S')}")


    run_config=RunConfig(name="bert",
                            stop=stop,
                            checkpoint_config=checkpoint_config,
                            #log_to_file="./wandb/latest-run/files/stdoutanderr.log",
                            callbacks=[
                                WandbLoggerCallback(project="factorySim_TRAIN",
                                                    log_config=True,
                                                    upload_checkpoints=True,
                                                    name=name,
                                                    group="default",
                                                    ),
                                #MyCallback(),
                        ],
                        )
    


    

    path = Path.joinpath(Path.home(), "ray_results/klaus")
    #path = "/home/unhe/gitRepo/factorySim/artifacts/checkpoint_PPO_FactorySimEnv_038cc_00000:v5"

    if Tuner.can_restore(path):
        print("--------------------------------------------------------------------------------------------------------")
        print(f"Restoring from {path.as_posix()}")
        print("--------------------------------------------------------------------------------------------------------")
        #Continuing training

        tuner = Tuner.restore(path.as_posix(), "PPO", param_space=ppo_config)
        results = tuner.fit() 

    else:
        if NO_TUNE:
            algo = ppo_config.build()
            for i in range(stop.get("training_iteration",2)):
                results = algo.train()
                if "envrunners" in results:
                    mean_return = results["env_runners"].get(
                        "episode_return_mean", np.nan
                    )
                    print(f"iter={i} R={mean_return}", end="")
                if "evaluation" in results:
                    Reval = results["evaluation"]["envrunners"][
                        "episode_return_mean"
                    ]
                    print(f" R(eval)={Reval}", end="")
                print()

        tuner = Tuner("PPO", run_config=run_config, param_space=ppo_config)
        results = tuner.fit()


    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")




    ray.shutdown()



# std log and std error need to go to wandb, they are in the main folder of the run


if __name__ == "__main__":
    from gymnasium import logger
    logger.set_level(logger.ERROR)
    run()