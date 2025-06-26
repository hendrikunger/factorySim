
import numpy as np
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union, Optional, Sequence
from pathlib import Path
import os
from env.factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv
import gymnasium as gym

import ray

from ray.tune import Tuner
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.air.config import RunConfig, CheckpointConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
#from ray.rllib.core.rl_module.rl_module import RLModule
#from factorySim.customRLModulTorch import MyPPOTorchRLModule
#from factorySim.customRLModulTF import MyXceptionRLModule
#from factorySim.customModelsTorch import MyXceptionModel

from ray.air.integrations.wandb import WandbLoggerCallback
from typing import Dict





from ray.rllib.connectors.env_to_module.observation_preprocessor import ObservationPreprocessor

from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from ray.tune.registry import register_env

import wandb
import yaml
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


def env_creator(env_config):
    return FactorySimEnv(env_config=env_config)  # return an env instance

register_env("FactorySimEnv", env_creator)

NO_TUNE = True
ALGO = "PPO"  # "Dreamer" or "PPO"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

class MyAlgoCallback(RLlibCallback):
    def __init__(self, env_runner_indices: Optional[Sequence[int]] = None):
        super().__init__()
        self.ratings = ['TotalRating', 'EvaluationResult', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'ratingCollision', 'routeContinuity', 'routeWidthVariance', 'Deadends', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability']

    def on_episode_start(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        if env_runner.config["env_config"]["evaluation"]:
            infos = episode.get_infos()
            for info in infos:
                episode_id = int(info.get('evalEnvID', 0)+1)
                print(f"Episode {episode_id} started")
                #delete old data
                metrics_logger.delete(("myData",episode_id), key_error=False)

  

    # def on_episode_step(
    #     self,
    #     *,
    #     episode,
    #     env_runner,
    #     metrics_logger,
    #     env,
    #     env_index,
    #     rl_module,
    #     **kwargs,
    # ) -> None:
        
    #     if env_runner.config["env_config"]["evaluation"]:
    #         pass


    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        

        # test You can base your custom logic on whether the calling EnvRunner is a regular “training” EnvRunner, used to collect training samples, or an evaluation EnvRunner, used to play through episodes for evaluation only. Access the env_runner.config.in_evaluation boolean flag, which is True on evaluation EnvRunner actors and False on EnvRunner actors used to collect training data.
        if env_runner.config["env_config"]["evaluation"]:
            infos = episode.get_infos()
            #Save as a dict with key "myData" and the evalEnvID as subkey, so different episodes can be parsed later
            for info in infos:
                episode_id = int(infos[0].get('evalEnvID', 0)+1)
                metrics_logger.log_dict(info, key=("myData",episode_id), reduce=None, clear_on_reduce=False)
                #Full Logging of all metrics
                for key, value in info.items():
                    if key in self.ratings:
                        metrics_logger.log_value(("myLogs",episode_id,key), value, reduce="mean", clear_on_reduce=True)


            
            
        
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

        
        if data:
            column_names = [key for key in next(iter(data.values()))]
            tbl = wandb.Table(columns=["id"] + column_names)
            #iterate over all eval episodes
            for episode_id, infos in data.items():
                #infos is a dict of all metrics each value is a list of the values of all steps
                for step in range(len(infos['Step'])):
                    row = []                     
                    row_id = f"{episode_id}_{infos['Step'][step]}"
                    row.append(row_id)
                    for key, values in infos.items():

                        value = values[step]
                        if key == "Image":
                            value = wandb.Image(value, caption=row_id, grouping=int(episode_id))
                        row.append(value)
                    tbl.add_data(*row)
            evaluation_metrics["table"] = tbl
            




class NormalizeObservations(ObservationPreprocessor):
    def preprocess(self, observation: Dict[AgentID, Dict[str, np.ndarray]]) -> Dict[AgentID, Dict[str, np.ndarray]]:

        return observation / 255.0


       
def _env_to_module(env=None, spaces=None, device=None) -> ObservationPreprocessor:
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
    ray.init(num_gpus=NUMGPUS, runtime_env=runtime_env) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))

    stop = {
    "training_iteration": 2,
    #"num_env_steps_sampled_lifetime": 15000000,
    #"episode_reward_mean": 5,
    }

    checkpoint_config = CheckpointConfig(checkpoint_at_end=True, 
                                         checkpoint_frequency=100, 
                                         checkpoint_score_order="max",
                                         checkpoint_score_attribute="env_runners/episode_return_mean", 
                                         num_to_keep=5 
    )

    match ALGO:
        case "PPO":
    
            algo_config = PPOConfig()
            algo_config.training(
                            train_batch_size=f_config['train_batch_size_per_learner'],
                            minibatch_size=f_config['mini_batch_size_per_learner'],


            ) 
            #algo_config.lr=0.00005
                        #0.003
                        #0.000005
            algo_config.rl_module(model_config=DefaultModelConfig(
                                                #Input is 84x84x2 output needs to be [B, X, 1, 1] for PyTorch), where B=batch and X=last Conv2D layer's number of filters

                                                conv_filters= [
                                                                [32, 8, 4],  # Reduces spatial size from 84x84 -> 20x20
                                                                [64, 4, 2],  # Reduces spatial size from 20x20 -> 9x9
                                                                [128, 3, 1],  # Reduces spatial size from 9x9 -> 7x7
                                                                [256, 7, 1],  # Reduces spatial size from 7x7 -> 1x1
                                                            ],
                                                conv_activation="relu",
                                                head_fcnet_hiddens=[256],
                                                vf_share_layers=True,
                                            ),
                                #rl_module_spec=myRLModule,
                                )
            algo_config.vf_loss_coeff=0.5   # Coefficient of the value function loss. IMPORTANT: you must tune this if you set vf_share_layers=True inside your model’s config.
        #PPO END ------------------------------------------------------------------------------------------------------
        case "Dreamer":
            algo_config = DreamerV3Config()

            #Dreamer needs 64x64x3 input
            f_config['env_config']['width'] = 64
            f_config['env_config']['height'] = 64

            w = algo_config.world_model_lr
            c = algo_config.critic_lr
            algo_config.training(
                model_size="M",
                training_ratio=512, #Should be lower for larger models e.g. 64 for XL  
                batch_size_B=16 * (NUMGPUS or 1),
                # Use a well established 4-GPU lr scheduling recipe:
                # ~ 1000 training updates with 0.4x[default rates], then over a few hundred
                # steps, increase to 4x[default rates].
                world_model_lr=[[0, 0.4 * w], [8000, 0.4 * w], [10000, 3 * w]],
                critic_lr=[[0, 0.4 * c], [8000, 0.4 * c], [10000, 3 * c]],
                actor_lr=[[0, 0.4 * c], [8000, 0.4 * c], [10000, 3 * c]],
            )
        #Dreamer END ------------------------------------------------------------------------------------------------------

    algo_config.environment("FactorySimEnv", env_config=f_config['env_config'], render_env=False)

    algo_config.callbacks(MyAlgoCallback)
    algo_config.debugging(logger_config={"type": "ray.tune.logger.NoopLogger"}) # Disable slow tbx logging
    algo_config.env_runners(num_env_runners=int(os.getenv("SLURM_CPUS_PER_TASK", f_config['num_workers']))-1,
                        num_envs_per_env_runner=1,  #2
                        num_cpus_per_env_runner=1,
                        env_to_module_connector=_env_to_module,
                        )
    algo_config.learners( num_learners=NUMGPUS,
                         num_gpus_per_learner=0 if sys.platform == "darwin" else 1,
                         )


    eval_config = f_config['evaluation_config']["env_config"]
    eval_config['inputfile'] = ifcpath

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation") 

    eval_duration = len([x for x in os.listdir(path) if ".ifc" in x])
    algo_config.evaluation(evaluation_duration=eval_duration,
                          evaluation_duration_unit="episodes", 
                          evaluation_interval=f_config["evaluation_interval"],
                          evaluation_config={"env_config": eval_config},
                          evaluation_parallel_to_training=False,
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

        tuner = Tuner.restore(path.as_posix(), "PPO", param_space=algo_config)
        results = tuner.fit() 

    else:
        if NO_TUNE:
            algo = algo_config.build_algo()
            for i in range(stop.get("training_iteration",2)):
                results = algo.train()
                if "envrunners" in results:
                    mean_return = results["env_runners"].get(
                        "episode_return_mean", np.nan
                    )
                    print(f"iter={i} R={mean_return}", end="")
                if "evaluation" in results:
                    Reval = results["evaluation"]["env_runners"]["agent_episode_returns_mean"]["default_agent"]
                    print(f" R(eval)={Reval}", end="")
                print()
        else:
            tuner = Tuner("PPO", run_config=run_config, param_space=algo_config)
            results = tuner.fit()


    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")




    ray.shutdown()



# std log and std error need to go to wandb, they are in the main folder of the run


if __name__ == "__main__":
    #from gymnasium import logger
    #logger.set_level(logger.ERROR)
    run()