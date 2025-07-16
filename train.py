import numpy as np
import sys

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union, Optional, Sequence
from pathlib import Path
import os
import platform
from env.factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv

import ray

from ray.tune import Tuner
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.tune import RunConfig, CheckpointConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
#from ray.rllib.core.rl_module.rl_module import RLModule
#from factorySim.customRLModulTorch import MyPPOTorchRLModule
#from factorySim.customRLModulTF import MyXceptionRLModule
#from factorySim.customModelsTorch import MyXceptionModel

from ray.air.integrations.wandb import WandbLoggerCallback
from typing import Dict

import pprint


from ray.rllib.connectors.env_to_module.observation_preprocessor import SingleAgentObservationPreprocessor

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
    env = FactorySimEnv(env_config=env_config)

    return  env # return an env instance

register_env("FactorySimEnv", env_creator)

NO_TUNE = False
ALGO = "APPO"  # "Dreamer" or "PPO" or "APPO"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"


#Callbacks----------------------------------------------------------------------------------------------------------------------------------------------------------

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
    #     episode : SingleAgentEpisode,
    #     env_runner,
    #     metrics_logger,
    #     env,
    #     env_index,
    #     rl_module,
    #     **kwargs,
    # ) -> None:
        
    #     if env_runner.config["env_config"]["evaluation"]:
    #         episode.custom_data["Experiment"] = f"{len(episode)}"


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
                
        if env_runner.config["env_config"]["evaluation"]:
            infos = episode.get_infos()
            episode_id = str(int(infos[0].get('evalEnvID', 0)+1))
            #Save as a dict with key "myData" and the evalEnvID as subkey, so different episodes can be parsed later

            for info in infos:
                metrics_logger.log_dict(info, key=("myData",episode_id), reduce=None, clear_on_reduce=True)
                #Full Logging of all metrics
                for key, value in info.items():
                    if key in self.ratings:
                        metrics_logger.log_value(("means",episode_id,key), value, reduce="mean", clear_on_reduce=True)

            


            
            
        
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

        evaluation_metrics["env_runners"].pop("myData", None)

        #pprint.pp(metrics_logger.stats)
        #Workaround for the fact that the metrics_logger does not respect the reduce= None setting when having nested keys


        data = {}


        myData = metrics_logger.peek(('evaluation','env_runners', 'myData'), compile=False)
        episodes = list(myData.keys())
        column_names = list(myData["1"].keys())
        for index in episodes:
            data[index] = {}
            for key in column_names:
                data[index][key] = metrics_logger.peek(('evaluation','env_runners', 'myData', index, key), compile=False)

        
        #num_iterations = int(evaluation_metrics["env_runners"]['num_episodes_lifetime']/len(episodes))
        
        if data:
            #column_names = [key for key in next(iter(data.values()))]
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

        del(myData)
        del(data)
            
#Preprocessor----------------------------------------------------------------------------------------------------------------------------------------------------------



class NormalizeObservations(SingleAgentObservationPreprocessor):
    def preprocess(self, observation: Dict[AgentID, Dict[str, np.ndarray]], episode: SingleAgentEpisode) -> Dict[AgentID, Dict[str, np.ndarray]]:
        output= observation / 255.0
        return output.astype(np.float32)


       
def _env_to_module(env=None, spaces=None, device=None) -> SingleAgentObservationPreprocessor:
# Create the env-to-module connector pipeline.
    return NormalizeObservations()

#RL Module----------------------------------------------------------------------------------------------------------------------------------------------------------



# myRLModule = SingleAgentRLModuleSpec(
#     module_class=MyPPOTorchRLModule,
#     model_config_dict={"model":"resnet34", "pretrained": False},
# )





#Main Function----------------------------------------------------------------------------------------------------------------------------------------------------------

def run():

    with open('config.yaml', 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)

    #f_config['env'] = FactorySimEnv
    f_config['env_config']['inputfile'] = ifcpath


    runtime_env = {
    "env_vars": {"PYTHONWARNINGS": "ignore::UserWarning"},
    "working_dir": os.path.join(os.path.dirname(os.path.realpath(__file__))),
    "excludes": ["/.git",
                "/.vscode",
                "/wandb",
                "/artifacts",
                "*.skp",
                "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/.git/",
                "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/.vscode/",
                "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/wandb/",
                "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/artifacts/"],
    }
    NUMGPUS = int(os.getenv("$SLURM_GPUS",
                0 if sys.platform == "darwin" else 1))
    
  
    if "SLURM_JOB_ID" in os.environ or sys.platform == "darwin" or platform.node() == "pop-os":
        ray.init(num_gpus=NUMGPUS, runtime_env=runtime_env) 
    else:
        #we are running on ray cluster
        runtime_env["py_modules"] = ["env/factorySim"]
        runtime_env["env_vars"] = {
            "PYTHONWARNINGS": "ignore::UserWarning",
        }
        ray.init(runtime_env=runtime_env)
        NUMGPUS = 1




    stop = {
    "training_iteration": f_config.get("training_iteration", 2), #Number of training iterations
    #"num_env_steps_sampled_lifetime": 15000000,
    #"episode_reward_mean": 5,
    }

    checkpoint_config = CheckpointConfig(checkpoint_at_end=True, 
                                         checkpoint_frequency=100, 
                                         checkpoint_score_order="max",
                                         #checkpoint_score_attribute="env_runners/episode_return_mean", 
                                         num_to_keep=5 
    )

    match ALGO:
        case "PPO":
    
            algo_config = PPOConfig()
            algo_config.training(
                            vf_loss_coeff= f_config['vf_loss_coeff'],
                            train_batch_size=f_config['train_batch_size_per_learner'],
                            minibatch_size=f_config['mini_batch_size_per_learner'],

            ) 

            algo_config.rl_module(model_config=DefaultModelConfig(
                                                #Input is 84x84x2 output needs to be [B, X, 1, 1] for PyTorch), where B=batch and X=last Conv2D layer's number of filters
                                                
                                                conv_filters= [# [ num_filters, kernel, stride]
                                                                [32, 5, 2],   # 128x128 → 62x62
                                                                [64, 4, 2],   # 62x62 → 30x30
                                                                [128, 4, 2],  # 30x30 → 14x14
                                                                [128, 3, 2],  # 14x14 → 6x6
                                                                [256, 3, 2],  # 6x6   → 2x2
                                                                [256, 2, 2],  # 2x2   → 1x1   

                                                                # [32, 8, 4],  # Reduces spatial size from 84x84 -> 20x20
                                                                # [64, 4, 2],  # Reduces spatial size from 20x20 -> 9x9
                                                                # [128, 3, 1],  # Reduces spatial size from 9x9 -> 7x7
                                                                # [256, 7, 1],  # Reduces spatial size from 7x7 -> 1x1
                                                            ],
                                                conv_activation="relu",
                                                head_fcnet_hiddens=[256],
                                                vf_share_layers=True,   #Need to tune  vf_loss_coeff on Training config if you set this to True

                                            ),
                                #rl_module_spec=myRLModule,
                                )
        #PPO END ------------------------------------------------------------------------------------------------------
        case "APPO":
            algo_config = APPOConfig()


            algo_config.training(
                train_batch_size=f_config['train_batch_size_per_learner'],  # Note: [1] uses 32768.
                circular_buffer_num_batches=16,  # matches [1]
                circular_buffer_iterations_per_batch=20,  # Note: [1] uses 32 for HalfCheetah.
                target_network_update_freq=2,
                target_worker_clipping=2.0,  # matches [1]
                clip_param=0.4,  # matches [1]
                num_gpu_loader_threads=1,
                # Note: The paper does NOT specify, whether the 0.5 is by-value or
                # by-global-norm.
                grad_clip=0.5,
                grad_clip_by="value",
                lr=0.0005,  # Note: [1] uses 3e-4.
                vf_loss_coeff=0.5,  # matches [1]
                gamma=0.995,  # matches [1]
                lambda_=0.995,  # matches [1]
                entropy_coeff=0.0,  # matches [1]
                use_kl_loss=True,  # matches [1]
                kl_coeff=1.0,  # matches [1]
                kl_target=0.04,  # matches [1]
            )


            algo_config.rl_module(model_config=DefaultModelConfig(
                                                #Input is 84x84x2 output needs to be [B, X, 1, 1] for PyTorch), where B=batch and X=last Conv2D layer's number of filters
                                                
                                                conv_filters= [# [ num_filters, kernel, stride]
                                                                [32, 5, 2],   # 128x128 → 62x62
                                                                [64, 4, 2],   # 62x62 → 30x30
                                                                [128, 4, 2],  # 30x30 → 14x14
                                                                [128, 3, 2],  # 14x14 → 6x6
                                                                [256, 3, 2],  # 6x6   → 2x2
                                                                [256, 2, 2],  # 2x2   → 1x1   

                                                                # [32, 8, 4],  # Reduces spatial size from 84x84 -> 20x20
                                                                # [64, 4, 2],  # Reduces spatial size from 20x20 -> 9x9
                                                                # [128, 3, 1],  # Reduces spatial size from 9x9 -> 7x7
                                                                # [256, 7, 1],  # Reduces spatial size from 7x7 -> 1x1
                                                            ],
                                                conv_activation="relu",
                                                head_fcnet_hiddens=[256],
                                                vf_share_layers=True,   #Need to tune  vf_loss_coeff on Training config if you set this to True

                                            ),
                                        )
        #Dreamer END ------------------------------------------------------------------------------------------------------
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
    algo_config.learners(num_learners= 0 if NUMGPUS <= 1 else NUMGPUS,  
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
                          evaluation_parallel_to_training=f_config["evaluation_parallel_to_training"],
                          evaluation_num_env_runners=f_config.get("evaluation_num_env_runners", 0), #Number of env runners for evaluation
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
                print(f"Training iteration {i+1}/{stop.get('training_iteration',2)}")
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
            tuner = Tuner(algo_config.algo_class, run_config=run_config, param_space=algo_config)
            results = tuner.fit()

        print("Training finished")


    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")




# std log and std error need to go to wandb, they are in the main folder of the run


if __name__ == "__main__":

    run()



# Todo: CoordConv: "An intriguing failing of CNNs and the CoordConv solution" (Liu et al., 2018).  Add coordinates to the input image, so that the model can learn to use them.