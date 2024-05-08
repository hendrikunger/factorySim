


from pathlib import Path
import os
from env.factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv

import ray

from ray.tune import Tuner, Callback
from ray.air.config import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MemoryTrackingCallbacks

from factorySim.customRLModulTorch import MyPPOTorchRLModule
#from factorySim.customRLModulTF import MyXceptionRLModule
from factorySim.customModelsTorch import MyXceptionModel


from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict

from ray.rllib.connectors.agent.lambdas import register_lambda_agent_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType


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
ModelCatalog.register_custom_model("my_model", MyXceptionModel)




class MyAlgoCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, Any] = None):
        super().__init__(legacy_callbacks_dict)    
        self.ratingkeys = ['TotalRating', 'ratingCollision', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability', 'routeContinuity', 'routeWidthVariance', 'Deadends',]

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
        episode.media["tabledata"]["currentStep"] = []
        episode.media["tabledata"]["evalEnvID"] = []

        #this should give us the initial info dict, but it is empty
        #info = episode._last_infos['agent0']

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
            episode.media["tabledata"]["currentStep"] += [info.get('Step', -1)]
            episode.media["tabledata"]["evalEnvID"] += [f"{info.get('evalEnvID', 0)}_{info.get('Step', -1)}"]
      


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
        algorithm,
        **kwargs,
    ):
        print(f"--------------------------------------------EVAL START")



    def on_evaluate_end(
        self,
        *,
        algorithm,
        evaluation_metrics: dict,
        **kwargs,
    ):


        print(f"--------------------------------------------EVAL END")

        data = evaluation_metrics["evaluation"]["episode_media"].pop("tabledata", None)

        tbl = wandb.Table(columns=["id", "episode","evalEnvID", "image"] + self.ratingkeys)
        if data:
            for episode_id, episode in enumerate(data):
                for step, image, caption , rating, evalEnvID in zip(episode["currentStep"], episode["images"], episode["captions"], episode["ratings"], episode["evalEnvID"]):
                    logImage = wandb.Image(image, caption=caption, grouping=episode_id) 
                    tbl.add_data(f"{episode_id}_{step}", episode_id, evalEnvID, logImage, *rating)

            evaluation_metrics["evaluation"]["episode_media"]["Eval_Table"] = tbl


       

    


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        pass
        #print(f"Got result: {result}")



def preprocessObs(data: Dict[str, TensorStructType]) -> Dict[str, TensorStructType]:
    #data[SampleBatch.NEXT_OBS] = data[SampleBatch.NEXT_OBS][:-2]
    print(data[SampleBatch.NEXT_OBS].shape)
    print(data[SampleBatch.NEXT_OBS].max())
    print(data[SampleBatch.NEXT_OBS].max())
    return data

MyAgentConnector = register_lambda_agent_connector(
    "MyAgentConnector", preprocessObs
)

#policy.agent_connectors.prepend(MyAgentConnector(ctx))


with open('config.yaml', 'r') as f:
    f_config = yaml.load(f, Loader=yaml.FullLoader)

#f_config['env'] = FactorySimEnv
f_config['env_config']['inputfile'] = ifcpath

myRLModule = SingleAgentRLModuleSpec(
    module_class=MyPPOTorchRLModule,
    model_config_dict={"model":"resnet34", "pretrained": False},
)




def run():
    ray.init(num_gpus=int(os.getenv("$SLURM_GPUS", "1")), include_dashboard=False) #int(os.environ.get("RLLIB_NUM_GPUS", "0"))

    stop = {
    #"training_iteration": 2,
    "timesteps_total": 15000000,
    #"episode_reward_mean": 5,
    }

    checkpoint_config = CheckpointConfig(checkpoint_at_end=True, 
                                         checkpoint_frequency=10, 
                                         checkpoint_score_order="max", 
                                         checkpoint_score_attribute="episode_reward_mean", 
                                         num_to_keep=5 
    )

    ppo_config = PPOConfig()
    ppo_config.experimental(_enable_new_api_stack=False)
    ppo_config.train_batch_size=200
    ppo_config.lr=0.00005
                 #0.003
                 #0.000005
    #ppo_config.rl_module(rl_module_spec=myRLModule,))
        

    ppo_config.training(model={
                                        "use_attention": False,
                                        "use_lstm": False,
                                        #"custom_model": "my_model",

                                    },
                        )
    ppo_config.environment(FactorySimEnv, env_config=f_config['env_config'], render_env=False)

    ppo_config.callbacks(MyAlgoCallback)
    #ppo_config.callbacks(MemoryTrackingCallbacks)
    ppo_config.rollouts(num_rollout_workers=int(os.getenv("SLURM_CPUS_PER_TASK", f_config['num_workers']))-1,  #f_config['num_workers'], 
                        num_envs_per_worker=1,  #2
                        )
    #ppo_config.train_batch_size=256
    ppo_config.framework(framework="torch",
                         eager_tracing=False,)

    eval_config = f_config['env_config'].copy()
    eval_config['evaluation'] = True
    eval_config['render_mode'] = "rgb_array"
    eval_config['randomSeed'] = 42 
    eval_config['createMachines'] = False

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation") 

    eval_duration = len([x for x in os.listdir(path) if ".ifc" in x])

    ppo_config.evaluation(evaluation_duration=eval_duration,
                          evaluation_duration_unit="episodes", 
                          evaluation_interval=1,
                          evaluation_config={"env_config": eval_config},
                        )   
    ppo_config.resources(num_gpus=int(os.getenv("$SLURM_GPUS", "1")),
                         num_learner_workers=1,
                         num_gpus_per_learner_worker=int(os.getenv("$SLURM_GPUS", "1")),
                         )
    ppo_config._disable_preprocessor_api=True
    ppo_config.rollouts(enable_connectors=True,)
    
    

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
        tuner = Tuner("PPO", run_config=run_config, param_space=ppo_config)
        results = tuner.fit()

    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")




    ray.shutdown()



# std log and std error need to go to wandb, they are in the main folder of the run


if __name__ == "__main__":
    run()