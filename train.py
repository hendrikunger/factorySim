




from pathlib import Path
import os
from env.factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv

import ray

from ray.tune import Tuner, Callback
from ray.air.config import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from factorySim.customRLModulTorch import MyPPOTorchRLModule
#from factorySim.customRLModulTF import MyXceptionRLModule



from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict

import wandb
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
#ModelCatalog.register_custom_model("my_model", MyXceptionModel)




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
        tbl = wandb.Table(columns=["image"] + self.ratingkeys)
        if data:
            for episode_id, episode in enumerate(data):
                for image, caption , rating in zip(episode["images"], episode["captions"], episode["ratings"]):
                    logImage = wandb.Image(image, caption=caption, grouping=episode_id) 
                    tbl.add_data(logImage, *rating)

            evaluation_metrics["evaluation"]["episode_media"]["Eval_Table"] = tbl


       

    


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        pass
        #print(f"Got result: {result}")



with open('config.yaml', 'r') as f:
    f_config = yaml.load(f, Loader=yaml.FullLoader)

#f_config['env'] = FactorySimEnv
f_config['env_config']['inputfile'] = ifcpath

myRLModule = SingleAgentRLModuleSpec(
    module_class=MyPPOTorchRLModule,
    model_config_dict={"model":"resnet34", "pretrained": False},
)




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
    ppo_config.exploration_config={}
    ppo_config.lr=0.00001
                 #0.000005
    ppo_config.rl_module(_enable_rl_module_api=True,)
                            #rl_module_spec=myRLModule,)
    ppo_config.training(_enable_learner_api=True,)
    ppo_config.environment(FactorySimEnv, env_config=f_config['env_config'], render_env=False)
    #ppo_config.update_from_dict(f_config)
    ppo_config.callbacks(MyAlgoCallback)
    ppo_config.rollouts(num_rollout_workers=22,  #f_config['num_workers'], 
                        num_envs_per_worker=1,  #2
                        )
    #ppo_config.train_batch_size=256
    ppo_config.framework(framework="torch",
                         eager_tracing=False,)

    eval_config = f_config['env_config'].copy()
    eval_config['evaluation'] = True
    eval_config['render_mode'] = "rgb_array"
    ppo_config.evaluation(evaluation_duration=10,
                          evaluation_duration_unit="episodes", 
                          evaluation_interval=1,
                          evaluation_config={"env_config": eval_config},
                        )   
    ppo_config.resources(num_learner_workers=0,
                         num_gpus_per_learner_worker=1,
                         )
    

    #my_ppo = ppo_config.build(use_copy=False)

    run_config=RunConfig(name="klaus",
                            stop=stop,
                            checkpoint_config=checkpoint_config,
                            #log_to_file="./wandb/latest-run/files/stdoutanderr.log",
                            callbacks=[
                                WandbLoggerCallback(project="factorySim_TRAIN",
                                                    log_config=True,
                                                    upload_checkpoints=False,
                                                    save_checkpoints=False,
                                                    ),
                                MyCallback(),
                        ],
                        )
    


    

    path = Path.home() /"ray_results"
    print(path)

    if Tuner.can_restore(path):
        pass
        #Continuing training

        #tuner = Tuner.restore(path, trainable=my_ppo)
        #results = tuner.fit() 

    else:
        tuner = Tuner("PPO", run_config=run_config, param_space=ppo_config)
        results = tuner.fit()

    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")



    ray.shutdown()


#Todo:
# Eval crashes - why?, eval metrics returns nan
# std log and std error need to go to wandb, they are in the main folder of the run