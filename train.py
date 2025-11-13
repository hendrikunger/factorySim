import sys
from datetime import datetime
from pathlib import Path
import os
import platform
from dataclasses import asdict

import numpy as np
import yaml


import ray
from ray.tune import Tuner
from ray.tune import RunConfig, CheckpointConfig, TuneConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from experimental.intrinsic_curiosity_learners import PPOTorchLearnerWithCuriosity, ICM_MODULE_ID
from experimental.intrinsic_curiosity_model_rlm import IntrinsicCuriosityModel
from experimental.safe_sac_module import SafeSACTorchRLModule
from experimental.vision_sac_catalog import VisionSACCatalog
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.connectors.env_to_module.observation_preprocessor import SingleAgentObservationPreprocessor
from ray.tune.registry import register_env
from helpers.cli import get_args
from helpers.pipeline import NormalizeObservations, env_creator
from helpers.callbacks import EvalCallback, AlgorithFix, CurriculumCallback



#filename = "Overlapp"
filename = "Basic"
#filename = "EP_v23_S1_clean"
#filename = "Simple"
#filename = "SimpleNoCollisions"
#filename = "LShape"

#ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1", filename + ".ifc")
basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)))

#Import Custom Models
#from ray.rllib.models import ModelCatalog
#ModelCatalog.register_custom_model("my_model", MyXceptionModel)



NO_TUNE = False  

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"




register_env("FactorySimEnv", env_creator)



#Preprocessor----------------------------------------------------------------------------------------------------------------------------------------------------------       
def _env_to_module(env=None, spaces=None, device=None) -> SingleAgentObservationPreprocessor:
# Create the env-to-module connector pipeline.
    return NormalizeObservations()




#Main Function----------------------------------------------------------------------------------------------------------------------------------------------------------

def run():

    args = get_args()
    if args.config:
        config_path = args.config
    elif args.configID != 0:
        exp_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Experiments"))
        expFiles = [p for p in exp_dir.rglob("*") if p.suffix in {".yaml", ".yml"}]
        expFiles.sort()
        print(f"Found {len(expFiles)} experiment config files")
        print(expFiles)
        possible_configs = [f for f in expFiles if f.name.startswith(f"{args.configID}_")]
        if len(possible_configs) == 1:
            config_path = possible_configs[0]
        else:
            raise ValueError(f"Could not find unique config file for ID {args.configID}, found: {possible_configs}")
    else:
        config_path = "config.yaml"
    
    with open(config_path, 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)


    ifcpath= os.path.join(basepath, *f_config.get('subdir', ["Input", "1"]))
    f_config['env_config']['inputfile'] = ifcpath

    print("--------------------------------------------------------------------------------------------------------")
    print(f"Using config file: {config_path}")
    print(f"Using input file path: {ifcpath}")
    print("--------------------------------------------------------------------------------------------------------")



    runtime_env = {
    "env_vars": {"PYTHONWARNINGS": "ignore::UserWarning",
                 "NCCL_P2P_DISABLE":"1"      # Disable NCCL P2P communication on slurm cluster, because it is flaky
                 },
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
    NUMGPUS = int(os.getenv("SLURM_GPUS",
                0 if sys.platform == "darwin" else 1))
    INDEX = (os.getenv("SLURM_STEP_GPUS"))
    
    print(f"Using {NUMGPUS} GPUs, with index {INDEX}, Cuda visible {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")


    if "SLURM_JOB_UID" in os.environ or sys.platform == "darwin" or platform.node() == "pop-os":
        ray.init(num_gpus=NUMGPUS, runtime_env=runtime_env, include_dashboard=True, dashboard_host="127.0.0.1") 
    else:
        #we are running on ray cluster
        runtime_env["py_modules"] = ["env/factorySim"]
        runtime_env["env_vars"] = {
            "PYTHONWARNINGS": "ignore::UserWarning",
        }
        NUMGPUS = f_config.get("num_gpus", 1)
        ray.init(runtime_env=runtime_env, include_dashboard=True, dashboard_host="0.0.0.0")


    training_iteration = f_config.get("training_iteration", 2)
    if args.test:
        training_iteration = 2
        f_config['evaluation_config']['evaluation_interval'] = 1
    stop = {
    "training_iteration": training_iteration, #Number of training iterations
    #"num_env_steps_sampled_lifetime": 15000000,
    #"episode_reward_mean": 5,
    }

    checkpoint_config = CheckpointConfig(checkpoint_at_end=True, 
                                         checkpoint_frequency=100, 
                                         checkpoint_score_order="max",
                                         #checkpoint_score_attribute="env_runners/episode_return_mean", 
                                         num_to_keep=5 
    )

    match f_config['env_config']['algo']:
        case "PPO":
    
            algo_config = PPOConfig()
            algo_config.training(
                            vf_loss_coeff= f_config['vf_loss_coeff'],
                            train_batch_size=f_config['train_batch_size_per_learner'],
                            minibatch_size=f_config['mini_batch_size_per_learner'],
                            learner_class=PPOTorchLearnerWithCuriosity,
                            learner_config_dict={
                                # Intrinsic reward coefficient.
                                "intrinsic_reward_coeff": 0.05,
                                # Forward loss weight (vs inverse dynamics loss). Total ICM loss is:
                                # L(total ICM) = (
                                #     `forward_loss_weight` * L(forward)
                                #     + (1.0 - `forward_loss_weight`) * L(inverse_dyn)
                                # )
                                "forward_loss_weight": 0.2,
            }


            ) 


            algo_config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # The "main" RLModule (policy) to be trained by our algo.
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        module_class=DefaultPPOTorchRLModule,
                        model_config=DefaultModelConfig(
                            conv_filters= [# [ num_filters, kernel, stride]
                                    [32, 5, 2],   # 128x128 → 62x62
                                    [48, 4, 2],   # 62x62   → 30x30
                                    [64, 4, 2],   # 30x30   → 14x14
                                    [64, 3, 2],   # 14x14   → 6x6
                                    [96, 3, 2],   # 6x6     → 2x2
                                    [128, 2, 2],   # 2x2     → 1x1

                                        # [32, 8, 4],  # Reduces spatial size from 84x84 -> 20x20
                                        # [64, 4, 2],  # Reduces spatial size from 20x20 -> 9x9
                                        # [128, 3, 1],  # Reduces spatial size from 9x9 -> 7x7
                                        # [256, 7, 1],  # Reduces spatial size from 7x7 -> 1x1
                                    ],
                            conv_activation="relu",
                            head_fcnet_hiddens=[256],
                            vf_share_layers=True,   #Need to tune  vf_loss_coeff on Training config if you set this to True
                        )
                    ),
                    # The intrinsic curiosity model.
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=IntrinsicCuriosityModel,
                        # Only create the ICM on the Learner workers, NOT on the
                        # EnvRunners.
                        learner_only=True,
                        # Configure the architecture of the ICM here.
                        model_config={
                            "feature_dim": 512,
                            "feature_net_hiddens": (256, 256),
                            "feature_net_activation": "relu",
                            "inverse_net_hiddens": (256, 256),
                            "inverse_net_activation": "relu",
                            "forward_net_hiddens": (256, 256),
                            "forward_net_activation": "relu",
                        },
                    ),
                }
            ),
            # Use a different learning rate for training the ICM.
            algorithm_config_overrides_per_module={
                ICM_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
            },
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
        #APPO END ------------------------------------------------------------------------------------------------------
        case "Dreamer":
            algo_config = DreamerV3Config()

            #Dreamer needs 64x64x3 input
            f_config['env_config']['width'] = 64
            f_config['env_config']['height'] = 64
            f_config['evaluation_config']['env_config']['width'] = 64 
            f_config['evaluation_config']['env_config']['height'] = 64  
            #Dreamer needs 3 channels
            f_config['env_config']['coordinateChannels'] = False
            f_config['evaluation_config']['env_config']['coordinateChannels'] = False  
            w = algo_config.world_model_lr
            c = algo_config.critic_lr
            algo_config.training(
                model_size="L",
                training_ratio=64, #512, #Should be lower for larger models e.g. 64 for XL  
                batch_size_B= 16 * (f_config["num_gpus"] or 1),
                # Use a well established 4-GPU lr scheduling recipe:
                # ~ 1000 training updates with 0.4x[default rates], then over a few hundred
                # steps, increase to 4x[default rates].
                world_model_lr=[[0, 0.4 * w], [8000, 0.4 * w], [10000, 3 * w]],
                critic_lr=[[0, 0.4 * c], [8000, 0.4 * c], [10000, 3 * c]],
                actor_lr=[[0, 0.4 * c], [8000, 0.4 * c], [10000, 3 * c]],
            )
            # algo_config.env_runners(
            #     rollout_fragment_length = 128,  # 64 steps per env runner
            # )
            algo_config.reporting(min_sample_timesteps_per_iteration=512)

        #Dreamer END ------------------------------------------------------------------------------------------------------
        case "SAC":
            algo_config = SACConfig()


            algo_config.training(
                    initial_alpha=1.001,
                    # lr=0.0006 is very high, w/ 4 GPUs -> 0.0012
                    # Might want to lower it for better stability, but it does learn well.
                    actor_lr=2e-4 * (f_config["num_gpus"] or 1) ** 0.5,
                    critic_lr=8e-4 * (f_config["num_gpus"] or 1) ** 0.5,
                    alpha_lr=9e-4 * (f_config["num_gpus"] or 1) ** 0.5,
                    lr=None,
                    target_entropy="auto",
                    n_step=(1,5),  # 1?
                    tau=0.005,
                    train_batch_size_per_learner=f_config['train_batch_size_per_learner'],
                    target_network_update_freq=1,
                    replay_buffer_config={
                        "type": "PrioritizedEpisodeReplayBuffer",
                        "capacity": 65536,
                        "alpha": 0.6,
                        "beta": 0.4,
                        "storage_unit": "episodes",
                    },
                    num_steps_sampled_before_learning_starts=10000,
            )
            
            model_config = DefaultModelConfig(
                                            conv_filters= [# [ num_filters, kernel, stride]
                                                        [32, 5, 2],   # 128x128 → 62x62
                                                        [64, 4, 2],   # 62x62 → 30x30
                                                        [128, 4, 2],  # 30x30 → 14x14
                                                        [128, 3, 2],  # 14x14 → 6x6
                                                        [256, 3, 2],  # 6x6   → 2x2
                                                        [256, 2, 2],  # 2x2   → 1x1   

                                                    ],
                                            conv_activation="relu",
                                            head_fcnet_hiddens=[256],
                                            fcnet_hiddens=[256, 256],
                                            fcnet_activation="relu",
                                            )
            
            model_config = asdict(model_config)         
            model_config["twin_q"] = True,           

            algo_config .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=SafeSACTorchRLModule,   
                    catalog_class=VisionSACCatalog,      
                    model_config=model_config,
                )
            )
                                    

        #SAC END ------------------------------------------------------------------------------------------------------


    algo_config.environment("FactorySimEnv", env_config=f_config['env_config'], render_env=False, disable_env_checking=True)

    if f_config['env_config'].get('curriculum_learning', False):
        algo_config.callbacks(callbacks_class=[EvalCallback, AlgorithFix, CurriculumCallback])
    else:
        algo_config.callbacks(callbacks_class=[EvalCallback, AlgorithFix])
    algo_config.env_runners(num_env_runners= int(os.getenv("SLURM_CPUS_PER_TASK", f_config['num_workers']))-1,
                        num_envs_per_env_runner=f_config.get('num_envs_per_env_runner', 1),
                        num_cpus_per_env_runner=1,
                        env_to_module_connector=_env_to_module,
                        num_gpus_per_env_runner=0,
                        gym_env_vectorize_mode="SYNC",
                        create_local_env_runner=True,           
                        create_env_on_local_worker=True,       
                        )
    algo_config.resources(num_cpus_for_main_process=1)
    algo_config.learners(num_learners=f_config['num_learners'],
                         num_gpus_per_learner=0 if sys.platform == "darwin" else 1,
                         #num_aggregator_actors_per_learner=0,  #Does not work with curiosity
                         )
    algo_config.framework("torch",
                        #   torch_compile_learner=False,
                        #   torch_ddp_kwargs={"static_graph": True,
                        #   "find_unused_parameters": False},
                          )
    

    eval_config = f_config['evaluation_config']["env_config"]


    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation") 

    eval_duration = len([x for x in os.listdir(path) if ".ifc" in x])
    algo_config.evaluation(evaluation_duration=eval_duration,
                          evaluation_duration_unit="episodes", 
                          evaluation_interval=f_config["evaluation_interval"], 
                          evaluation_config={"env_config": eval_config},
                          evaluation_parallel_to_training=f_config["evaluation_parallel_to_training"],
                          evaluation_num_env_runners=f_config.get("evaluation_num_env_runners", 0), #Number of env runners for evaluation
                        )   
    

    name = os.getenv("SLURM_ARRAY_JOB_ID", None)
    if name is None:
        name = os.getenv("SLURM_JOB_ID", f"{f_config['group']}_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}_{args.configID}")
    else:
        name = f"{f_config['group']}_{name}_{os.getenv('SLURM_ARRAY_TASK_ID', '')}"
    


    run_config=RunConfig(name="bert",
                            stop=stop,
                            checkpoint_config=checkpoint_config,
                            #log_to_file="./wandb/latest-run/files/stdoutanderr.log",
                            callbacks=[
                                WandbLoggerCallback(project=f_config.get("project", "factorySim_TRAIN")  ,
                                                    log_config=True,
                                                    upload_checkpoints=False,
                                                    name=name,
                                                    group=f_config.get("group", "default"),
                                                    notes=f_config.get("notes", ""),
                                                    tags=f_config.get("tags", []),
                                                    ),

                        ],
                        )



    tune_config = TuneConfig(
        num_samples=1,
        #metric="evaluation/episode_return_mean",
        #mode="max",
    )

    if args.resume:
        path = Path.joinpath(Path.home(), "ray_results/bert/")
    else:
        path= None
    

    if path and Tuner.can_restore(path):
        print("--------------------------------------------------------------------------------------------------------")
        print(f"Restoring from {path.as_posix()}")
        print("--------------------------------------------------------------------------------------------------------")
        #Continuing training

        tuner = Tuner.restore(path.as_posix(), f_config['env_config']['algo'], param_space=algo_config, resume_errored=True)
        results = tuner.fit() 

    else:
        print("--------------------------------------------------------------------------------------------------------")
        print(f"New RUN")
        print("--------------------------------------------------------------------------------------------------------")
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
                    eval_return = results["evaluation"].get(
                        "episode_return_mean", np.nan
                    )
                    print(f" EVAL R={eval_return}", end="")
                print()
        else:
            tuner = Tuner(algo_config.algo_class, run_config=run_config, param_space=algo_config, tune_config=tune_config)
            results = tuner.fit()

        print("Training finished")


    #Loading for Evaluation

    #agent = ppo.PPO(config=config, env=MultiFactorySimEnv)
    #agent.restore("/root/ray_results/PPO/PPO_MultiEnv_2fa55_00000_0_2022-11-19_10-08-59/checkpoint_000667/")



if __name__ == "__main__":

    run()



# Todo: CoordConv: "An intriguing failing of CNNs and the CoordConv solution" (Liu et al., 2018).  Add coordinates to the input image, so that the model can learn to use them.c