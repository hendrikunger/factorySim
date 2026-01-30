from typing import Optional,  Sequence
from functools import partial
from datetime import datetime, UTC
import json
import warnings
from pprint import pprint
import os
import wandb
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from torch import Tensor, dist
from helpers.pipeline import env_creator
from env.factorySim.utils import check_internet_conn
import gspread
from env.factorySim.utils import map_factorySpace_to_unit

CREATOR = "Hendrik Unger"



class EvalCallback(RLlibCallback):
    def __init__(self, env_runner_indices: Optional[Sequence[int]] = None):
        super().__init__()
        self.ratings = ['TotalRating', 'EvaluationResult', 'ratingMF', 'ratingTrueMF', 'MFIntersection', 'ratingCollision', 'routeContinuity', 'routeWidthVariance', 'Deadends', 'routeAccess', 'pathEfficiency', 'areaUtilisation', 'Scalability']
        self.id_counter = {}
        

        warnings.filterwarnings(
            "ignore",
            message="Mean of empty slice",
            category=RuntimeWarning,
        )
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
            episode_id = infos[0].get('evalEnvID', '0')
            if episode_id not in self.id_counter:
                self.id_counter[episode_id] = 0
            else:
                self.id_counter[episode_id] += 1
            print(f"Episode {episode_id} Iteration {self.id_counter.get(episode_id, 0)} started")

  

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
            episode_id = infos[0].get('evalEnvID', '0')
            #Save as a dict with key "myData" and the evalEnvID as subkey, so different episodes can be parsed later
            for info in infos:
                metrics_logger.log_dict(info, key=("myData", episode_id, str(self.id_counter[episode_id])), reduce="item_series")
                #Full Logging of all metrics
                for key, value in info.items():
                    if key in self.ratings:
                        metrics_logger.log_value(("means", episode_id, key), value, reduce="mean", window=100)
            

        
    def on_evaluate_start(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        print(f"--------------------------------------------EVAL START--------------------------------------------")
        if algorithm.eval_env_runner_group is None:
            return

        def _reset(env_runner):
            for cb in getattr(env_runner, "_callbacks", []):
                if isinstance(cb, EvalCallback):
                    cb.id_counter.clear()

        algorithm.eval_env_runner_group.foreach_env_runner(
            func=_reset,
            local_env_runner=True,
        )


    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:


        print(f"--------------------------------------------EVAL END--------------------------------------------")
        self.id_counter = {}
        data = {}
        
        #myData = metrics_logger.peek(('evaluation','env_runners'), compile=False)
        
        # data structure in episode:
        # episode_id -> id_counter -> metric_name -> values of all steps
        # these values lists do not appear when print, but can only be accessed via metrics_logger.peek

        myData = metrics_logger.peek(('evaluation','env_runners', 'myData'), compile=False)
        episodes = list(myData.keys())
        first_episode = next(iter(myData.values()))
        iterations = list(first_episode.keys())
        first_iteration = next(iter(first_episode.values()))
        column_names = list(first_iteration.keys())
        for episode in episodes:
            data[episode] = {}
            for iteration in iterations:
                data[episode][iteration] = {}
                for key in column_names:
                    if key == "config":
                        #special handling for config dict
                        config_dict = {}
                        config_data = metrics_logger.peek(('evaluation','env_runners', 'myData', episode, iteration, key), compile=False)
                        for machine_id, machine_data in config_data.items():
                            config_dict[machine_id] = {}
                            for m_key in machine_data.keys():
                                config_dict[machine_id][m_key] = metrics_logger.peek(('evaluation','env_runners', 'myData', episode, iteration, key, machine_id, m_key), compile=False)
                        data[episode][iteration][key] = config_dict
                    else:
                        data[episode][iteration][key] = metrics_logger.peek(('evaluation','env_runners', 'myData', episode, iteration, key), compile=False)



        if data:
            self.upload_google_sheets(data, algo=algorithm.__class__.__name__, current_iteration=algorithm.iteration)
            tbl = wandb.Table(columns=["id"] + column_names)
            #iterate over all eval episodes
            for episode_id, episode in data.items():
                for iteration, infos in episode.items():
                    #infos is a dict of all metrics each value is a list of the values of all steps
                    for step in range(len(infos['Step'])):
                        row = []                     
                        row_id = f"{episode_id}___{iteration:02}_{infos['Step'][step]}"
                        row.append(row_id)
                        for key, values in infos.items():
                            
                            if key == "config":
                                value = {}
                                for machine_id, machine_data in values.items():
                                    value[machine_id] = {}
                                    for m_key in machine_data.keys():
                                        value[machine_id][m_key] = machine_data[m_key][step]
                                fulljson = self.createUploadConfig(values, step, infos.get("Reward", -1.0)[step], episode_id, CREATOR)
                                value = json.dumps(fulljson)
                            else:
                                value = values[step]
                                if key == "Image":
                                    value = wandb.Image(value, caption=row_id, grouping=int(episode_id))

                            row.append(value)
                        tbl.add_data(*row)
            evaluation_metrics["table"] = tbl

        del(myData)
        del(data)
 

    def createUploadConfig(self, data: dict, currentStep: int, reward: float, problem_id: str, creator: str)-> dict:
        #target structure:
        #{"reward": 0.5029859136876851, "individual": [0.13721125779061238, 0.6535398256936433, 0.4487295072012506, 0.39291817669707396, 0.8750841990540823, 0.9755956458283996, 0.8728589587741787, 0.19230067903513304, 0.22079792120893404, 0.6560738790950633, 0.2890835816381917, 0.7347251307805799, 0.5664206508096447, 0.5509087684904029, 0.8285469248387858, 0.7105327717426778, 0.026577764997291697, 0.049459137146015686], "problem_id": "02", "creator": "Hendrik Unger", "config": {"0": {"position": [0.13721125779061238, 0.6535398256936433], "rotation": 0.4487295072012506}, "1": {"position": [0.39291817669707396, 0.8750841990540823], "rotation": 0.9755956458283996}, "2": {"position": [0.8728589587741787, 0.19230067903513304], "rotation": 0.22079792120893404}, "3": {"position": [0.6560738790950633, 0.2890835816381917], "rotation": 0.7347251307805799}, "4": {"position": [0.5664206508096447, 0.5509087684904029], "rotation": 0.8285469248387858}, "5": {"position": [0.7105327717426778, 0.026577764997291697], "rotation": 0.049459137146015686}}}
        config_dict = {}
        config_dict["reward"] = reward
        individual = []
        output_config = {}
        sorted_dict = dict(sorted(data.items(), key=lambda item: item[0]))
        for mid, machine_data in sorted_dict.items():
            posX = machine_data["posX"][currentStep]
            posY = machine_data["posY"][currentStep]
            rot = machine_data["rotation"][currentStep]
            individual+= [posX, posY, rot]
            output_config[mid] = {"posX":posX, "posY":posY, "rotation": rot}
        config_dict["individual"] = individual
        config_dict["problem_id"] = problem_id
        config_dict["creator"] = creator
        config_dict["config"] = output_config
        return config_dict

    def upload_google_sheets(self, data: dict, algo:str, current_iteration: int):
        if check_internet_conn():
            path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(path, "..", "factorysimleaderboard-credentials.json")
            gc = gspread.service_account(filename=path)
            sh = gc.open("FactorySimLeaderboard")
            worksheet = sh.worksheet("Scores")
        
            rows = []

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for episode_id, episode in data.items():
                for _, infos in episode.items():
                    for step in range(len(infos['Step'])):
                        reward = infos.get("Reward", -1.0)[step]
                        if reward < 0.7:
                            continue
                        else:
                            config_dict = self.createUploadConfig(infos["config"], step, reward, episode_id, CREATOR)
                            rows.append([current_time, episode_id, "V1.0", reward, CREATOR, f"factorySim-{algo}-{current_iteration}", json.dumps(config_dict)])

            if len(rows) == 0:
                print("No results to upload", flush=True)
            else:
                worksheet.append_rows(rows, value_input_option="USER_ENTERED")
                print(f"Uploaded {len(rows)} results to leaderboard", flush=True)
        else:
            print("No connection to internet", flush=True)

class AlgorithFix(RLlibCallback):
    def __init__(self, **kwargs):
        super().__init__()

    def on_checkpoint_loaded(self, *, algorithm: Algorithm, **kwargs, ) -> None:
        def betas_tensor_to_float(learner):
            param_grp = next(iter(learner._optimizer_parameters.keys())).param_groups[0]
            if not param_grp['capturable'] and isinstance(param_grp["betas"][0], Tensor):
                param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
        algorithm.learner_group.foreach_learner(betas_tensor_to_float)


# Remote function to change the environment on the worker for curriculum learning
def _remote_fn(env_runner, new_maxMF_Elements: int):
    # We recreate the entire env object by changing the env_config on the worker,
    # then calling its `make_env()` method.
    env_runner.config.environment(env_config={"maxMF_Elements": new_maxMF_Elements})
    env_runner.make_env()


class CurriculumCallback(RLlibCallback):
    """Custom callback implementing curriculum learning based on episode return."""

    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        # Set the initial task to 3 elements, the practical minimum.
        algorithm._counters["current_maxMF_Elements"] = 3

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        current_task = algorithm._counters["current_maxMF_Elements"]
        new_task = None

        # If episode return is consistently `args.upgrade_task_threshold`, we switch
        # to a more difficult task (if possible). If we already mastered the most
        # difficult task, we publish our victory in the result dict.
        result["task_solved"] = 0.0
        current_return = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        if current_return > 0.3:
            if current_task < 12:
                new_task = current_task + 1
                print(
                    f"Switching maxMF_Elements on all EnvRunners to #{new_task} (3=easiest, "
                    f"12=hardest), b/c R={current_return} on current task."
                )

            # Hardest task was solved (1.0) -> report this in the results dict.
            elif current_return > 0.8:
                result["task_solved"] = 1.0
        # Emergency brake: If return is very small AND we are already at a harder task (1 or
        # 2), we go back to task=0.
        elif current_return < -1.0 and current_task > 3:
            print(
                "Emergency brake: Our policy seemed to have collapsed -> Setting maxMF_Elements "
                "back to 3."
            )
            new_task = 3
            
        if new_task:
            algorithm.env_runner_group.foreach_env_runner(
                    func=partial(_remote_fn, new_maxMF_Elements=new_task)
                )
            algorithm._counters["current_maxMF_Elements"] = new_task