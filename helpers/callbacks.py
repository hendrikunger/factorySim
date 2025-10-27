from typing import Optional,  Sequence
from functools import partial
import wandb
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from torch import Tensor
from helpers.pipeline import env_creator






class EvalCallback(RLlibCallback):
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

        # If episode return is consistently `args.upgrade_task_threshold`, we switch
        # to a more difficult task (if possible). If we already mastered the most
        # difficult task, we publish our victory in the result dict.
        result["task_solved"] = 0.0
        current_return = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        if current_return > 0.0:
            if current_task < 11:
                new_task = current_task + 1
                print(
                    f"Switching maxMF_Elements on all EnvRunners to #{new_task} (0=easiest, "
                    f"11=hardest), b/c R={current_return} on current task."
                )
                algorithm.env_runner_group.foreach_env_runner(
                    func=partial(_remote_fn, new_task=new_task)
                )
                algorithm._counters["current_maxMF_Elements"] = new_task

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
            algorithm.env_runner_group.foreach_env_runner(
                func=partial(_remote_fn, new_task=3)
            )
            algorithm._counters["current_maxMF_Elements"] = 0