from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union, Optional, Sequence
import numpy as np
import gymnasium as gym

from ray.rllib.connectors.env_to_module.observation_preprocessor import SingleAgentObservationPreprocessor
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from env.factorySim.factorySimEnv import FactorySimEnv#, MultiFactorySimEnv

#Creator----------------------------------------------------------------------------------------------------------------------------------------------------------
def env_creator(env_config):
    if env_config['algo'] == "Dreamer":
        env = ZeroOneActionWrapper(FactorySimEnv(env_config=env_config))
    else:
        env = FactorySimEnv(env_config=env_config)

    return  env # return an env instance


#Env----------------------------------------------------------------------------------------------------------------------------------------------------------
# This wrapper is just for Dreamer V3, which expects actions in [-1,1] and observations in [0,1] and does not have  env_to_module connectors yet
class ZeroOneActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Let RLlib think the env also lives in [-1,1]
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(h, w, c), dtype=np.float32)

    def action(self, act):
        # scale [-1,1] -> [0,1]
        return (act + 1.0) / 2.0
    
    def observation(self, obs):
        return (obs.astype(np.float32) / 255.0)
    
#Preprocessor----------------------------------------------------------------------------------------------------------------------------------------------------------

class NormalizeObservations(SingleAgentObservationPreprocessor):
    def preprocess(self, observation: Dict[AgentID, Dict[str, np.ndarray]], episode: SingleAgentEpisode) -> Dict[AgentID, Dict[str, np.ndarray]]:
        output= observation / 255.0
        return output.astype(np.float32)

