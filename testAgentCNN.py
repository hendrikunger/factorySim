import gym
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
import gym_factorySim
import numpy as np
import os
import imageio



def make_env(env_id, rank, ifcpath, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make('factorySimEnv-v0',inputfile = ifcpath, uid=rank, width=500, heigth=500, Loglevel=0)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init



if __name__ == "__main__":
  #filename = "Overlapp"
  filename = "Basic"
  #filename = "EP_v23_S1_clean"
  #filename = "Simple"
  #filename = "SimpleNoCollisions"

  ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
    "Input",  
    filename + ".ifc")


  #check_env(gym.make('factorySimEnv-v0',inputfile = ifcpath, width=500, heigth=500, Loglevel=0))

  # The algorithms require a vectorized environment to run
  num_cpu = 8  # Number of processes to use
  env_id = 'factorySimEnv-v0'

  # Create the vectorized environment
  #env = SubprocVecEnv([lambda : gym.make('factorySimEnv-v0',inputfile = ifcpath, uid=i, width=500, heigth=500, Loglevel=0) for i in range(num_cpu)])
  env = SubprocVecEnv([make_env(env_id, i, ifcpath) for i in range(num_cpu)])



  model = PPO2(CnnLstmPolicy, env, tensorboard_log="./ppo2_factorySim_tensorboard/", verbose=1)
  model.learn(total_timesteps=200, tb_log_name="first_run")
  obs = env.reset()
  #model.save("ppo2_CnnLstmPolicy_testagent200k")

  #del model 
  #model = PPO2.load("ppo2_testagent100k", env=env, custom_objects={"tensorboard_log":"./ppo2_factorySim_tensorboard/"})

  #model.learn(total_timesteps=75000, tb_log_name="second_run")
  #obs = env.reset()
  #model.save("ppo2_testagent100k")


  done = [False for _ in range(env.num_envs)]
  # Passing state=None to the predict function means
  # it is the initial state
  state = None

  prefix = 0
  images = []
  obs = model.env.reset()
  img = model.env.render(mode='rgb_array')

  for i in range(200):
    images.append(img)
    action, state = model.predict(obs, state=state, mask=done)
    obs, rewards, done, info = env.step(action)
    img = env.render(mode = 'rgb_array', prefix = None)

  imageio.mimsave('test.gif', [np.array(img) for i, img in enumerate(images)], fps=4)