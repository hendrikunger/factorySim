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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


def prepareEnv(ifc_filename):

  num_cpu =10  # Number of processes to use
  env_id = 'factorySimEnv-v0'

  ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
    "Input",  
    ifc_filename + ".ifc")

  return SubprocVecEnv([make_env(env_id, i, ifcpath) for i in range(num_cpu)])


if __name__ == "__main__":
  #filename = "Overlapp"
  filename = "Basic"
  #filename = "EP_v23_S1_clean"
  #filename = "Simple"
  #filename = "SimpleNoCollisions"

  #check_env(gym.make('factorySimEnv-v0',inputfile = ifcpath, width=500, heigth=500, Loglevel=0))

  # Create the vectorized environment
  env = prepareEnv("Basic")
  model = PPO2(CnnLstmPolicy, env, tensorboard_log="./ppo2_factorySim_tensorboard/",  nminibatches=5, verbose=1)
  model.learn(total_timesteps=500000, tb_log_name="Basic_1",reset_num_timesteps=True)

  #close old env and make new one
  env.close()
  env = prepareEnv("Simple")
  model.set_env(env)
  model.learn(total_timesteps=300000, tb_log_name="Simple_1",reset_num_timesteps=True)

  env.close()
  env = prepareEnv("Overlapp")
  model.set_env(env)
  model.learn(total_timesteps=300000, tb_log_name="Simple_1",reset_num_timesteps=True)

  model.save("ppo2")
  #del model 
  #model = PPO2.load("ppo2_testagent100k", env=env, custom_objects={"tensorboard_log":"./ppo2_factorySim_tensorboard/"})


#---------------------------------------------------------------------------------------------------------------------
#Evaluation
#---------------------------------------------------------------------------------------------------------------------

  #env = model.get_env()
  done = [False for _ in range(env.num_envs)]
  # Passing state=None to the predict function means
  # it is the initial state
  state = None

  prefix = 0
  images = []
  single_images = []
  obs = model.env.reset()
  img = model.env.render(mode='rgb_array')
  single_img = env.get_images()

  for i in range(50):
    images.append(img)
    single_images.append(single_img)
    action, state = model.predict(obs, state=state, mask=done)
    obs, rewards, done, info = model.env.step(action)
    #ALl immages in one
    img = model.env.render(mode = 'rgb_array', prefix = "")
    #Single Images per Environment
    single_img = env.get_images()

  imageio.mimsave('Overlapp.gif', [np.array(img) for i, img in enumerate(images)], fps=4)
  
  for i, fullimage in enumerate(single_images):
    for envid, sImage in enumerate(fullimage):
      imageio.imsave(f"./Output/Overlapp/{envid }.{i}.png", np.array(sImage))

  env.close()
  env = prepareEnv("Simple")
  model.set_env(env)

  #env = model.get_env()
  done = [False for _ in range(env.num_envs)]
  # Passing state=None to the predict function means
  # it is the initial state
  state = None

  prefix = 0
  images = []
  single_images = []
  obs = model.env.reset()
  img = model.env.render(mode='rgb_array')
  single_img = env.get_images()

  for i in range(50):
    images.append(img)
    single_images.append(single_img)
    action, state = model.predict(obs, state=state, mask=done)
    obs, rewards, done, info = model.env.step(action)
    #ALl immages in one
    img = model.env.render(mode = 'rgb_array', prefix = "")
    #Single Images per Environment
    single_img = env.get_images()

  imageio.mimsave('simple.gif', [np.array(img) for i, img in enumerate(images)], fps=4)
  
  for i, fullimage in enumerate(single_images):
    for envid, sImage in enumerate(fullimage):
      imageio.imsave(f"./Output/Simple/{envid }.{i}.png", np.array(sImage))