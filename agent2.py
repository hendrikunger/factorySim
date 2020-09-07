import gym
from stable_baselines.common.policies import CnnLnLstmPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import PPO2
import gym_factorySim
import numpy as np
import os
import imageio

import math
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

do_train = True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        info = self.training_env.get_attr("info")
        for envinfo in info:
          summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/TotalRating', simple_value=envinfo.get('TotalRating')),
            tf.Summary.Value(tag='episode_reward/Collision_Rating', simple_value=envinfo.get('ratingCollision')),
            tf.Summary.Value(tag='episode_reward/MF_Rating', simple_value=envinfo.get('ratingMF'))
            ])
          self.locals['writer'].add_summary(summary, self.num_timesteps)

          if 'done' in envinfo:
            summary = tf.Summary(value=[tf.Summary.Value(tag='Clean/FinalMF', simple_value=envinfo.get('ratingMF')),
            tf.Summary.Value(tag='Clean/FinalCollision', simple_value=envinfo.get('ratingCollision'))
            ])
            self.locals['writer'].add_summary(summary, self.num_timesteps)

          if(envinfo.get('ratingCollision') == 1):
            summary = tf.Summary(value=[tf.Summary.Value(tag='Clean/_MF_Rating_without_Collision', simple_value=envinfo.get('ratingMF'))
            ])
            self.locals['writer'].add_summary(summary, self.num_timesteps)

            if 'done' in envinfo:
              summary = tf.Summary(value=[tf.Summary.Value(tag='Clean/FinalMF_Rating_without_Collision', simple_value=envinfo.get('ratingMF'))])
              self.locals['writer'].add_summary(summary, self.num_timesteps)

        return True


def make_env(env_id, rank, ifcpath, scaling=1.0, seed=0, maxElements=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make('factorySimEnv-v0',inputfile = ifcpath, uid=rank, width=128, heigth=128, maxMF_Elements=maxElements, outputScale=4, objectScaling=scaling, Loglevel=0)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def prepareEnv(ifc_filename = "", objectScaling = 1.0, maxElements = None):

  num_cpu = 32  #52  # Number of processes to use
  env_id = 'factorySimEnv-v0'
  if(True):
    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", ifc_filename)
  else:
    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
      "Input",  
      ifc_filename + ".ifc")

  return SubprocVecEnv([make_env(env_id, i, ifcpath, scaling=objectScaling, maxElements=maxElements) for i in range(num_cpu)])


def takePictures(model, prefix):
  #env = model.get_env()
  done = [False for _ in range(model.env.num_envs)]
  # Passing state=None to the predict function means
  # it is the initial state
  state = None

  #images = []
  single_images = []
  obs = model.env.reset()
  img = model.env.render(mode='rgb_array')
  single_img = env.get_images()

  for i in range(20):
    #images.append(img)
    single_images.append(single_img)
    action, state = model.predict(obs, state=state, mask=done)
    obs, _rewards, done, _info = model.env.step(action)
    #ALl immages in one
    #img = model.env.render(mode = 'rgb_array', prefix = "")
    #Single Images per Environment
    single_img = env.get_images()


  #imageio.mimsave('./Output/2.gif', [np.array(img) for i, img in enumerate(images)], fps=4)

  os.makedirs(f"./Output/{prefix}/", exist_ok=True)
  for i, fullimage in enumerate(single_images):
    for envid, sImage in enumerate(fullimage):
      imageio.imsave(f"./Output/{prefix}/{envid }.{i}.png", np.array(sImage))

#---------------------------------------------------------------------------------------------------------------------
#Training
#---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  if do_train:

    #ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input", "1")
    #check_env(gym.make('factorySimEnv-v0', inputfile = ifcpath, width=128, heigth=128, objectScaling=0.5, Loglevel=0))

    env = prepareEnv(ifc_filename = "1", objectScaling=0.5, maxElements=4)
    model = PPO2(CnnLnLstmPolicy,
      env,
      tensorboard_log="./log/",
      gamma=0.95, # Tradeoff between short term (=0) and longer term (=1) rewards. If to big, we are factoring in to much unnecessary info |0.99
      n_steps=512, # | 128 
      ent_coef=0.01,  #Speed of Entropy drop if it drops to fast, increase | 0.01 *
      learning_rate=0.00025, # | 0.00025 *
      vf_coef=0.5, # | 0.5
      max_grad_norm=0.5, # | 0.5
      lam=0.95,   #Tradeoff between current value estimate (maybe high bias) and acually received reward (maybe high variance) | 0.95
      nminibatches=4, # | 4
      noptepochs=4, # | 4
      verbose=1)
    

    model.learn(total_timesteps=5000000, tb_log_name="Batch_1",reset_num_timesteps=True, callback=TensorboardCallback())
    takePictures(model,1)
    model.save(f"./models/ppo2_1")
    #close old env and make new one
    env.close()

    for i in range(1,9):
      env = prepareEnv(ifc_filename = str(2 - i%2), objectScaling=min(0.5 + i/10, 1), maxElements=4+ math.ceil(i/3))
      model.set_env(env)
      model.learn(total_timesteps=4000000, tb_log_name=f"Batch_{i+1}",reset_num_timesteps=False, callback=TensorboardCallback())
      takePictures(model,i+1)
      model.save(f"./models/ppo2_{i+1}")
      env.close()
    
  else:
    env = prepareEnv()
    model = PPO2.load("./models/ppo2_", env=env, tensorboard_log="./log/")

    

#---------------------------------------------------------------------------------------------------------------------
#Evaluation
#---------------------------------------------------------------------------------------------------------------------
  env = prepareEnv(ifc_filename = "2", objectScaling=1)
  model.set_env(env)

  takePictures(model,"X")

  env.close()
  