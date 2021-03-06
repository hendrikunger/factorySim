import gym
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import TRPO
import gym_factorySim
import numpy as np
import os
import imageio

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

        # Log scalar value (here a random variable)
        info = self.training_env.get_attr("info")
        for envinfo in info:
          summary = tf.Summary(value=[tf.Summary.Value(node_name='factorySim', tag='TotalRating', simple_value=envinfo['TotalRating'])])
          self.locals['writer'].add_summary(summary, self.num_timesteps)
          summary = tf.Summary(value=[tf.Summary.Value(node_name='factorySim', tag='MF_Rating', simple_value=envinfo['ratingMF'])])
          self.locals['writer'].add_summary(summary, self.num_timesteps)
          summary = tf.Summary(value=[tf.Summary.Value(node_name='factorySim', tag='Collision_Rating', simple_value=envinfo['ratingCollision'])])
          self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True


def make_env(env_id, rank, ifcpath, scaling=1.0, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make('factorySimEnv-v0',inputfile = ifcpath, uid=rank, width=128, heigth=128, outputScale=4, objectScaling=scaling, Loglevel=0)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def prepareEnv(ifc_filename = "", objectScaling = 1.0):

  num_cpu = 16  # Number of processes to use
  env_id = 'factorySimEnv-v0'
  if(ifc_filename == ""):
    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input")
  else:
    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
      "Input",  
      ifc_filename + ".ifc")

  return SubprocVecEnv([make_env(env_id, i, ifcpath, scaling=objectScaling) for i in range(num_cpu)])

#---------------------------------------------------------------------------------------------------------------------
#Training
#---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  if do_train:
    #filename = "Overlapp"
    filename = "Basic"
    #filename = "EP_v23_S1_clean"
    #filename = "Simple"
    #filename = "SimpleNoCollisions"

    #ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Input")
    #check_env(gym.make('factorySimEnv-v0', inputfile = ifcpath, width=128, heigth=128, objectScaling=0.5, Loglevel=0))

    # Create the vectorized environment
    #env = prepareEnv("Basic")
    env = prepareEnv(objectScaling=0.5)
    model = TRPO(CnnLstmPolicy,
        env,
        gamma=0.99, 
        timesteps_per_batch=1024, 
        max_kl=0.01, 
        cg_iters=10, 
        lam=0.98, 
        entcoeff=0.0, 
        cg_damping=0.01, 
        vf_stepsize=0.0003, 
        vf_iters=3, 
        verbose=0, 
        tensorboard_log="./log/",
        _init_setup_model=True, 
        policy_kwargs=None, 
        full_tensorboard_log=False, 
        seed=None, 
        n_cpu_tf_sess=None)
      
    #model = TRPO.load("trpo", env=env, tensorboard_log="./log/")
    model.learn(total_timesteps=2000, tb_log_name="Batch_A",reset_num_timesteps=True, callback=TensorboardCallback())
    #model.learn(total_timesteps=1500, tb_000log_name="Basic1",reset_num_timesteps=True)
    

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

    for i in range(100):
      images.append(img)
      single_images.append(single_img)
      action, state = model.predict(obs, state=state, mask=done)
      obs, rewards, done, info = model.env.step(action)
      #ALl immages in one
      img = model.env.render(mode = 'rgb_array', prefix = "")
      #Single Images per Environment
      single_img = env.get_images()

    imageio.mimsave('./Output/1.gif', [np.array(img) for i, img in enumerate(images)], fps=4)

    os.makedirs("./Output/1/", exist_ok=True)
    for i, fullimage in enumerate(single_images):
      for envid, sImage in enumerate(fullimage):
        imageio.imsave(f"./Output/1/{envid }.{i}.png", np.array(sImage))


    #close old env and make new one
    env.close()
    env = prepareEnv(objectScaling=0.7)
    model.set_env(env)
    model.learn(total_timesteps=2000, tb_log_name="Batch_B",reset_num_timesteps=True, callback=TensorboardCallback())
    #model.learn(total_timesteps=1200000, tb_log_name="Simple1",reset_num_timesteps=True)

    #env.close()
    #env = prepareEnv("Basic")
    #model.set_env(env)
    #model.learn(total_timesteps=1000000, tb_log_name="Overlapp_1",reset_num_timesteps=True)

    model.save("trpo")
    
  else:
    env = prepareEnv()
    model = PPO2.load("trpo", env=env, tensorboard_log="./log/")

    

#---------------------------------------------------------------------------------------------------------------------
#Evaluation
#---------------------------------------------------------------------------------------------------------------------
  env.close()
  env = prepareEnv(objectScaling=0.7)
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

  for i in range(100):
    images.append(img)
    single_images.append(single_img)
    action, state = model.predict(obs, state=state, mask=done)
    obs, rewards, done, info = model.env.step(action)
    #ALl immages in one
    img = model.env.render(mode = 'rgb_array', prefix = "")
    #Single Images per Environment
    single_img = env.get_images()


  imageio.mimsave('./Output/2.gif', [np.array(img) for i, img in enumerate(images)], fps=4)

  os.makedirs("./Output/2/", exist_ok=True)
  for i, fullimage in enumerate(single_images):
    for envid, sImage in enumerate(fullimage):
      imageio.imsave(f"./Output/2/{envid }.{i}.png", np.array(sImage))

  env.close()
  