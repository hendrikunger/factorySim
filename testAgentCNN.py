import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
import gym_factorySim
import pandas as pd
import os

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
  num_cpu = 4  # Number of processes to use
  # Create the vectorized environment
  env = SubprocVecEnv([lambda : gym.make('factorySimEnv-v0',inputfile = ifcpath, width=500, heigth=500, Loglevel=0) for _ in range(num_cpu)])




  model = PPO2(CnnLstmPolicy, env, tensorboard_log="./ppo2_factorySim_tensorboard/", verbose=1)
  model.learn(total_timesteps=2000, tb_log_name="first_run")
  obs = env.reset()
  model.save("ppo2_CnnLstmPolicy_testagent200k")

  #del model 
  #model = PPO2.load("ppo2_testagent100k", env=env, custom_objects={"tensorboard_log":"./ppo2_factorySim_tensorboard/"})

  #model.learn(total_timesteps=75000, tb_log_name="second_run")
  #obs = env.reset()
  #model.save("ppo2_testagent100k")

  prefix = 0
  for i in range(200):
    print(i)
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode = 'imageseries', prefix = prefix)
    if done:
      prefix += 1