import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import gym_factorySim
import pandas as pd
import os


#filename = "Overlapp"
#filename = "EP_v23_S1_clean"
filename = "Simple"
#filename = "SimpleNoCollisions"

ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
  "Input",  
  filename + ".ifc")




# The algorithms require a vectorized environment to run
env = gym.make('factorySimEnv-v0',inputfile = ifcpath, Loglevel=2) 
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000)
obs = env.reset()
model.save("ppo2_testagent")

#del model 
#model = PPO2.load("ppo2_testagent")

for i in range(200):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render(mode = 'imageseries')