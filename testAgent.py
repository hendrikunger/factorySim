import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import gym_factorySim
import pandas as pd



# The algorithms require a vectorized environment to run
env = gym.make('factorySimEnv-v0',inputfile = "/workspace/factorySim/Input/Simple.ifc", Loglevel=2) 
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=200)
obs = env.reset()
model.save("ppo2_testagent")

#del model 
#model = PPO2.load("ppo2_testagent")

for i in range(200):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render(mode = 'imageseries')