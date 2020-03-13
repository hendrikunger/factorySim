import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_factorySim import FactorySimEnv
import pandas as pd



# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: FactorySimEnv("/workspace/factorySim/Input/Simple.ifc", Loglevel=2)])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=200)
obs = env.reset()

for i in range(200):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()