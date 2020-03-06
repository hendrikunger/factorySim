from gym.envs.registration import register
 
register(id='factorySimEnv-v0', 
    entry_point='gym_factorySim.envs:FactorySimEnv',
)