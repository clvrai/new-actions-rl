import sys
sys.path.insert(0, '.')
import gym
import envs.recogym
import random
from envs.recogym import env_1_args, Configuration
import numpy as np
from numpy.random import choice
# from agents import Agent

# Our total products
num_products = 10
prods = list(range(num_products))
action_set_size = 40

env_1_args['random_seed'] = 42
env_1_args['num_products'] = num_products
#env_1_args['prob_bandit_to_organic'] = 0.0
# When 0 there is a direct correlation between views and a product click. A
# number greater than 0 will distort this correlation.
env_1_args['number_of_flips']=0
# Dimension of the action embedding
env_1_args['K'] = 10
#env_1_args['prob_leave_organic'] = 0.0


num_iters = 1000
num_clicks = 0
num_events = 0

random.seed(42)

reco = random.choices(prods, k=10)

env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)

total_steps = 0
total_reward = 0
max_ep_reward = 0
max_obs_len = 0
for i in range(num_iters):
    reward = None

    #aval_actions = random.sample(prods, action_set_size)
    env.reset()
    obs, _, _, _ = env.step(None)
    done = False

    #print('After')
    ep_reward = 0.0
    i = 0
    while not done:
        obs, reward, done, info = env.step(reco[i])
        max_obs_len = max(len(obs.sessions()), max_obs_len)
        ep_reward += reward
        total_reward += reward
        total_steps += 1
        #print('recommended %i (reward %.2f)' % (reco[i], reward))
        #print(i, obs.sessions())
        i += 1
        if i == len(reco):
            i = 0

    max_ep_reward = max(max_ep_reward, ep_reward)

print('%.5f' % (float(total_reward) / float(total_steps)))
print(total_reward)
print(total_steps)
print(max_ep_reward)
print(max_obs_len)
