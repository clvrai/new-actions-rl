import sys
sys.path.insert(0, '.')

import gym
import envs.recogym
from collections import namedtuple
import time


arg_names = [
        'reco_n_prods',
        'reco_prod_dim',
        'exp_type',
        'action_seg_loc',
        'both_train_test',
        'train_split',
        'test_split',
        'action_set_size'
        ]

Args = namedtuple( 'Args', ' '.join(arg_names))
args = Args(1000, 4, 'rnd', './envs/action_segs_0703', False, True, False, 200)
env = gym.make('RecoEnv-v0')
env.set_args(args, False)
#env.debug_log = True


for i in range(1000):
    env.seed(31)
    obs = env.reset()

    done = False

    while not done:
        start = time.time()
        obs, reward, done, info = env.step(0)
        end = time.time()

