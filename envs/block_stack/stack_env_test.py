from collections import namedtuple, defaultdict
import time
import gym
import sys
sys.path.insert(0, '.')
from rlf.rl.utils import save_mp4
import envs.block_stack
import numpy as np


#envs.block_stack.register_envs()

env = gym.make('StackEnvSimplestR-v0')


arg_names = [
        'high_render_dim',
        'high_render_freq',
        'test_split',
        'action_seg_loc',
        'exp_type',
        'action_set_size',
        'both_train_test',
        'train_split',
        'stack_reward',
        'stack_height',
        'sparse_height',
        'render_result_figures',
        'only_sparse_reward',
        'stack_min_steps',
        'stack_dim',
        'constrain_physics',
        'separate_skip',
        'stack_render_high',
        'contacts_off',
        'double_place_pen',
        ]

Args = namedtuple(
    'Args', ' '.join(arg_names))

HIGH_RENDER = True

args = Args(512, 10, False, './data/action_segs', 'rnd', 10, True,
        False, 0.1, 3.0, 3.0, True, False, 300, 1, False, False, True, True, 0.0)
env.set_args(args, HIGH_RENDER)
if HIGH_RENDER:
    fps = 30.0
else:
    fps = 1.0


gps = defaultdict(list)
for i, p in enumerate(env.all_polygons):
    gps[p.ply].append(i)

#env.all_polygons[0] = env.all_polygons[gps['horizontal_rectangle'][-1]]
#env.all_polygons[0].scale = 0.2    # reset scale
## env.all_polygons[0].subtract_dist = 0.8     # reset subtract_dist (scaled manually)
#
## set base obj
#env.all_polygons[1] = env.all_polygons[gps['sphere'][-2]]
#env.all_polygons[1].scale = 0.2
## env.all_polygons[1].subtract_dist = 0.50     # reset subtract_dist (scaled manually)

env.debug_log = True

env.is_fixed_action_space = True
env.seed = 42
aval = env.get_aval()

aval_types = defaultdict(list)
#for i, aval_i in enumerate(aval):
#    aval_types[env.all_polygons[aval_i].ply].append(i)
for i, p in enumerate(env.all_polygons):
    aval_types[p.ply].append(i)

print(aval_types['cylinder'])

max_cnt = 0
max_idx = 0

for t, idxs in aval_types.items():
    if len(idxs) > max_cnt:
        max_idx = t
        max_cnt = len(idxs)
env.aval_idx = aval_types['cylinder'][-10:]
env.usable_objs = [env.all_polygons[i] for i in env.aval_idx]
env.start_pos = {}
env._init_scene()

print('all types', list(aval_types.keys()))


# use_type = aval_types[max_idx]
use_type = env.aval_idx
print('Using ', max_idx)

for i in range(1):
    done = False

    # while not done:
    env.reset()
    all_frames = []
    done = False
    j = 0

    while not done and j < len(use_type):
        # a = [use_type[j], 0.0, 0.0]
        a = [j, 0.0, 0.0]
        start = time.time()

        obs, reward, done, info = env.step(a)
        end = time.time()
        print('time', end - start)
        print('')
        if not HIGH_RENDER:
            all_frames.append(obs)

        frames = env.render()
        if HIGH_RENDER:
            all_frames.append(frames)
        j += 1

    save_mp4(all_frames, './vids', 'stack_%i' %
             i, fps=fps, no_frame_drop=True)

    # save_mp4(obs,
