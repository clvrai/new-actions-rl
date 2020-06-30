from envs.gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import os.path as osp
import random
import torch
import numpy as np
from envs.action_env import split_action_set, get_aval_actions


def convert_gridworld(env, args):
    if args.orig_crossing_env:
        env.max_steps = 4 * env.width * env.width
        env.safe_wall_gen = False
    else:
        if args.max_grid_steps is not None:
            env.max_steps = args.max_grid_steps
        else:
            env.max_steps = 4 * env.width * env.width
        # if the walls should only be generated in the center
        env.safe_wall_gen = args.grid_safe_wall
        env.use_subgoal_reward = args.grid_subgoal
        env.fixed_rivers = args.grid_fixed_rivers

    env = FullyObsWrapper(env, args)

    return env


def expand_action(prev_actions, n):
    if n == 0:
        return prev_actions
    new_actions = []
    for i in range(8):
        for prev_action in prev_actions:
            new_actions.append([*prev_action, i])
    return expand_action(new_actions, n - 1)


def create_action_bank(args):
    assert args.up_to_option is not None, 'must specify up to option'

    parts = args.up_to_option.split('_')
    n = int(parts[0])
    diag = len(parts) > 1

    possible_actions = []
    for i in range(n):
        possible_actions.extend(expand_action([[]], i + 1))

    use_options = []
    args.action_bank = []
    for i, action in enumerate(possible_actions):
        skip_diag = (len(action) == n and not diag) or args.no_diag
        has_diag = ((4 in action) or (5 in action) or (6 in action) or (7 in action))

        if has_diag and skip_diag:
            continue
        if args.not_upto and len(action) != n:
            continue
        args.action_bank.append(action)


def load_training_fixed_set_grid(args):
    new_dir = osp.join(args.action_seg_loc, '%s_%s' % ('grid', args.exp_type))
    with open(osp.join(new_dir, 'set_train.npy'), 'rb') as f:
        train_set = np.load(f)
    args.training_overall_actions = train_set
    args.training_fixed_action_set = list(args.training_overall_actions)


def get_overall_aval_actions_grid(args):
    args.overall_aval_actions = get_aval_actions(args, 'grid')
    create_action_bank(args)


def get_option_properties(args, quadrant=True, octant=False,
    distance=False, man_distance=False,
    num_turns=False):

    option_properties = []
    parts = args.up_to_option.split('_')
    n = int(parts[0])

    # if quadrant:
    #     label_list = ['quadrant ' + str(x) for x in range(1,5)]
    # if octant:
    #     label_list = ['octant ' + str(x) for x in range(1,9)]
    # if distance:
    #     label_list = ['distance '+str(x) for x in range(0,n+1)]
    # if man_distance:
    #     label_list = ['man_distance ' + str(x) for x in range(0,n+1)]
    # if num_turns:
    #     label_list = ['num_turns '+str(x) for x in range(0, n+1)]

    for option_id in range(len(args.action_bank)):
        action_seq = args.action_bank[option_id]
        # 0 = right
        # 1 = down
        # 2 = left
        # 3 = up
        act_str = [str(i) for i in action_seq]
        skill_str = ''.join(act_str)
        N_count = skill_str.count('3')
        S_count = skill_str.count('1')
        W_count = skill_str.count('2')
        E_count = skill_str.count('0')

        N_count += skill_str.count('6') + skill_str.count('7')
        S_count += skill_str.count('4') + skill_str.count('5')
        W_count += skill_str.count('5') + skill_str.count('6')
        E_count += skill_str.count('4') + skill_str.count('7')


        final_pos = (N_count - S_count, E_count - W_count)

        mag = abs(final_pos[0]) >= abs(final_pos[1])

        if quadrant:
            if final_pos[0] >= 0 and final_pos[1] > 0:
                label = chr(0x2197) # Quadrant 1
            elif final_pos[0] < 0 and final_pos[1] >= 0:
                label = chr(0x2196) # Quadrant 2
            elif final_pos[0] <= 0 and final_pos[1] < 0:
                label = chr(0x2199) # Quadrant 3
            elif final_pos[0] > 0 and final_pos[1] <= 0:
                label = chr(0x2198) # Quadrant 4
            else:
                label = chr(0x21BA) # Origin

        if octant:
            if final_pos[0] >= 0 and final_pos[1] >= 0 and mag:
                label = 'octant 1'
            elif final_pos[0] >= 0 and final_pos[1] >= 0:
                label = 'octant 2'
            elif final_pos[0] < 0 and final_pos[1] >= 0 and not mag:
                label = 'octant 3'
            elif final_pos[0] < 0 and final_pos[1] >= 0:
                label = 'octant 4'
            elif final_pos[0] < 0 and final_pos[1] < 0 and mag:
                label = 'octant 5'
            elif final_pos[0] < 0 and final_pos[1] < 0:
                label = 'octant 6'
            elif final_pos[0] >= 0 and final_pos[1] < 0 and not mag:
                label = 'octant 7'
            elif final_pos[0] >= 0 and final_pos[1] < 0:
                label = 'octant 8'

        if distance:
            dist = int(np.sqrt(final_pos[0] ** 2 + final_pos[1] ** 2))
            # dist = int(abs(final_pos[0]) + abs(final_pos[1]))
            label = 'distance ' + str(dist)

        if man_distance:
            dist = int(abs(final_pos[0]) + abs(final_pos[1]))
            label = 'man_distance ' + str(dist)

        if num_turns:
            turns = int(len([action_seq[p] for p in \
                range(1, len(action_seq)) if action_seq[p] != action_seq[p-1]]))
            label = 'num_turns ' + str(turns)

        option_properties.append(label)
    label_list = np.unique(option_properties)

    return np.array(option_properties), label_list


def discretize_gridworld(args, action_space, settings, emb_mem):
    def convert_to_delta(a):
        if a == 0:
            return [-1, 0]
        elif a == 1:
            return [0, -1]
        elif a == 2:
            return [1, 0]
        elif a == 3:
            return [0, 1]
        elif a == 4:
            return [-1, -1]
        elif a == 5:
            return [1, -1]
        elif a == 6:
            return [1, 1]
        elif a == 7:
            return [-1, 1]

    for i, action_seq in enumerate(args.action_bank):
        # 0 is right
        # 1 is down
        # 2 is left
        # 3 is up
        # Calculate the change in distance from starting to ending state
        # Expand each action out to deltas.
        deltas = [convert_to_delta(a) for a in action_seq]
        total_delta = np.sum(deltas, axis=0)
        emb_mem.add_embedding(total_delta, i)
    print('%i possible actions' % len(emb_mem))


