from arguments import get_args
import random
import os.path as osp
import os
import numpy as np
from envs.gym_minigrid.action_sets import create_action_bank
from envs.create_game.tool_gen import ToolGenerator
from envs.block_stack.poly_gen import *
from envs.recogym.action_set import get_rnd_action_sets

# Ex usage python scripts/gen_action_sets.py --action-seg-loc envs/action_segs_new --env-name MiniGrid

args = get_args()

if args.env_name.startswith('MiniGrid'):
    create_action_bank(args)
    n_skills = len(args.action_bank)

    if args.exp_type == 'rnd':
        train, test = get_rnd_action_sets(n_skills)
    elif args.exp_type == 'all':
        train = np.arange(n_skills)
        test = np.arange(n_skills)
        random.shuffle(train)
        random.shuffle(test)
    else:
        raise ValueError('Invalid exp type')

    new_dir = osp.join(args.action_seg_loc, 'grid_%s' % args.exp_type)
    if not osp.exists(new_dir):
        os.makedirs(new_dir)

    with open(osp.join(new_dir, 'set_train.npy'), 'wb') as f:
        np.save(f, train)

    with open(osp.join(new_dir, 'set_test.npy'), 'wb') as f:
        np.save(f, test)

    print('Training set: ', len(train))
    print('Test set: ', len(test))

elif args.env_name.startswith('Create'):
    tool_gen = ToolGenerator(args.gran_factor)

    train_tools, test_tools = tool_gen.get_train_test_split(args)

    # Randomize here
    np.random.shuffle(train_tools)
    np.random.shuffle(test_tools)

    add_str = ('_' + args.split_type) if (args.split_type is not None and 'New' in args.exp_type) else ''

    new_dir = osp.join(args.action_seg_loc, 'create_' + args.exp_type + add_str)
    if not osp.exists(new_dir):
        os.makedirs(new_dir)
    train_filename = osp.join(new_dir, 'set_train.npy')
    with open(train_filename, 'wb') as f:
        np.save(f, train_tools)

    test_filename = osp.join(new_dir, 'set_test.npy')
    with open(test_filename, 'wb') as f:
        np.save(f, test_tools)

elif args.env_name.startswith('Stack'):
    all_polys, _, polygon_types = gen_polys('envs/block_stack/assets/stl/')

    if args.exp_type == 'rnd':
        train, test = rnd_train_test_split(all_polys, polygon_types)
    elif args.exp_type == 'full':
        train, test = full_train_test_split(all_polys)
    elif args.exp_type == 'all':
        train = np.arange(len(all_polys))
        test = np.arange(len(all_polys))
        random.shuffle(train)
        random.shuffle(test)
    else:
        raise ValueError('Invalid exp type')

    train_types = set([all_polys[i].type for i in train])
    test_types = set([all_polys[i].type for i in test])
    print('')
    print('In train (%i)' % len(train))
    for t in train_types:
        print('     - %s' % (t))
    print('')
    print('In test (%i)' % len(test))
    for t in test_types:
        print('     - %s' % (t))
    new_dir = osp.join(args.action_seg_loc, 'stack_%s' % args.exp_type)
    if not osp.exists(new_dir):
        os.makedirs(new_dir)

    with open(osp.join(new_dir, 'set_train.npy'), 'wb') as f:
        np.save(f, train)

    with open(osp.join(new_dir, 'set_test.npy'), 'wb') as f:
        np.save(f, test)

elif args.env_name.startswith('Reco'):
    if args.exp_type == 'rnd':
        train, test = get_rnd_action_sets(args.reco_n_prods)
    else:
        raise ValueError('Invalid exp type')

    new_dir = osp.join(args.action_seg_loc, 'reco_%s' % args.exp_type)
    if not osp.exists(new_dir):
        os.makedirs(new_dir)

    with open(osp.join(new_dir, 'set_train.npy'), 'wb') as f:
        np.save(f, train)

    with open(osp.join(new_dir, 'set_test.npy'), 'wb') as f:
        np.save(f, test)

    print('Training set: ', len(train))
    print('Test set: ', len(test))
else:
    print('Unspecified Environment!')
