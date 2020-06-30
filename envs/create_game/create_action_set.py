import numpy as np
import os.path as osp
from .tool_gen import ToolGenerator, get_all
from .tool_gen_filters import get_tools_from_filters
from collections import defaultdict
import random
import copy
from enum import Enum

class UseSplit(Enum):
    TRAIN = 0
    TEST = 1
    TRAIN_TEST = 2
    VALIDATION = 3


def gen_action_set(settings, tool_gen, allowed_actions, rng):
    if settings.action_set_size is None:
        return allowed_actions
    else:
        return np.random.choice(allowed_actions, settings.action_set_size, replace=False)

def get_allowed_actions(settings):
    tool_gen = ToolGenerator(settings.gran_factor)

    load_dir = osp.join(settings.action_seg_loc, settings.split_name)

    # Load in the test and train splits
    train_filename = osp.join(load_dir, 'set_train.npy')
    with open(train_filename, 'rb') as f:
        train_set = np.load(f)

    test_filename = osp.join(load_dir, 'set_test.npy')
    with open(test_filename, 'rb') as f:
        test_set = np.load(f)

    overall_aval_actions = None

    if settings.split_type == UseSplit.TRAIN_TEST:
        tool_ids = [t.tool_id for t in tool_gen.tools]
        overall_aval_actions = np.sort(np.unique(tool_ids))
        if not (overall_aval_actions == np.arange(len(overall_aval_actions))).all():
            raise ValueError('must include all actions')
    elif settings.split_type == UseSplit.TRAIN:
        overall_aval_actions = train_set
    elif settings.split_type == UseSplit.TEST:
        _, overall_aval_actions = split_action_set(test_set,
                settings.validation_ratio)
    elif settings.split_type == UseSplit.VALIDATION:
        overall_aval_actions, _ = split_action_set(test_set,
                settings.validation_ratio)

    return overall_aval_actions

def split_action_set(test_set, validation_ratio):
    # split in a predictable way
    rng = np.random.RandomState(41)

    ret_set = test_set[:]
    rng.shuffle(ret_set)
    N = int(len(ret_set) * validation_ratio)
    eval_set = ret_set[:N]
    test_set = ret_set[N:]
    return eval_set, test_set


