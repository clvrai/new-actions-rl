import gym
import numpy as np
import os.path as osp
import gym.spaces as spaces
import random


def split_action_set(test_set, args):
    """
    TODO: This function should be removed and it should be incorperated into
    the env interface class to provide common behavior for action set loading
    to all environments
    """
    # split in a predictable way
    rng = np.random.RandomState(41)

    ret_set = test_set[:]
    rng.shuffle(ret_set)
    N = int(len(ret_set) * args.eval_split_ratio)
    eval_set = ret_set[:N]
    test_set = ret_set[N:]
    return eval_set, test_set


def get_aval_actions(args, env_name):
    new_dir = osp.join(args.action_seg_loc, '%s_%s' % (env_name,
        args.exp_type))
    with open(osp.join(new_dir, 'set_train.npy'), 'rb') as f:
        train_set = np.load(f)
    with open(osp.join(new_dir, 'set_test.npy'), 'rb') as f:
        test_set = np.load(f)

    if args.both_train_test:
        use_split = sorted(np.unique([*train_set, *test_set]))
    elif args.train_split:
        use_split = train_set
    elif args.test_split:
        _, use_split = split_action_set(test_set, args)
    elif args.eval_split:
        use_split, _ = split_action_set(test_set, args)
    else:
        raise ValueError('Undefined split')

    return use_split



class ActionEnv(gym.Env):
    """
    Abstracted this behavior because it's common between multiple classes.
    """

    def __init__(self):
        self.is_fixed_action_space = False

    def get_env_name(self):
        return None

    def set_args(self, args, set_eval):
        self.args = args

        self.use_split = get_aval_actions(args, self.get_env_name())

        if args.action_set_size is None:
            args.action_set_size = len(self.use_split)

        self.action_set_size = args.action_set_size

        self._reset_action_space()

    def load_training_fixed_set(self, args):
        new_dir = osp.join(args.action_seg_loc, '%s_%s' % (self.get_env_name(), args.exp_type))
        with open(osp.join(new_dir, 'set_train.npy'), 'rb') as f:
            train_set = np.load(f)
        args.training_overall_actions = train_set
        args.training_fixed_action_set = self.set_fixed_action_space(args, args.training_overall_actions)

    def set_fixed_action_space(self, args, action_set):
        sub_split = np.copy(action_set)
        return sub_split

    def _sample_ac(self, sub_split, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        if hasattr(self.args, 'sample_clusters') and self.args.sample_clusters and hasattr(self.args, 'emb_mem'):
            if self.args.emb_mem.action_groups is None:
                self.args.emb_mem.compute_clusterings(sub_split)
            ac_groups = self.args.emb_mem.action_groups
        elif hasattr(self.args, 'gt_clusters') and self.args.gt_clusters:
            raise ValueError('Not implemented!')
        else:
            return rng.choice(sub_split, self.args.action_set_size, replace=False)

        group_keys = list(ac_groups.keys())

        set_arr = []
        while(len(set_arr) < self.args.action_set_size):
            if self.args.half_tools and self.args.half_tool_ratio is not None:
                group_keys_samples = random.sample(
                    group_keys, int(self.args.half_tool_ratio * len(group_keys)))
            else:
                group_keys_samples = group_keys[:]
            type_selections = rng.choice(group_keys_samples, self.args.action_set_size, replace=True)
            for type_sel in type_selections:
                for trial in range(10):
                    idx = rng.choice(ac_groups[type_sel])
                    if idx not in set_arr:
                        break
                set_arr.append(idx)
                if len(set_arr) == self.args.action_set_size:
                    break
        return np.array(set_arr)

    def _reset_action_space(self):
        if self.is_fixed_action_space:
            # Our action space is already set and is fixed don't reset
            return

        sub_split = np.copy(self.use_split)

        if self.action_set_size is not None:
            sub_split = self._sample_ac(sub_split)
        self._set_action_space(sub_split)

    def _set_action_space(self, sub_split):
        self.aval_idx = sub_split
        self.action_space = spaces.Discrete(len(self.aval_idx))

    def reset(self):
        self._reset_action_space()

    def get_aval(self):
        return self.aval_idx

    def _log(self, s):
        if self.debug_log:
            print(s)

    def get_train_test_action_sets(self, args, env_name):
        new_dir = osp.join(args.action_seg_loc, '%s_%s' % (env_name,
            args.exp_type))
        with open(osp.join(new_dir, 'set_train.npy'), 'rb') as f:
            train_set = np.load(f)
        with open(osp.join(new_dir, 'set_test.npy'), 'rb') as f:
            test_set = np.load(f)

        return train_set, test_set, sorted(np.unique([*train_set, *test_set]))
