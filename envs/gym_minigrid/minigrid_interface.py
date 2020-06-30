from rlf import EnvInterface
from envs.gym_minigrid.action_sets import convert_gridworld, create_action_bank, get_overall_aval_actions_grid, load_training_fixed_set_grid
import numpy as np
import torch
import os.path as osp


class MiniGridInterface(EnvInterface):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        super().setup(args, task_id)
        get_overall_aval_actions_grid(args)
        if args.load_fixed_action_set:
            load_training_fixed_set_grid(args)
        sets = self.get_train_test_action_sets(self.args, 'grid')
        self.train_action_set, self.test_action_set, self.train_test_action_set = sets

    def get_train_test_action_sets(self, args, env_name):
        new_dir = osp.join(args.action_seg_loc, '%s_%s' % ('grid',
            args.exp_type))
        with open(osp.join(new_dir, 'set_train.npy'), 'rb') as f:
            train_set = np.load(f)
        with open(osp.join(new_dir, 'set_test.npy'), 'rb') as f:
            test_set = np.load(f)

        return train_set, test_set, sorted(np.unique([*train_set, *test_set]))

    def env_trans_fn(self, env, set_eval):
        return convert_gridworld(env, self.args)

    def get_play_env_name(self):
        return 'MiniGrid-Empty-Random-80x80-v0'

    def get_special_stat_names(self):
        return ['ep_success', 'ep_subgoal_reached', 'ep_len']

    def get_gt_embs(self):
        # Delta embs
        if not hasattr(self.args, 'action_bank'):
            create_action_bank(self.args)
        deltas_map = {
               0: [-1, 0],
               1: [0, -1],
               2: [1, 0],
               3: [0, 1],
               }
        deltas = [np.sum([deltas_map[move] for move in skill], axis=0)
               for skill in self.args.action_bank]
        delta_embs = torch.FloatTensor(deltas)

        # one hot embs
        one_hot_map = {
                0: [1, 0, 0, 0],
                1: [0, 1, 0, 0],
                2: [0, 0, 1, 0],
                3: [0, 0, 0, 1],
                4: [1, 1, 0, 0],
                5: [0, 1, 1, 0],
                6: [0, 0, 1, 1],
                7: [1, 0, 0, 1],
                }

        one_hots = [np.concatenate([deltas_map[move] for move in skill], axis=0)
            for skill in self.args.action_bank]
        one_hot_embs = torch.FloatTensor(one_hots)
        gt_embs = torch.cat([delta_embs, one_hot_embs], -1)
        return gt_embs

    def mod_render_frames(self, frames, infos, cur_frame):
        if not self.args.render_info_grid:
            return
        # Visualizing intermediate frames
        cur_frames = [info['frames'] for info in infos]
        for j in range(max([len(x) for x in cur_frames])):
            frames_j = [(cur[j]  if j < len(cur) else cur[-1]) for cur in cur_frames]
            im = tile_images(frames_j)
            frames.append(im)
        frames.append(cur_frame) # Display last frame twice


