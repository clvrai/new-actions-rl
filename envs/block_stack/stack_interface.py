from envs.block_stack.poly_gen import gen_polys
from envs.action_env import get_aval_actions
from envs.action_env_interface import ActionEnvInterface
from envs.block_stack.stack_env import StackEnv
from envs.block_stack.stack_env_new import StackEnvNew
from envs.block_stack.stack_env_simplest import StackEnvSimplest
import numpy as np
import os.path as osp
import torch
from rlf.baselines.common.atari_wrappers import wrap_deepmind

class StackInterface(ActionEnvInterface):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        super().setup(args, task_id)
        args.overall_aval_actions = get_aval_actions(args, 'stack')


    def env_trans_fn(self, env, set_eval):
        if isinstance(env, StackEnv) or isinstance(env, StackEnvNew) or isinstance(env, StackEnvSimplest):
            env = super()._generic_setup(env, set_eval)
        else:
            env.update_args(self.args)

        return wrap_deepmind(env, episode_life=False, clip_rewards=False,
            frame_stack=False, scale=True, ignore_fire=True, ignore_wrap=False,
            grayscale=True)

    def get_play_env_name(self):
        return 'BlockPlayImg-v0'


    def get_env_option_names(self):
        all_polygons, _, _ = gen_polys('envs/block_stack/assets/stl/')
        indv_labels = [p.type for p in all_polygons]
        label_list = sorted(list(set(indv_labels)))

        return indv_labels, label_list

    def get_id(self):
        return 'BS'

    def get_gt_embs(self):
        asset_path = osp.join(osp.dirname(
            osp.realpath(__file__)), 'assets/stl')
        polys, _, _ = gen_polys(asset_path)
        all_types = list(set([p.ply for p in polys]))
        one_hots = {}
        for i, t in enumerate(all_types):
            one_hot = np.zeros(len(all_types))
            one_hot[i] = 1.0
            one_hots[t] = one_hot

        gt = []
        for p in polys:
            gt.append([p.scale, *p.angle, *one_hots[p.ply]])
        return torch.Tensor(gt)

    def get_special_stat_names(self):
        return ['ep_final_height', 'ep_max_height', 'ep_repeat', 'ep_no_op']



class StackPlayInterface(StackInterface):
    def setup(self, args, task_id):
        super().setup(args, task_id)
        #all_polygons, _, _ = gen_polys('envs/block_stack/assets/stl/')
        #args.overall_aval_actions = np.arange(len(all_polygons))
        args.overall_aval_actions = get_aval_actions(args, 'stack')
