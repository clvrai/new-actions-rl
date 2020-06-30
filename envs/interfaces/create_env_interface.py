from envs.interfaces.create_action_set_emb import get_allowed_actions, gen_action_set, get_train_test_action_sets
from envs.create_game import ToolGenerator
from rlf.baselines.common.atari_wrappers import wrap_deepmind
from envs.action_env_interface import ActionEnvInterface
from rlf import EnvInterface
import torch
from envs.create_game import UseSplit, CreateLevelFile, CreateGameSettings
import re

def get_render_mode(args):
    mode = 'rgb_array'
    if args.render_high_res:
        mode += '_high'
    if args.render_text:
        mode += '_text'
    if args.render_ball_traces:
        mode += '_ball_trace'
    if args.render_changed_colors:
        mode += '_changed_colors'
    if args.render_mega_res:
        mode += '_mega'
    return mode


def convert_args_to_create(args, **add_kwargs):
    if args.test_split:
        use_split = UseSplit.TEST
    elif args.eval_split:
        use_split = UseSplit.VALIDATION
    elif args.both_train_test:
        use_split = UseSplit.TRAIN_TEST
    else:
        use_split = UseSplit.TRAIN

    return CreateGameSettings(
        marker_reward=args.marker_reward,
        render_ball_traces=args.render_ball_traces,
        render_mega_res=args.render_mega_res,
        mega_res_interval=args.mega_res_interval,
        large_steps=args.large_steps,
        action_random_sample=args.action_random_sample,
        action_set_size=args.action_set_size,
        split_name=args.split_type,
        split_type = use_split,
        action_seg_loc=args.action_seg_loc,
        action_extra=args,
        use_overlap=args.threshold_overlap,
        evaluation_mode=args.evaluation_mode,
        validation_ratio=args.eval_split_ratio,
        max_num_steps=args.create_max_num_steps,
        permanent_goal=args.create_permanent_goal,
        high_res_width=args.image_resolution,
        high_res_height=args.image_resolution,
        gran_factor=args.gran_factor,
        target_reward=args.create_target_reward,
        sec_goal_reward=args.create_sec_goal_reward,

        get_allowed_actions_fn=get_allowed_actions,
        action_sample_fn=gen_action_set,
        separate_skip=args.separate_skip,
        **add_kwargs
    )


def label_modifier(label):
    label = label.replace('Hinge_Constrained', 'See_Saw').replace('Fixed_','').replace('Box','Square').replace('Hinge','Lever')
    label = re.sub(r"Bouncy_(.+)", r"\1_Bouncy", label)
    return label


class CreateGameInterface(EnvInterface):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        super().setup(args, task_id)

    def env_trans_fn(self, env, set_eval):
        settings = convert_args_to_create(self.args)
        env.set_settings(settings)
        self.args.aval_actions = env.inventory
        self.train_action_set, self.test_action_set, self.train_test_action_set = get_train_test_action_sets(settings)

        if self.args.action_random_sample and self.args.load_fixed_action_set:
            # save the sampling used from training
            self.args.training_fixed_action_set = env.get_fixed_sampling(self.train_action_set)

        if isinstance(env, CreateLevelFile):
            env.set_task_id(self.task_id)

        return wrap_deepmind(env, episode_life=False, clip_rewards=False,
                             frame_stack=False, scale=True, ignore_fire=True, ignore_wrap=False,
                             grayscale=True)

    def get_play_env_name(self):
        if self.args.play_env_name is not None:
            return self.args.play_env_name
        elif self.args.exp_type == 'NewMain':
            return 'StateCreateGamePlay-v0'
        else:
            raise ValueError('Unknown exp type')

    def get_env_option_names(self):
        tool_gen = ToolGenerator(self.args.gran_factor)
        indv_labels = [label_modifier(x.tool_type) for x in tool_gen.tools]
        label_list = sorted(list(set(indv_labels)))

        label_list.remove('no_op')

        return indv_labels, label_list

    def get_gt_embs(self):
        tool_gen = ToolGenerator(self.args.gran_factor)
        import numpy as np
        gt_embs = np.array(tool_gen.gt_embs)
        gt_embs = torch.Tensor(gt_embs)
        return gt_embs

    def get_special_stat_names(self):
        return ['ep_target_hit', 'ep_goal_hit']

    def get_render_mode(self):
        return get_render_mode(self.args)


class CreatePlayInterface(ActionEnvInterface):
    def setup(self, args, task_id):
        super().setup(args, task_id)
        settings = convert_args_to_create(self.args)
        args.overall_aval_actions = settings.get_allowed_actions_fn(settings)


    def get_env_option_names(self):
        tool_gen = ToolGenerator(self.args.gran_factor)
        indv_labels = [label_modifier(x.tool_type) for x in tool_gen.tools]
        label_list = sorted(list(set(indv_labels)))

        label_list.remove('no_op')

        return indv_labels, label_list

    def get_render_mode(self):
        return get_render_mode(self.args)

    def env_trans_fn(self, env, set_eval):
        env.update_args(self.args)
        return env
