from collections import namedtuple
from .create_action_set import gen_action_set, get_allowed_actions, UseSplit
import os.path as osp

class CreateGameSettings(object):
    def __init__(self, default_reward=0.01, no_op_reward=0.0, goal_reward=10.0,
            dense_reward_scale=0.0, invalid_action_reward=-0.01,
            blocked_action_reward=-0.01, sec_goal_reward=2.0,
            sec_goal_radius=3.0, permanent_goal=True, marker_reward='reg',
            target_reward=1.0, marker_gone_reward=0., screen_width=84, screen_height=84,
            render_width=84, render_height=84, high_res_width=1024,
            high_res_height=1024, render_ball_traces=False,
            evaluation_mode=False, render_mega_res=False, mega_res_interval=4,
            max_num_steps=30, large_steps=40, gravity=(0.0, -2.0),
            min_velocity=0.05, no_overlap_env=False, overlap_threshold=0.3,
            move_thresh=0.03, use_overlap=True, action_random_sample=True,
            action_sample_fn=gen_action_set,
            get_allowed_actions_fn=get_allowed_actions, action_set_size=40,
            action_extra={},
            split_name='full_clean', split_type=UseSplit.TRAIN,
            action_seg_loc=osp.join(osp.dirname(osp.abspath(__file__)), 'splits'),
            validation_ratio=0.5, gran_factor=1.0,
            override_level_settings=False, with_subgoals=True,
            separate_skip=False):

        ######################
        # Reward modifiers
        ######################
        self.default_reward = default_reward
        self.no_op_reward = no_op_reward
        self.goal_reward = goal_reward
        self.dense_reward_scale = dense_reward_scale
        self.invalid_action_reward = invalid_action_reward
        self.blocked_action_reward = blocked_action_reward
        self.sec_goal_reward = sec_goal_reward
        self.sec_goal_radius = sec_goal_radius
        self.permanent_goal = permanent_goal
        # For marker ball levels
        self.marker_reward = marker_reward
        self.target_reward = target_reward
        self.with_subgoals = with_subgoals
        self.marker_gone_reward = marker_gone_reward

        ######################
        # Render settings
        ######################
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.render_width = render_width
        self.render_height = render_height
        self.high_res_width = high_res_width
        self.high_res_height = high_res_height
        self.render_ball_traces = render_ball_traces
        self.evaluation_mode = evaluation_mode
        self.render_mega_res = render_mega_res
        self.mega_res_interval = mega_res_interval

        ######################
        # Simulation settings
        ######################
        self.max_num_steps = max_num_steps
        self.large_steps = large_steps
        self.gravity = gravity
        # Minimum velocity for motion to be considered stopped
        self.min_velocity = min_velocity
        self.no_overlap_env = no_overlap_env
        self.overlap_threshold = overlap_threshold
        self.move_thresh = move_thresh
        self.use_overlap = use_overlap
        self.override_level_settings = override_level_settings

        ######################
        # Action space settings
        ######################
        self.action_random_sample = action_random_sample
        self.action_sample_fn = action_sample_fn
        self.get_allowed_actions_fn = get_allowed_actions_fn
        self.action_set_size = action_set_size
        self.action_extra = action_extra
        self.split_name = split_name
        self.split_type = split_type
        self.action_seg_loc = action_seg_loc
        self.validation_ratio = validation_ratio
        self.separate_skip = separate_skip

        ######################
        # Tool generation settings
        ######################
        self.gran_factor = gran_factor



