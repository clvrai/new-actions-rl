from ..create_game import CreateGame
import json
from ..tools.tool_factory import ToolTypes
import numpy as np
from numpy.random import uniform
from collections import defaultdict
from ..tools.goal import GOAL_RADIUS
from ..tools.segment import LINE_THICKNESS
from ..constants import *
import random

# Ball FLOOR OFFSET
OFFSET = 0.11
LIGHT_BOX_MASS = 1.0
DEF_NOISE = 0.05
HIGH_NOISE = 0.2
MID_NOISE = 0.1
LARGE_NOISE = 0.5


def get_noise(rnd_map, name):
    if name in rnd_map:
        noise = np.array(rnd_map[name])
        return noise
    else:
        return 0.0


class CreateLevelFile(CreateGame):
    def __init__(self, ):
        super().__init__()
        self.rnd_map = {}

    def set_settings(self, settings):
        super().set_settings(settings)

        self.load_file()
        reward_type = self.get_reward_type()

        self.task_id = None
        self.eval_rnd_map = None

        if reward_type == 'dense':
            self.dense_reward_scale = 0.1
        elif reward_type.startswith('dense'):
            self.dense_reward_scale = float(reward_type.split(':')[1])
        elif reward_type == 'sparse':
            self.dense_reward_scale = 0.0
        else:
            raise ValueError('Invalid reward type')

    def get_json_file(self):
        return None

    def _is_json_str(self):
        return '{' in self.get_json_file()

    def get_is_rnd(self):
        raise ValueError('not implemented')

    def get_reward_type(self):
        return 'sparse'

    def get_tools(self, env_jf, tool_factory):
        """
        Get all the tools we have defined in the JSON file.
        """
        tool_name_mapping = {t.value: t for t in ToolTypes}
        tools = []

        marker_sec_goals = []
        if 'marker_sec_goals' in self.jf and self.settings.with_subgoals:
            for marker_sec_goal in self.jf['marker_sec_goals']:
                pos = eval(marker_sec_goal)
                marker_sec_goals.append(tool_factory.create(ToolTypes.GOAL, pos,
                        {'color': subgoal_color, 'radius':
                            self.settings.sec_goal_radius}))

        self.marker_sec_goals = marker_sec_goals

        target_sec_goals = []
        if 'target_sec_goals' in self.jf and self.settings.with_subgoals:
            for target_sec_goal in self.jf['target_sec_goals']:
                pos = eval(target_sec_goal)
                target_sec_goals.append(tool_factory.create(ToolTypes.GOAL, pos,
                        {'color': subgoal_color, 'radius':
                            self.settings.sec_goal_radius}))

        self.target_sec_goals = target_sec_goals

        for env_tool in env_jf:
            for prop, val in env_tool.items():
                if prop != 'name' and prop != 'color' and isinstance(val, str):
                    env_tool[prop] = eval(val)
                elif prop == 'color' and isinstance(val, str):
                    env_tool[prop] = val

            name = env_tool['name']
            pos = np.array(env_tool['pos'])
            lookup_name = name + ''
            if 'id' in env_tool:
                tool_id = env_tool['id']
                lookup_name = name + ':' + str(tool_id)
            noise = get_noise(self.eval_rnd_map, lookup_name)

            pos += noise
            pass_params = {k: v for k, v in env_tool.items() if k not in ['name','pos', 'id']}

            tool_type = tool_name_mapping[name]
            tools.append(tool_factory.create(tool_type, pos, pass_params))

        tools.extend(marker_sec_goals)
        tools.extend(target_sec_goals)
        return tools

    def load_file(self):
        """
        Load in the data specified in the JSON file.
        """
        self._check_setup()
        json_str = self.get_json_file()
        if json_str is None:
            return

        if not self._is_json_str():
            with open(json_str, 'r') as f:
                jf = json.load(f)
        else:
            jf = json.loads(json_str)


        self.jf = jf

        target = jf['target']
        if isinstance(target, str):
            target = eval(target)

        goal = jf['goal']
        if isinstance(goal, str):
            goal = eval(goal)

        self.gen_target_pos = np.array(target)
        self.gen_goal_pos = np.array(goal)

        if 'place_walls' in jf:
            self.place_walls = jf['place_walls']

        if self.get_is_rnd():
            self.rnd_map = jf['rnd']
        self.env_jf = jf['env']

    def gen_noise_apply_map(self):
        """
        Generate how much noise we should apply for this task configuration.
        """
        eval_rnd_map = {}
        for k, v in self.rnd_map.items():
            objs = k.replace(' ', '')
            objs = objs.split(',')
            eval_v = eval(v)
            for obj in objs:
                eval_rnd_map[obj] = eval_v
        return eval_rnd_map


    def get_parts(self, tool_factory):
        """
        Called every reset by the environment.
        """
        if not self.settings.override_level_settings:
            if 'max_num_steps' in self.jf:
                self.max_num_steps = self.jf['max_num_steps']
            if 'overlap_thresh' in self.jf:
                self.settings.overlap_threshold = self.jf['overlap_thresh']


        if 'marker_must_hit' in self.jf:
            self.marker_must_hit = self.jf['marker_must_hit']
        if 'sec_goal_reward' in self.jf:
            self.sec_goal_reward = self.jf['sec_goal_reward']
        if 'target_reward' in self.jf:
            self.target_reward = self.jf['target_reward']
        if 'goal_is_basket' in self.jf:
            self.goal_is_basket = self.jf['goal_is_basket']
        if 'ball_is_basket' in self.jf:
            self.ball_is_basket = self.jf['ball_is_basket']
        if 'moving_goal' in self.jf:
            self.moving_goal = self.jf['moving_goal']
        if 'target_ball_radius' in self.jf:
            self.target_ball_radius = self.jf['target_ball_radius']

        if self.task_id is None or self.eval_rnd_map is None:
            self.eval_rnd_map = self.gen_noise_apply_map()

        target_pos = self.gen_target_pos + get_noise(self.eval_rnd_map, 'target')
        goal_pos = self.gen_goal_pos + get_noise(self.eval_rnd_map, 'goal')

        if self.ball_is_basket:
            target_ball = tool_factory.create(ToolTypes.BASKET_BALL, target_pos)
        elif self.target_ball_radius is not None:
            target_ball = tool_factory.create(ToolTypes.TARGET_BALL, target_pos, {'radius': self.target_ball_radius})
        else:
            target_ball = tool_factory.create(ToolTypes.TARGET_BALL, target_pos)
        env_tools = self.get_tools(self.env_jf, tool_factory)

        return env_tools, target_ball, goal_pos


    def set_task_id(self, task_id):
        self.task_id = task_id
        self.eval_rnd_map = self.gen_noise_apply_map()



