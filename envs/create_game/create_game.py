from gym.spaces import Discrete, Box, Dict
import numpy as np
from .tools.tool_factory import *
from .base_env import BaseEnv
from .tool_gen import ToolGenerator
from .constants import placed_wall_color

from .settings import CreateGameSettings

class SampleDict(Dict):
    def sample(self):
        rnd_pos = self.spaces['pos'].sample()
        rnd_sel = self.spaces['index'].sample()
        return [rnd_sel, *rnd_pos]

GET_TOOL_LIST = -2

class CreateGame(BaseEnv):
    def __init__(self):
        self.place_walls = False
        self.different_walls = False
        self.total_subgoal_add_reward = 0.0

        super().__init__()

        self.placed_tools = []

        self.target_sec_goals = []
        self.done_target_sec_goals = []

        self.marker_sec_goals = []
        self.done_marker_sec_goals = []

        self.goal_is_basket = False
        self.ball_is_basket = False
        self.moving_goal = False
        self.target_ball_radius = None
        self.dense_reward_scale = None
        self.no_action_space_resample = False

        self.inventory = None
        self.server_mode = False
        self.has_reset = False
        self.episode_len  = 0

        # Initialize to the defaults
        self.set_settings(CreateGameSettings())

        # place holder action space.
        self._create_action_space(1)

    def _create_action_space(self, n_opts):
        if self.settings.separate_skip:
            self.action_space = SampleDict({
                'index': Discrete(n_opts),
                'skip': Discrete(2),
                'pos': Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            })
        else:
            self.action_space = SampleDict({
                'index': Discrete(n_opts),
                'pos': Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            })

    def set_settings(self, settings):
        super().set_settings(settings)
        self.sec_goal_reward = settings.sec_goal_reward
        self.dense_reward_scale = settings.dense_reward_scale

        self.tool_gen = ToolGenerator(settings.gran_factor)
        self.allowed_actions = settings.get_allowed_actions_fn(settings)
        self._generate_inventory()

    def get_parts(self, tool_factory):
        """
        Returns additional objects to place and simulate in the scene. Must be
        overriden by a child class.
        """
        raise NotImplemented()

    def seed(self, seed=None):
        np.random.seed(seed)

    def get_all_objs(self):
        """
        Returns all objects that should be rendered. Returns the specific
        necessary objects for CREATE.
        """
        objs = super().get_all_objs()
        if self.goal_obj is None:
            return [self.target_obj, *self.placed_tools, *objs]
        else:
            return [self.target_obj, self.goal_obj, *self.placed_tools, *objs]

    def update_available_tools(self, tools):
        self.inventory = tools
        self._create_action_space(len(tools))


    def _generate_inventory(self):
        if self.no_action_space_resample:
            return
        if self.settings.action_random_sample:
            # Generate a new set of tools this episode for this agent to use
            tools = self.settings.action_sample_fn(self.settings,
                    self.tool_gen, self.allowed_actions,
                    np.random.RandomState())
            self.update_available_tools(tools)
        else:
            tools = self.get_fixed_sampling()
            self.update_available_tools(tools)

    def get_fixed_sampling(self, override_allowed=None):
        # Used a fixed random number so we always have the same fixed action
        # space. This is necessary so we can get the same fixed action space
        # during test time.
        rnd_state = np.random.RandomState(123)

        use_allowed = self.allowed_actions
        if override_allowed is not None:
            use_allowed = override_allowed

        use_allowed = np.array(use_allowed)
        rnd_state.shuffle(use_allowed)

        return use_allowed

        # Tool set has not been generated yet, generate now.
        #tools = self.settings.action_sample_fn(self.settings, self.tool_gen,
        #        use_allowed, rnd_state, fixed_sample=True)
        #return tools


    def reset(self):
        self._check_setup()
        # Set our action space
        self._generate_inventory()

        # Get the objects in the environment
        env_tools, target_obj, goal_pos = self.get_parts(self.tool_factory)
        if self.place_walls and not self.different_walls:
            # Add walls.
            env_tools.append(self.tool_factory.create(ToolTypes.WALL, [1.0, 0.0],
                                                      {'length': 90, 'color': placed_wall_color, 'sensor': False}))
            env_tools.append(self.tool_factory.create(ToolTypes.WALL, [-1.0, 0.0],
                                                      {'length': 90, 'color': placed_wall_color, 'sensor': False}))
            env_tools.append(self.tool_factory.create(ToolTypes.FLOOR, [0.0, 1.0],
                                                      {'length': 90, 'friction': 1.0, 'color': placed_wall_color, 'sensor': False}))

        super().reset(env_tools)

        # Set up our target and goal object
        self.target_obj = target_obj
        self.goal_pos = np.array(goal_pos)
        self.target_obj_start_pos = target_obj.pos[:]

        self.target_obj.add_to_space(self.space)

        # We can set a variety of different settings for the goal.
        # It can be a basket image
        if self.goal_is_basket:
            goal_obj = self.tool_factory.create(
                ToolTypes.BASKET, self.goal_pos)
        # It can be moving and a part of the scene
        elif self.moving_goal:
            goal_obj = self.tool_factory.create(
                ToolTypes.GOAL_BALL, self.goal_pos)
        # It can be just fixed and static
        else:
            goal_obj = self.tool_factory.create(ToolTypes.GOAL_STAR, self.goal_pos)

        goal_obj.add_to_space(self.space)
        self.goal_pos = convert_action(self.goal_pos, self.settings)
        self.goal_obj = goal_obj

        self.placed_tools = []
        self.env_pos = np.array([x.pos for x in env_tools])

        self.episode_len = 0
        self.episode_reward = 0.0

        self.target_hit = 1.
        self.goal_hit = 0

        self.invalid_action_count = 0
        self.overlap_action_count = 0
        self.blocked_action_count = 0
        self.no_op_count = 0
        self.episode_dense_reward = 0.0

        self.total_subgoal_add_reward = 0.0

        self.prev_dist = self.calc_distance(
            self.target_obj.body.position, self.goal_pos)
        self.init_dist = self.calc_distance(
            self.target_obj.body.position, self.goal_pos)

        self.done_target_sec_goals = []
        self.done_marker_sec_goals = []

        self.zero_vel_steps = 0

        obs = self.render()
        self.has_reset = True

        return obs

    def get_placed_pos(self):
        positions = []
        if self.env_pos.shape[0] >= 1 and self.settings.no_overlap_env:
            positions.append(self.env_pos)

        obj_pos = np.array([self.target_obj.body.position])
        if self.settings.no_overlap_env:
            positions.append(obj_pos)

        if len(self.placed_tools) >= 1:
            positions.append(
                np.array([np.array(x.pos).squeeze() for x in self.placed_tools]))

        return np.apply_along_axis(self.normalize_action, 1,
                                   np.concatenate(positions, axis=0)) if len(positions) > 0 else np.array([])

    def check_overlap(self, action_pos):
        all_pos = self.get_placed_pos()
        if len(all_pos) == 0:
            return False
        cur_pos = np.repeat(np.expand_dims(action_pos, 0),
                            all_pos.shape[0], axis=0).squeeze()
        dist = np.sqrt(np.sum((cur_pos - all_pos) ** 2, axis=-1))
        return (dist <= self.settings.overlap_threshold).any()

    def check_collisions(self, tool, action_pos, all_objs):
        for obj in all_objs:
            if len(tool.shape.shapes_collide(obj.shape).points) > 0:
                return True
        return False

    def check_out_of_range(self, action_pos):
        return not self.action_space.spaces['pos'].contains(action_pos)

    def motion_stopped(self):
        all_objs = self.get_all_objs()
        vel = np.array([np.sqrt(sum(x.shape.body.velocity ** 2))
                        for x in all_objs])
        if (vel < self.settings.min_velocity).all():
            self.zero_vel_steps += 1
        return (self.zero_vel_steps > 1)

    def is_valid_place_tool(self, action, actually_place=False):
        """
        Places a tool without stepping in the environment
        """
        action_index = int(np.round(action[0]))
        action_pos = action[1:]

        action_pos = np.clip(action_pos, -1.0, 1.0)
        use_tool_type = self.inventory[action_index]
        action_pos = action[1:]

        placed_obj = False

        if self.check_out_of_range(action_pos):
            placed_obj = False
        elif self.settings.use_overlap and self.check_overlap(action_pos):
            placed_obj = False
        else:
            tool = self.tool_gen.get_tool(use_tool_type, action_pos, self.settings)
            all_objs = self.get_all_objs()
            tool.add_to_space(self.space)
            if self.check_collisions(tool, action_pos, all_objs):
                placed_obj = False
            else:
                if actually_place:
                    self.placed_tools.append(tool)
                placed_obj = True
            if not actually_place:
                tool.remove_from_space(self.space)

        return placed_obj

    def get_tool_list(self):
        return self.tool_gen.tools

    def get_aval(self):
        return self.inventory

    def step(self, action):
        """
        - action: tuple of format (integer between 0 and n_actions - 1, [x_pos, y_pos])
        """
        if not self.has_reset:
            raise ValueError('Must call reset() on the environment before stepping')
        if self.episode_len > self.max_num_steps and not self.server_mode:
            raise ValueError('Must call reset() after environment returns done=True')

        action_index = int(np.round(action[0]))
        reward = self.settings.default_reward
        info = {}
        # Observation is going to be a sequence of frames
        obs = []

        use_tool_type = self.inventory[action_index]
        action_pos = action[-2:]

        if not (self.settings.separate_skip and int(np.round(action[1])) == 1):
            action_pos = np.clip(action_pos, -1.0, 1.0)

            if self.check_out_of_range(action_pos):
                reward += self.settings.invalid_action_reward
                self.invalid_action_count += 1
            elif (not self.settings.separate_skip) and (self.tool_gen.tools[use_tool_type].tool_type == 'no_op'):
                reward += self.settings.no_op_reward
                self.no_op_count += 1
            elif self.settings.use_overlap and self.check_overlap(action_pos):
                reward += self.settings.blocked_action_reward
                self.overlap_action_count += 1
            else:
                tool = self.tool_gen.get_tool(use_tool_type, action_pos,
                        self.settings)
                all_objs = self.get_all_objs()
                tool.add_to_space(self.space)
                if self.check_collisions(tool, action_pos, all_objs):
                    reward += self.settings.blocked_action_reward
                    self.overlap_action_count += 1
                    tool.remove_from_space(self.space)
                else:
                    self.placed_tools.append(tool)
        else:
            # We skipped the current action.
            self.no_op_count += 1

        obs, step_reward, done = self._create_step_forward()

        reward += step_reward

        # Add all possible log data to the info array
        info['frames'] = obs
        info['cur_goal_hit'] = self.goal_hit

        self.episode_reward += reward

        if done:
            # Only display episode long info once the episode ends.
            info['ep_len'] = self.episode_len
            info['ep_target_hit'] = self.target_hit
            info['ep_goal_hit'] = self.goal_hit
            info['ep_reward'] = self.episode_reward
            info['ep_subgoal_reward'] = self.total_subgoal_add_reward

            info['ep_no_op'] = self.no_op_count
            info['ep_invalid_action'] = self.invalid_action_count
            info['ep_blocked_action'] = self.blocked_action_count
            info['ep_overlap_action'] = self.overlap_action_count
            info['ep_dense_reward'] = self.episode_dense_reward
            info['ep_placed_tools'] = len(self.placed_tools)

        #info['aval'] = self.inventory

        return obs, reward, done, info


    def _create_step_forward(self):
        done = False
        reward = 0.0
        obs = super().step_forward()

        self.episode_len += 1

        # Check for termination conditions
        if self.episode_len > self.max_num_steps:
            done = True

        if not self.within_bounds(self.target_obj.body.position):
            done = True

        if self.motion_stopped():
            done = True

        # Compute rewards.
        # Compute the necessary distances to get dense rewards if they are
        # being used.
        cur_target_pos = self.target_obj.body.position
        move_dist = self.calc_distance(self.target_obj_start_pos, cur_target_pos)

        # Is the target ball moving?
        if self.target_hit and not self.goal_hit:
            distance = self.calc_distance(cur_target_pos, self.goal_pos)

            # Dense reward based off distance traveled to the goal.
            reward += self.dense_reward_scale * \
                (self.init_dist - distance)
            self.episode_dense_reward += self.dense_reward_scale * \
                (self.init_dist - distance)

            collided_goal = len(self.target_obj.shape.shapes_collide(
                self.goal_obj.shape).points) > 0
            has_goal = self.goal_obj is not None
            contact_goal = (hasattr(self.goal_obj.shape, 'target_contact')
                            and self.goal_obj.shape.target_contact)

            # Was the actual goal hit?
            if has_goal and (collided_goal or contact_goal):
                self.goal_hit += 1.
                reward += self.settings.goal_reward - \
                    0.1 * len(self.placed_tools)
                if not self.settings.permanent_goal:
                    # Remove the goal now
                    self.goal_obj.remove_from_space(self.space)
                    self.goal_obj = None
                if self.server_mode:
                    done = True

        # Add any additional reward based on subgoals or reward signals from
        # derived environments.
        subgoal_add_reward = self.compute_added_reward()
        self.total_subgoal_add_reward += subgoal_add_reward
        reward += subgoal_add_reward

        obs = np.array(obs)
        return obs, reward, done



    def compute_added_reward(self):
        """
        Computes an additional reward for the task. By default will compute the
        reward for achieving any subgoals in the scene
        """
        reward = 0.0
        for i in range(len(self.target_sec_goals)):
            if i not in self.done_target_sec_goals:
                if len(self.target_obj.shape.shapes_collide(self.target_sec_goals[i].shape).points) > 0:
                    self.done_target_sec_goals.append(i)
                    reward += max(0, self.sec_goal_reward - \
                        0.1 * len(self.placed_tools))
                    self.target_sec_goals[i].remove_from_space(self.space)
                    self.env_tools.remove(self.target_sec_goals[i])

        for i in range(len(self.marker_sec_goals)):
            if i not in self.done_marker_sec_goals:
                if len(self.marker_obj.shape.shapes_collide(self.marker_sec_goals[i].shape).points) > 0:
                    self.done_marker_sec_goals.append(i)
                    reward += max(0, self.sec_goal_reward - \
                        0.1 * len(self.placed_tools))
                    self.marker_sec_goals[i].remove_from_space(self.space)
                    self.env_tools.remove(self.marker_sec_goals[i])
        return reward
