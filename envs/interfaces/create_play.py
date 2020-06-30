import gym
from envs.create_game.tools.tool_factory import ToolTypes
from envs.create_game.base_env import BaseEnv
from gym.spaces import Discrete, Box, Dict
import numpy as np
import cv2
from collections import OrderedDict
from envs.create_game.tool_gen import ToolGenerator
from envs.create_game.constants import *
from envs.interfaces.create_env_interface import convert_args_to_create

MAX_NUM_TRIALS = 8

class PlayEnvWrapper(BaseEnv):
    def __init__(self, render_interval=15, large_steps=30):
        super().__init__()
        self.render_interval = render_interval
        self.large_steps = large_steps

class CreatePlay(gym.Env):
    def __init__(self, gran_factor=1.0, data_type='video'):
        self.tool_gen = ToolGenerator(gran_factor=gran_factor)
        self.ALL_TOOLS = self.tool_gen.tools

        self.data_type = data_type
        self.large_steps = 30

        self.play_env = PlayEnvWrapper(render_interval=10, large_steps=self.large_steps)
        self.run_steps = 3          # Edit: This was 4 for state

        if self.data_type == 'video':
            self.observation_space = self.play_env.observation_space
        elif self.data_type == 'state':
            self.observation_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.action_space = Discrete(len(self.ALL_TOOLS))
        self.grayscale = True

        self.collision_radius = 0.1 # for no op
        self.mode = 'rgb_array'     # Edit: Only for state

    def update_args(self, args):
        settings = convert_args_to_create(args)
        self.play_env.set_settings(settings)
        self.play_env.large_steps = args.play_large_steps

        self.args = args
        self.tool_gen = ToolGenerator(gran_factor=args.gran_factor)
        self.ALL_TOOLS = self.tool_gen.tools
        self.action_space = Discrete(len(self.ALL_TOOLS))

        self.run_steps = args.create_play_run_steps
        self.grayscale = not args.create_play_colored

        if self.data_type == 'video':
            self.observation_space = Box(low=0.0, high=255.0, shape=(args.image_resolution, args.image_resolution, 1 if self.grayscale else 3), dtype=np.float32)

        if args.render_high_res:
            self.mode += '_high'

    def reset(self):
        # All logic is in the step function. This assumes that reset is always
        # immediately called before step.
        return np.zeros(self.observation_space.shape)

    def step(self, action_id):
        selected_tool = self.ALL_TOOLS[action_id]

        # Place our tool anywhere in the scene
        if self.args.create_play_fixed_tool:
            select_pos = np.array([0.0, 0.0])
        else:
            select_pos = np.random.uniform([-0.7, -0.7], [0.7, 0.7])    # more constrained

        tool = self.tool_gen.get_tool(selected_tool.tool_id, select_pos, self.play_env.settings)

        # This might need to be expanded later
        test_obj_type = ToolTypes.MARKER_BALL

        collision = False
        for trial_num in range(MAX_NUM_TRIALS):
            states = []     # Edit: frames for video

            # Place the test body anywhere in the scene
            if self.args.create_play_fixed_tool:
                test_obj_pos = np.array([[-0.7, -0.7], [-0.7, 0.7], [0.7, -0.7], [0.7, 0.7]])[np.random.randint(4)]
                test_obj_pos += np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
            else:
                test_obj_pos = np.random.uniform([-1.0, -1.0], [1.0, 1.0])

            # Procedurally increase the magnitude of throwing velocity every trial
            to_vec = select_pos - test_obj_pos
            to_mag = np.linalg.norm(to_vec)
            to_vec = to_vec / to_mag
            vel_variance = 0.4
            vel_mean = (1. + (trial_num) * 0.4) * 100 * (1 + to_mag)/ 8
            vel_mag = np.random.uniform((1-vel_variance)*vel_mean,
                (1+vel_variance)*vel_mean)

            # Initialize the test object
            test_obj = self.play_env.tool_factory.create(test_obj_type, test_obj_pos)
            self.play_env.reset([tool, test_obj])

            # Restart if the test object overlaps with the tool at the start
            if tool.shape is not None and \
             len(test_obj.shape.shapes_collide(tool.shape).points) > 0:
                continue

            # Set velocity of test object
            test_obj.body.velocity = vel_mag * to_vec

            # Start simulation
            first_touched = -1
            for i in range(self.run_steps):
                # Simulate one step
                new_state, touched, ft = self.play_env.step_forward_play(
                    test_obj, select_pos, tool,
                    data_type=self.data_type,
                    collision_radius=self.collision_radius,
                    mode='rgb_array_high')
                new_state = np.array(new_state).astype(np.float32)

                if first_touched == -1 and ft != -1:
                    first_touched = len(states) + ft
                if self.grayscale and self.data_type == 'video':
                    grayscale_frames = []
                    for fra in new_state:
                        fr = cv2.cvtColor(fra, cv2.COLOR_RGB2GRAY)
                        grayscale_frames.append(fr)
                    new_state = grayscale_frames

                collision = collision or touched
                states.extend(new_state)

            if collision:   # Implies we got a meaningful trajectory
                break

        if collision:   # Split states around the collision state
            first = max(0, first_touched - (self.args.create_play_len - 1) // 2)
            last = min(len(states), first_touched + (self.args.create_play_len // 2 + 1))
            while ((last - first < self.args.create_play_len) and last < len(states)):
                last += 1
            while ((last - first < self.args.create_play_len) and first > 0):
                first -= 1
        else:
            mid = len(states) // 2
            first = mid - (self.args.create_play_len - 1) // 2
            last = mid + (self.args.create_play_len // 2 + 1)

        states = states[first:last]
        assert len(states) == self.args.create_play_len

        states = np.array(states)
        if self.data_type == 'video':
            states = (states/255.0) - 0.5
            if self.grayscale:
                states = np.expand_dims(states, -1)

        info = {
                'states': states,   # No post-processing will be applied to the states
                'actions': [0]  # Just a garbage value
                }

        return states[-1], 0.0, False, info


    def seed(self, seed_id):
        np.random.seed(seed_id)


    def render_obs(self, obs, action_id):
        if self.data_type == 'video':
            if self.grayscale:
                frames = np.repeat(obs, 3, axis=-1)
            else:
                frames = obs[:]
            frames = (frames + 0.5) * 255.0
        elif self.data_type == 'state':
            selected_tool = self.ALL_TOOLS[action_id]
            tool = self.tool_gen.get_tool(selected_tool.tool_id, obs[0][:2], self.play_env.settings)

            test_obj_type = ToolTypes.MARKER_BALL
            test_obj = self.play_env.tool_factory.create(test_obj_type, obs[0][2:4])
            self.play_env.reset([tool, test_obj])

            frames = []

            for ob in obs:
                ball_pos = ob[2:4]
                ball_vel = ob[4:]

                test_obj.body.position = self.play_env.denormalize(ball_pos)
                test_obj.body.velocity = self.play_env.denormalize(ball_vel)

                frame = self.play_env.render(mode=self.mode)
                frames.append(frame)
            frames = np.array(frames)
        return frames




class StateCreatePlay(CreatePlay):

    def __init__(self):
        super().__init__(data_type='state')

