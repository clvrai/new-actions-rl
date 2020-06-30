import gym
from .constants import *
from .tools.tool_factory import *
import copy
import cv2
from gym.spaces import Box
import numpy as np
import os
import pymunk.pygame_util
import pymunk
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"
from pygame.color import *
from pygame.locals import *
import pygame
pygame.mixer.quit()  # Stop all sounds



class BaseEnv(gym.Env):
    def __init__(self):
        self.tool_factory = ToolFactory()
        self.target_hit = 0
        self.goal_hit = 0

        self.ball_traces = []
        self.line_traces = []

        self.marker_ball_traces = []
        self.target_ball_traces = []

        self.marker_positions = []
        self.target_positions = []

        self.marker_lines = []
        self.target_lines = []

        self.int_frames = None

        self.is_setup = False
        self.viewer = None

    def set_settings(self, settings):
        """
        Configurate the settings of the environment. This also initializes
        all the configurations of the environment so it must be called before
        any other methods of the environment are invoked.
        """
        self.settings = settings
        self.tool_factory.set_settings(settings)

        self.screen = pygame.display.set_mode((self.settings.screen_width,
                                               self.settings.screen_height))
        self.scale = self.settings.high_res_width/self.settings.screen_width

        self.fps = 30.0

        self.max_num_steps = self.settings.max_num_steps
        self.large_steps = self.settings.large_steps

        self.space = pymunk.Space()
        self.space.gravity = self.settings.gravity
        self.space.sleep_time_threshold = 0.3
        self.observation_space = Box(low=0.0, high=255.0,
                                     shape=(self.settings.render_width,
                                            self.settings.render_height, 3),
                                     dtype=np.float32)
        self.is_setup = True

    def _check_setup(self):
        assert self.is_setup, 'Must call `set_settings(...)` first'

    def normalize_action(self, denorm_action):
        """
        Get the position from screen coordinates to normalized coordinates
        [-1, 1] and apply cliping to the resulting normalized position.
        """
        x = ((denorm_action[0] / self.settings.screen_width) * 2) - 1
        y = ((denorm_action[1] / self.settings.screen_height) * 2) - 1
        return np.clip(np.array([x, y]), -1.0, 1.0)

    def normalize_coordinates(self, denorm):
        """
        Get the position from screen coordinates to normalized coordinates
        [-1, 1]. Do not apply any clipping.
        """
        x = ((denorm[0] / self.settings.screen_width) * 2) - 1
        y = ((denorm[1] / self.settings.screen_height) * 2) - 1
        return np.array([x, y])

    def denormalize(self, norm_action):
        """
        Convert a position from [-1, 1] to being on the screen width and height
        """
        x = ((norm_action[0] + 1) * self.settings.screen_width) / 2
        y = ((norm_action[1] + 1) * self.settings.screen_height) / 2
        return np.array([x, y])

    def calc_distance(self, x, y):
        """
        Calculate the normalized distance between two coordinate points.
        - x point defined by 2D array
        - y point defined by 2D array
        """
        use_x = self.normalize_action(x)
        use_y = self.normalize_action(y)
        # normalize both of the coordinates
        dist = np.sqrt(np.sum(np.square(np.array(use_x) - np.array(use_y))))
        # distance between [-1, -1] and [1, 1] is sqrt(8). This is the
        # maximum possible distance between two objects.
        ratio = dist / np.sqrt(8)
        return ratio

    def within_bounds(self, pos):
        """
        Check if a desired place position is within the bounds of the screen
        """
        return (pos[0] < self.settings.screen_width and pos[0] > 0 and
                pos[1] < self.settings.screen_height and pos[1] > 0)

    def reset(self, env_tools):
        """
        Reset the environment
        """
        self._check_setup()
        self.space.remove([*self.space.bodies, *self.space.shapes,
                           *self.space.constraints])

        self.ball_traces = []
        self.line_traces = []

        self.marker_line_traces = []
        self.target_line_traces = []

        self.marker_ball_traces = []
        self.target_ball_traces = []

        self.marker_positions = []
        self.target_positions = []

        self.marker_lines = []
        self.target_lines = []

        # Add all the environments tools start of every episdoe
        for env_tool in env_tools:
            env_tool.add_to_space(self.space)
        self.env_tools = env_tools

    def prepare_traces(self, large_step_i):
        """
        Computing the traces of the ball for a nice rendered representation of
        the path of the ball in a single image.
        """
        if large_step_i % 40 == 0:
            if hasattr(self, 'marker_obj') and (len(self.marker_positions) == 0 or \
                    np.sqrt(np.sum((self.marker_positions[-1] - self.marker_obj.body.position)**2)) > 5.0):

                new_marker_trace = copy.deepcopy(self.marker_obj)
                new_marker_trace.is_trace = True
                new_marker_trace.color = 'royalblue'
                self.marker_positions.append(self.marker_obj.body.position)
                self.marker_ball_traces.append(new_marker_trace)

            if len(self.target_positions) == 0 or \
                    np.sqrt(np.sum((self.target_positions[-1] - self.target_obj.body.position)**2)) > 5.0:

                new_target_trace = copy.deepcopy(self.target_obj)
                new_target_trace.is_trace = True
                new_target_trace.color = 'lightcoral'
                self.target_positions.append(self.target_obj.body.position)
                self.target_ball_traces.append(new_target_trace)

        if large_step_i % 8 == 0:
            if hasattr(self, 'marker_obj'):
                new_marker_trace = copy.deepcopy(self.marker_obj)
                new_marker_trace.is_trace = True
                new_marker_trace.color = 'blue'
                new_marker_trace.segment = True
                self.marker_lines.append(self.marker_obj.body.position)
                if len(self.marker_lines) > 1:
                    new_marker_trace.prev_pos = self.marker_lines[-2]
                    self.marker_line_traces.append(new_marker_trace)

            new_target_trace = copy.deepcopy(self.target_obj)
            new_target_trace.is_trace = True
            new_target_trace.segment = True
            new_target_trace.color = 'firebrick'
            self.target_lines.append(self.target_obj.body.position)
            if len(self.target_lines) > 1:
                new_target_trace.prev_pos = self.target_lines[-2]
                self.target_line_traces.append(new_target_trace)

    def step_forward(self):
        """
        Step forward in the current configuration of the scene.
        Return a sequence of images as output
        """

        self._check_setup()

        int_frames = []
        dt = 1.0/self.fps
        for large_step_i in range(self.large_steps):
            if self.episode_len != 0 and self.settings.render_ball_traces and \
                    self.settings.evaluation_mode:
                self.prepare_traces(large_step_i)

            self.space.step(dt)
            if self.settings.render_mega_res and \
                    self.settings.evaluation_mode and \
                    large_step_i % self.settings.mega_res_interval == 0:
                int_frame = self.render('rgb_array_high_mega_res')
                int_frames.append(int_frame)

        frame = self.render()
        if len(int_frames) > 0:
            self.int_frames = int_frames
        return frame


    def step_forward_play(self, test_obj, tool_pos, tool, data_type='video', collision_radius=0.1, mode='rgb_array_high', add_text=None):
        """
        Place just two objects in the scene. The test_obj is used to probe the
        properties of the tool at tool_pos. This is supposed to be used to
        investigate the properties of the tools.
        """
        self._check_setup()
        states = []
        dt = 1.0/self.fps
        touched = False
        first_touched = -1
        for large_step_i in range(self.large_steps):
            if self.settings.render_ball_traces and data_type=='video':
                self.prepare_traces(large_step_i)

            self.space.step(dt)

            if tool.shape is not None and \
                len(test_obj.shape.shapes_collide(tool.shape).points) > 0:
                touched = True
            elif tool.shape is None: # This is for No Op Only
                touched = self.calc_distance(
                    test_obj.body.position, self.denormalize(tool_pos)) < collision_radius

            # Accumulate frames / states to trajectory
            if ((large_step_i+1) % self.render_interval == 0) or \
                    (large_step_i == (self.large_steps - 1) and states == []):
                if first_touched == -1 and touched:
                    first_touched = len(states)

                if data_type == 'video':
                    state = self.render(mode=mode, add_text=add_text)
                elif data_type == 'state':
                    ball_pos = self.normalize_coordinates(test_obj.body.position)
                    ball_vel = self.normalize_coordinates(test_obj.body.velocity)
                    state = np.concatenate([tool_pos, ball_pos, ball_vel], axis=-1)
                states.append(state)

        return states, touched, first_touched

    def get_all_objs(self):
        """
        Returns all objects that should be rendered
        """
        return self.env_tools

    def render(self, mode='rgb_array', add_text=None):
        """
        Render all currently placed objects to the scene.
        """
        prev_mode = '%s' % mode
        if prev_mode == 'human':
            mode='rgb_array_high'

        self._check_setup()

        if self.int_frames is not None and self.settings.evaluation_mode and 'mega' in mode:
            tmp = self.int_frames
            self.int_frames = None
            return tmp

        anti_alias = 'mega' in mode

        # Clear the screen
        # Note 'low' resolution also uses 'high' in mode
        if 'high' in mode:
            self.screen = pygame.display.set_mode(
                (self.settings.high_res_width, self.settings.high_res_height))
        self.screen.fill(THECOLORS["white"])

        render_objs = self.get_all_objs()

        if 'ball_trace' in mode:
            # Marker Ball Traces (Motion Blur)
            for i, render_obj in enumerate(self.marker_ball_traces):
                render_obj.set_alpha = 70 + \
                    (i+1) * (150 - 70)/len(self.marker_ball_traces)
                if 'high' in mode:
                    render_obj.render(self.screen, scale=self.scale,
                            anti_alias=anti_alias)
                else:
                    render_obj.render(self.screen)

            # Target Ball Traces (Motion Blur)
            for i, render_obj in enumerate(self.target_ball_traces):
                render_obj.set_alpha = 120 + \
                    (i+1) * (220 - 120)/len(self.target_ball_traces)
                if 'high' in mode:
                    render_obj.render(self.screen, scale=self.scale,
                            anti_alias=anti_alias)
                else:
                    render_obj.render(self.screen)

            # For all tools/balls in the environment
            for render_obj in render_objs:
                if 'high' in mode:
                    render_obj.render(self.screen, self.scale,
                            anti_alias=anti_alias)
                else:
                    render_obj.render(self.screen)

            # Marker Ball Line Traces
            for i, render_obj in enumerate(self.marker_line_traces):
                if 'high' in mode:
                    render_obj.render(self.screen, scale=self.scale,
                            anti_alias=anti_alias)
                else:
                    render_obj.render(self.screen)

            # Target Ball Line Traces
            for i, render_obj in enumerate(self.target_line_traces):
                if 'high' in mode:
                    render_obj.render(self.screen, scale=self.scale,
                            anti_alias=anti_alias)
                else:
                    render_obj.render(self.screen)

        else:
            for render_obj in render_objs:
                if 'high' in mode:
                    render_obj.render(self.screen, self.scale,
                            anti_alias=anti_alias)
                else:
                    render_obj.render(self.screen)

        frame = pygame.surfarray.array3d(self.screen)

        frame = np.fliplr(frame.swapaxes(0, 1))

        if 'text' in mode:
            text = '{} {:.3f}'.format(self.episode_len, self.episode_reward)
            if mode == 'rgb_array_text':
                x = cv2.putText(frame, text, (1, 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.25, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                x = cv2.putText(frame, text,
                                (int(self.scale * 1), int(self.scale * 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.scale * 0.25,
                                (0, 0, 0), 1, cv2.LINE_AA)
            frame = x.get()

        if add_text is not None:
            x = cv2.putText(frame, add_text,
                            (int(self.scale * 16), int(self.scale * 20)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            self.scale * 0.15,
                            (0, 0, 0), 1, cv2.LINE_AA)
            frame = x.get()

        if 'changed_colors' in mode:
            if self.goal_hit:
                frame = self._convert_color(
                    frame, [255, 255, 255], [200, 255, 255])

        if 'high' in mode:
            self.screen = pygame.display.set_mode(
                (self.settings.screen_width, self.settings.screen_height))

        if prev_mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(frame)
            return self.viewer.isopen
        else:
            return frame

    def _convert_color(self, image, from_color, to_color):
        image = image.copy()
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        if from_color is None:
            from_r, from_g, from_b = \
                image[:, :, 0].max(), image[:, :, 1].max(), image[:, :, 2].max()
            mask = (r == from_r) & (g == from_g) & (b == from_b)
        else:
            mask = (r == from_color[0]) & (
                g == from_color[1]) & (b == from_color[2])
        image[:, :, :3][mask] = to_color
        return image
