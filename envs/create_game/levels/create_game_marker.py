import numpy as np
from .create_level_file import CreateLevelFile

def ball_begin_handler(arbiter, space, data):
    obj_1 = arbiter.shapes[1]
    obj_2 = arbiter.shapes[0]

    if hasattr(obj_1, 'is_marker') and hasattr(obj_2, 'is_target'):
        obj_1.hit_target = True
    if hasattr(obj_2, 'is_marker') and hasattr(obj_1, 'is_target'):
        obj_2.hit_target = True
    return True


class CreateGameMarker(CreateLevelFile):
    """
    Defines additional behavior in logic game for when there is another ball
    that must first collide with the target ball. We call this the "Marker"
    ball. Inherent from this class to provide additional
    """
    def __init__(self, available_tools=None, gravity=(0.0, -2.0),
            tool_variety=True, tool_gen=None):
        super().__init__()

        self.hit_target_handler = None
        self.marker_must_hit = False
        self.target_reward = 0.0

    def set_settings(self, settings):
        super().set_settings(settings)
        self.target_reward = settings.target_reward

    def reset(self):
        obs = super().reset()
        self.marker_obj = self.env_tools[0]

        self.marker_obj.shape.is_marker = True
        self.target_obj.shape.is_target = True

        if self.hit_target_handler is None:
            self.hit_target_handler = self.space.add_collision_handler(self.marker_obj.shape.collision_type,
                    self.target_obj.shape.collision_type)
            self.hit_target_handler.begin = ball_begin_handler
        self.prev_dist = self.calc_distance(self.target_obj.body.position, self.marker_obj.body.position)
        self.target_hit = 0.0
        self.marker_collided = False
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        general_reward = reward

        # Dense reward based off of distance from target ball to the goal
        cur_target_pos = self.target_obj.body.position
        move_dist = self.calc_distance(self.target_obj_start_pos, cur_target_pos)

        if self.target_hit == 0 and move_dist > self.settings.move_thresh and \
                (not self.marker_must_hit or hasattr(self.marker_obj.shape, 'hit_target')):
            if self.settings.marker_reward == 'reg':
                self.target_hit += 1.
                reward += self.target_reward
            elif self.settings.marker_reward == 'dir':
                goal_on_left = self.target_obj_start_pos[0] < self.goal_pos[0]
                moved_target_left = self.target_obj_start_pos[0] < cur_target_pos[0]
                if goal_on_left == moved_target_left:
                    self.target_hit += 1.0
                    reward += self.target_reward
            else:
                raise ValueError('Unknown marker reward type')
            self.prev_dist = self.calc_distance(cur_target_pos, self.goal_pos)
        else:
            distance = self.calc_distance(cur_target_pos,
                    self.marker_obj.body.position)
            reward += self.dense_reward_scale * (self.prev_dist - distance)
            self.episode_dense_reward += self.dense_reward_scale * (self.prev_dist - distance)
            self.prev_dist = distance


        # Terminate if the marker ball is out of bounds AND target is not hit yet
        if (not self.within_bounds(self.marker_obj.body.position)) and self.target_hit == 0:
            done = True
            reward += self.settings.marker_gone_reward

        self.episode_reward += (reward - general_reward)

        if done:
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

        return obs, reward, done, info

