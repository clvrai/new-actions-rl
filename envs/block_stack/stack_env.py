import os.path as osp
import numpy as np
from envs.block_stack.stack_xml import XML
import mujoco_py as mjc
from envs.block_stack.poly_gen import gen_polys

import gym.spaces as spaces
from envs.action_env import ActionEnv
import cv2


DROP_Y = 0.0
DROP_Z = 1.0
SMALL_RENDER_DIM = 84
DEF_LARGE_RENDER_DIM = 512

DOUBLE_PLACE_PEN = -0.25
# Don't change this, change `args.stack_reward` relative to this.
HEIGHT_SPARSE_REWARD = 1.0

EPISODE_STEPS = 10
DEFAULT_EXTRA_RATIO = 2

PLACE_RANGE = 1.5

class StackEnv(ActionEnv):
    def __init__(self, placement2d=False, render_freq=25, min_steps=10,
                 max_steps=100, asset_path='assets/stl/',
                 episode_steps=EPISODE_STEPS, default_extra_ratio=DEFAULT_EXTRA_RATIO):
        super().__init__()
        self.render_freq = render_freq
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.max_episode_steps = episode_steps
        self.default_extra = max_steps * default_extra_ratio
        # Set to 0 in reset
        self.cur_episode_step = None
        self.num_penalty = 0
        self.prev_height = 0.0

        if not placement2d:
            self.take_action_dims = 2
        else:
            self.take_action_dims = 1
        self.placement2d = placement2d
        self.debug_log = False
        self.is_fig_mode = False

        self.asset_path = osp.join(osp.dirname(
            osp.realpath(__file__)), asset_path)

        self.all_polygons, self.shape_names, self.polygon_types = gen_polys(self.asset_path)
        self._log('with %i polygons' % len(self.all_polygons))

        self.placed_objs = []
        self.sim = None
        self.viewer = None
        self._init_scene(None)

        self.high_render = False
        self.high_render_dim = DEF_LARGE_RENDER_DIM
        self.observation_space = spaces.Box(low=0.0, high=255.0,
                                            shape=(SMALL_RENDER_DIM, SMALL_RENDER_DIM, 3), dtype=np.float32)

    def get_env_name(self):
        return 'stack'

    def _set_action_space(self, sub_split):
        super()._set_action_space(sub_split)
        if self.is_fig_mode:
            self.aval_idx = list(range(len(self.all_polygons)))

        self.action_space = spaces.Dict({
            # select any one of our tools
            'index': spaces.Discrete(len(self.aval_idx)),
            # x, y between -1.0 and 1.0
            'pos': spaces.Box(low=-1.0, high=1.0, shape=(self.take_action_dims,), dtype=np.float32)
        })

    def set_args(self, args, set_eval):
        self.high_render = set_eval
        self.high_render_dim = args.high_render_dim
        self.render_freq = args.high_render_freq
        self.stack_reward = args.stack_reward
        self.sparse_reward_height = args.stack_height
        super().set_args(args, set_eval)

    def seed(self, seed_val):
        np.random.seed(seed_val)

    def _init_scene(self, drop_pos):
        save_dat = {}
        if self.sim is not None:
            for n in self.xml_names:
                joint_ind = self.sim.model._joint_name2id[n]
                vel = self.sim.data.qvel[joint_ind * 6:(joint_ind+1) * 6]
                pos = self.sim.data.qpos[joint_ind * 7:(joint_ind+1) * 7]
                #pos[2] += (pos[2] * 0.1)
                # Zero out the velocity. This could help prevent NaN simulation
                # issues?
                vel[:] = 0
                save_dat[n] = (vel, pos)

        self._log('loading scene')
        xml = XML(self.asset_path, self.shape_names, place2d=self.placement2d)

        for poly in self.placed_objs:
            xml.add_mesh(poly.ply, pos=poly.pos, axangle=poly.angle,
                         scale=poly.scale, rgba=poly.rgba)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        self.sim = mjc.MjSim(model)

        self.xml_names = xml.names
        self._log('xml names ' + str(self.xml_names))

        qvel = self.sim.data.qvel
        qpos = self.sim.data.qpos
        for n, (vel, pos) in save_dat.items():
            self._log('restoring pos of ' + n)
            joint_ind = self.sim.model._joint_name2id[n]
            qvel[joint_ind * 6:(joint_ind+1) * 6] = vel
            qpos[joint_ind * 7:(joint_ind+1) * 7] = pos

        if len(self.xml_names) > 0:
            just_dropped = self.xml_names[-1]
            self._log('dropping ' + just_dropped)
            joint_ind = self.sim.model._joint_name2id[just_dropped]
            qvel[joint_ind * 6:(joint_ind+1) * 6] = 0
            qpos[joint_ind * 7:(joint_ind+1) * 7] = [*drop_pos, 0, 0, 0, 0]

        if qpos is not None:
            self.sim.data.qvel[:] = qvel
            self.sim.data.qpos[:] = qpos

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mjc.MjRenderContextOffscreen(self.sim, -1)
        else:
            self.viewer.update_sim(self.sim)
        return self.viewer

    def render_img(self, render_high, camera='fixed'):
        if render_high:
            render_dim = self.high_render_dim
        else:
            render_dim = SMALL_RENDER_DIM

        camera_id = self.sim.model.camera_name2id(camera)

        viewer = self._get_viewer()
        viewer.render(render_dim, render_dim, camera_id=camera_id)
        image = viewer.read_pixels(render_dim, render_dim, depth=False)
        return image

    def _get_rbound_top(self, place_obj_idx):
        # Get highest rounding box top z-coordinate of all placed objects

        # THRESHOLD = 0.8
        # SCALE_FACTOR = 1.0
        # THRESH = 1.0

        # breakpoint()

        # If no objects has been placed
        if self.sim.data.qpos is None or len(self.placed_objs) == 0:
            return 0.0
        # num_objs = len(self.sim.data.qpos) // 7
        # max_z = None
        # placed_poly = self.all_polygons[place_obj_idx]

        ymax_list = []
        for obj_name in self.xml_names:
            idx = self.sim.model.geom_name2id(obj_name)
            rbound = self.sim.model.geom_rbound[idx]
            ypos = self.sim.data.geom_xpos[idx, 2]
            ymax = ypos + rbound
            ymax_list.append(ymax)
        return max(ymax_list)


    def step(self, a):
        SAFE_DIST = 0

        drop_obj_idx = int(a[0])
        real_idx = self.aval_idx[drop_obj_idx]

        # x, y coordinates given by agent
        drop_pos = np.array(a[1:])
        if self.is_fig_mode:
            drop_pos *= PLACE_RANGE

        # Get highest rounding box top z-coordinate of all placed objects
        rb_top_z = self._get_rbound_top(real_idx)

        poly = self.all_polygons[real_idx]

        reward = 0
        done = False
        stacked = False

        # Allow to place the same object twice if rendering figures
        if poly in self.placed_objs and not self.is_fig_mode:
            # This polygon has already been placed.
            reward = DOUBLE_PLACE_PEN
            _, obs = self._settle_sim(render=self.high_render)
            offset = 0.5
            highest = self._get_highest_obj() + offset
            highest = max(highest, 0)
            self.num_penalty += 1
        else:
            # Place the new object below the table first
            init_drop_z = -1.0
            if self.placement2d:
                assert len(a) == 2
                drop_obj_pos = [*drop_pos, DROP_Y, init_drop_z]
            else:
                assert len(a) == 3
                drop_obj_pos = [*drop_pos, init_drop_z]

            # Place the new polygon
            self.placed_objs.append(poly)
            self._init_scene(drop_obj_pos)

            # Get rounding box of newly placed object and change its z-coordinates
            just_dropped = self.xml_names[-1]
            idx = self.sim.model.geom_name2id(just_dropped)
            rb_new = self.sim.model.geom_rbound[idx]

            # correct z-pos in data.qpos. Subtract a safe subtract_dist from the bounding radius
            # for this specific type of object so the distance is minimized
            joint_ind = self.sim.model._joint_name2id[just_dropped]
            rb_drop_z = rb_top_z + SAFE_DIST + rb_new
            correct_drop_z = rb_drop_z - poly.subtract_dist
            self.sim.data.qpos[joint_ind * 7 + 2] = correct_drop_z

            extra = 0
            if self.is_fig_mode and self.cur_episode_step == self.max_episode_steps - 1:
                print('extra!')
                extra = self.default_extra

            _, obs = self._settle_sim(render=self.high_render, extra=extra)

            offset = 0.5
            highest = self._get_highest_obj() + offset
            highest = max(highest, 0)

            height_diff = highest - self.prev_height

            if not self.args.sparse_height:
                reward += self.stack_reward * height_diff
            self.prev_height = highest

            if highest > self.sparse_reward_height:
                stacked = True
                done = True
                if self.args.sparse_height:
                    reward += HEIGHT_SPARSE_REWARD

            if highest > self.max_height:
                self.max_height = highest

            if not self.args.render_result_figures:
                self._add_info_to_render(highest, stacked)

        info = {
            'aval': self.aval_idx
        }

        self.cur_episode_step += 1
        if self.cur_episode_step >= self.max_episode_steps:
            done = True

        if self.args.only_sparse_reward:
            if done:
                reward = self.max_height
            else:
                reward = 0.

        if done:
            info['ep_final_height'] = highest
            info['ep_repeat'] = self.num_penalty

        return obs, reward, done, info

    def _add_info_to_render(self, height, is_success):
        height_txt = '%.2f' % height
        scale = self.high_render_dim // SMALL_RENDER_DIM

        def add_txt_fn(frame):
            frame = cv2.putText(frame, height_txt,
                    (int(scale * 1), int(scale * 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.25,
                    (255, 255, 255), 1, cv2.LINE_AA)
            if is_success:
                frame = cv2.putText(frame, 'Yes',
                    (int(scale * 1), int(scale * 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.25,
                    (46, 204, 113), 1, cv2.LINE_AA)
            return frame

        self.cur_render = [add_txt_fn(frame) for frame in self.cur_render]

    def _get_highest_obj(self):
        # Get every 3rd element (Z coord)
        num_objs = len(self.sim.model._joint_name2id)
        z_coords = [self.sim.data.qpos[2 + i * 7] for i in range(num_objs)]
        return max(z_coords)

    def _settle_sim(self, vel_threshold=0.1, render=False, extra=0):
        step = 0
        imgs = []
        for _ in range(self.min_steps + extra):
            self.sim.step()
            if render and step % self.render_freq == 0:
                imgs.append(self.render_img(True))
            step += 1

        max_vel = np.abs(self.sim.data.qvel).max()
        while max_vel > vel_threshold:
            self.sim.step()
            if render and step % self.render_freq == 0:
                imgs.append(self.render_img(True))

            max_vel = np.abs(self.sim.data.qvel[:, ]).max()
            step += 1
            if step > self.max_steps:
                break

        self.cur_render = imgs

        ret_img = self.render_img(False)
        if len(self.cur_render) == 0:
            self.cur_render.append(ret_img)

        return step, ret_img

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    def render(self, mode='rgb_array'):
        rndr_arr = np.array(self.cur_render)
        return rndr_arr

    def reset(self):
        super().reset()

        self.cur_episode_step = 0
        self.num_penalty = 0
        self.prev_height = 0.0
        self.placed_objs = []
        self.xml_names = []
        self.max_height = 0.
        img = self.render_img(False)
        return img



class StackCompressedEnv(StackEnv):
    def __init__(self, placement2d=False, render_freq=25, min_steps=10,
                 max_steps=50, asset_path='assets/stl/',
                 episode_steps=EPISODE_STEPS, default_extra_ratio=DEFAULT_EXTRA_RATIO):
        super().__init__(placement2d, render_freq, min_steps, max_steps, asset_path,
                 episode_steps, default_extra_ratio)



class StackLoadEnv(StackEnv):
    def __init__(self, placement2d=False, render_freq=25, min_steps=10,
                 max_steps=50, asset_path='assets/stl/',
                 episode_steps=EPISODE_STEPS, default_extra_ratio=DEFAULT_EXTRA_RATIO):
        super().__init__(placement2d, render_freq, min_steps, max_steps, asset_path,
                 episode_steps, default_extra_ratio)

    def reset(self):
        super(StackEnv, self).reset()
        self.cur_episode_step = 0
        self.num_penalty = 0
        self.prev_height = 0.0
        self.placed_objs = []
        self.max_height = 0.
        self.xml_names = []
        self.add_all_aval_objs()
        img = self.render_img(False)
        return img


    def add_all_aval_objs(self):
        polygons = np.array(self.all_polygons)[self.aval_idx]

        self._log('loading scene')

        # EDIT : Also try loading ALL objects in __init__, and then just dealing with repositioning of objects
        self.xml = XML(self.asset_path, self.shape_names, place2d=self.placement2d, contacts_off=True)

        add_pos = [-100.0, -100.0]
        FREEZE_Z = -10.0
        for j, poly in enumerate(polygons):
            # self.xml.add_mesh(poly.ply, pos=poly.pos, axangle=poly.angle,
            #              scale=poly.scale, rgba=poly.rgba)
            self.xml.add_mesh(poly.ply, pos=[*add_pos, FREEZE_Z * (j + 1)], axangle=poly.angle,
                         scale=poly.scale, rgba=poly.rgba)
            if add_pos[0] < 100.0:
                add_pos[0] += 1.0
            else:
                add_pos[0] = -100.0
                add_pos[1] += 1.0

        xml_str = self.xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)

        self.sim = mjc.MjSim(model)

        self.xml_names = self.xml.names

        for j in range(len(self.xml_names)):
            self.sim.model.geom_contype[4 + j] = 0
            self.sim.model.geom_conaffinity[4 + j] = 0
        self._log('xml names ' + str(self.xml_names))


    def step(self, a):
        SAFE_DIST = 0

        drop_obj_idx = int(a[0])
        real_idx = self.aval_idx[drop_obj_idx]

        # x, y coordinates given by agent
        drop_pos = np.array(a[1:])
        if self.is_fig_mode:
            drop_pos *= PLACE_RANGE

        # Get highest rounding box top z-coordinate of all placed objects
        rb_top_z = self._get_rbound_top(real_idx)

        poly = self.all_polygons[real_idx]

        reward = 0
        done = False
        stacked = False

        # Allow to place the same object twice if rendering figures
        if poly in self.placed_objs and not self.is_fig_mode:
            # This polygon has already been placed.
            reward = DOUBLE_PLACE_PEN
            _, obs = self._settle_sim(render=self.high_render)
            offset = 0.5
            highest = self._get_highest_obj() + offset
            highest = max(highest, 0)
            self.num_penalty += 1
        else:
            '''
                Edit here: place all the objects in the scene to begin with
                Try if it actually makes things slower.
                If not, then no need to re-load xml file for sure
            '''

            # Place the new object below the table first
            init_drop_z = -1.0
            if self.placement2d:
                assert len(a) == 2
                drop_obj_pos = [*drop_pos, DROP_Y, init_drop_z]
            else:
                assert len(a) == 3
                drop_obj_pos = [*drop_pos, init_drop_z]

            # Place the new polygon
            self.placed_objs.append(poly)

            self.xml_names = self.xml.names
            drop_obj_name = self.xml_names[drop_obj_idx]

            self._drop_object(drop_obj_name, drop_obj_pos)

            # Get rounding box of newly placed object and change its z-coordinates
            idx = self.sim.model.geom_name2id(drop_obj_name)
            rb_new = self.sim.model.geom_rbound[idx]

            # correct z-pos in data.qpos. Subtract a safe subtract_dist from the bounding radius
            # for this specific type of object so the distance is minimized
            joint_ind = self.sim.model._joint_name2id[drop_obj_name]
            rb_drop_z = rb_top_z + SAFE_DIST + rb_new
            correct_drop_z = rb_drop_z - poly.subtract_dist
            self.sim.data.qpos[joint_ind * 7 + 2] = correct_drop_z

            self.sim.model.geom_contype[4 + drop_obj_idx] = 1
            self.sim.model.geom_conaffinity[4 + drop_obj_idx] = 1


            extra = 0
            if self.is_fig_mode and self.cur_episode_step == self.max_episode_steps - 1:
                print('extra!')
                extra = self.default_extra

            _, obs = self._settle_sim(render=self.high_render, extra=extra)

            offset = 0.5
            highest = self._get_highest_obj() + offset
            highest = max(highest, 0)

            height_diff = highest - self.prev_height

            if not self.args.sparse_height:
                reward += self.stack_reward * height_diff
            self.prev_height = highest

            if highest > self.sparse_reward_height:
                stacked = True
                done = True
                if self.args.sparse_height:
                    reward += HEIGHT_SPARSE_REWARD

            if highest > self.max_height:
                self.max_height = highest

            if not self.args.render_result_figures:
                self._add_info_to_render(highest, stacked)

        info = {
            'aval': self.aval_idx
        }

        self.cur_episode_step += 1
        if self.cur_episode_step >= self.max_episode_steps:
            done = True

        if self.args.only_sparse_reward:
            if done:
                reward = self.max_height
            else:
                reward = 0.

        if done:
            info['ep_final_height'] = highest
            info['ep_repeat'] = self.num_penalty

        return obs, reward, done, info


    def _drop_object(self, drop_obj_name, drop_pos):
        qvel = self.sim.data.qvel
        qpos = self.sim.data.qpos

        self._log('dropping ' + drop_obj_name)
        joint_ind = self.sim.model._joint_name2id[drop_obj_name]
        qvel[joint_ind * 6:(joint_ind+1) * 6] = 0
        qpos[joint_ind * 7:(joint_ind+1) * 7] = [*drop_pos, 0, 0, 0, 0]

        if qpos is not None:
            self.sim.data.qvel[:] = qvel
            self.sim.data.qpos[:] = qpos






class StackStartEnv(StackEnv):
    def __init__(self, placement2d=False, render_freq=25, min_steps=10,
                 max_steps=100, asset_path='assets/stl/',
                 episode_steps=EPISODE_STEPS, default_extra_ratio=DEFAULT_EXTRA_RATIO):
        super().__init__(placement2d, render_freq, min_steps, max_steps, asset_path,
                 episode_steps, default_extra_ratio)
        self.add_all_objs()

    def reset(self):
        super(StackEnv, self).reset()
        self.cur_episode_step = 0
        self.num_penalty = 0
        self.prev_height = 0.0
        self.placed_objs = []
        self.max_height = 0.
        self.xml_names = []
        self.reset_obj_pos()
        img = self.render_img(False)
        return img


    def add_all_objs(self):
        self.xml = XML(self.asset_path, self.shape_names, place2d=self.placement2d, contacts_off=True)
        self.default_positions = []
        add_pos = [-100.0, -100.0]
        FREEZE_Z = -10.0
        for j, poly in enumerate(self.all_polygons):
            self.default_positions.append([*add_pos, FREEZE_Z * (j + 1)])
            self.xml.add_mesh(poly.ply, pos=self.default_positions[j], axangle=poly.angle,
                         scale=poly.scale, rgba=poly.rgba)
            if add_pos[0] < 100.0:
                add_pos[0] += 1.0
            else:
                add_pos[0] = -100.0
                add_pos[1] += 1.0

        xml_str = self.xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        self.sim = mjc.MjSim(model)

        self.xml_names = self.xml.names
        self._log('xml names ' + str(self.xml_names))


    def reset_obj_pos(self):
        for j in range(len(self.all_polygons)):
            self.sim.model.geom_contype[4 + j] = 0
            self.sim.model.geom_conaffinity[4 + j] = 0
            self.sim.model.geom_condim[4 + j] = 3
            self.sim.data.qvel[j * 6:(j+1) * 6] = 0
            self.sim.data.qpos[j * 7:(j+1) * 7] = [*self.default_positions[j], 0, 0, 0, 0]

    def step(self, a):
        SAFE_DIST = 0

        drop_obj_idx = int(a[0])
        real_idx = self.aval_idx[drop_obj_idx]

        # x, y coordinates given by agent
        drop_pos = np.array(a[1:])
        if self.is_fig_mode:
            drop_pos *= PLACE_RANGE

        # Get highest rounding box top z-coordinate of all placed objects
        rb_top_z = self._get_rbound_top(real_idx)

        poly = self.all_polygons[real_idx]

        reward = 0
        done = False
        stacked = False

        # Allow to place the same object twice if rendering figures
        if poly in self.placed_objs and not self.is_fig_mode:
            # This polygon has already been placed.
            reward = DOUBLE_PLACE_PEN
            _, obs = self._settle_sim(render=self.high_render)
            offset = 0.5
            highest = self._get_highest_obj() + offset
            highest = max(highest, 0)
            self.num_penalty += 1
        else:
            '''
                Edit here: place all the objects in the scene to begin with
                Try if it actually makes things slower.
                If not, then no need to re-load xml file for sure
            '''

            # Place the new object below the table first
            init_drop_z = -1.0
            if self.placement2d:
                assert len(a) == 2
                drop_obj_pos = [*drop_pos, DROP_Y, init_drop_z]
            else:
                assert len(a) == 3
                drop_obj_pos = [*drop_pos, init_drop_z]

            # Place the new polygon
            self.placed_objs.append(poly)

            self.xml_names = self.xml.names
            drop_obj_name = self.xml_names[real_idx]

            self._drop_object(drop_obj_name, drop_obj_pos)

            # Get rounding box of newly placed object and change its z-coordinates
            idx = self.sim.model.geom_name2id(drop_obj_name)
            rb_new = self.sim.model.geom_rbound[idx]

            # correct z-pos in data.qpos. Subtract a safe subtract_dist from the bounding radius
            # for this specific type of object so the distance is minimized
            joint_ind = self.sim.model._joint_name2id[drop_obj_name]
            rb_drop_z = rb_top_z + SAFE_DIST + rb_new
            correct_drop_z = rb_drop_z - poly.subtract_dist
            self.sim.data.qpos[joint_ind * 7 + 2] = correct_drop_z

            self.sim.model.geom_contype[4 + real_idx] = 1
            self.sim.model.geom_conaffinity[4 + real_idx] = 1
            self.sim.model.geom_condim[4 + real_idx] = 6

            extra = 0
            if self.is_fig_mode and self.cur_episode_step == self.max_episode_steps - 1:
                print('extra!')
                extra = self.default_extra

            _, obs = self._settle_sim(render=self.high_render, extra=extra)

            offset = 0.5
            highest = self._get_highest_obj() + offset
            highest = max(highest, 0)

            height_diff = highest - self.prev_height

            if not self.args.sparse_height:
                reward += self.stack_reward * height_diff
            self.prev_height = highest

            if highest > self.sparse_reward_height:
                stacked = True
                done = True
                if self.args.sparse_height:
                    reward += HEIGHT_SPARSE_REWARD

            if highest > self.max_height:
                self.max_height = highest

            if not self.args.render_result_figures:
                self._add_info_to_render(highest, stacked)

        info = {
            'aval': self.aval_idx
        }

        self.cur_episode_step += 1
        if self.cur_episode_step >= self.max_episode_steps:
            done = True

        if self.args.only_sparse_reward:
            if done:
                reward = self.max_height
            else:
                reward = 0.

        if done:
            info['ep_final_height'] = highest
            info['ep_repeat'] = self.num_penalty

        return obs, reward, done, info


    def _drop_object(self, drop_obj_name, drop_pos):
        qvel = self.sim.data.qvel
        qpos = self.sim.data.qpos

        self._log('dropping ' + drop_obj_name)
        joint_ind = self.sim.model._joint_name2id[drop_obj_name]
        qvel[joint_ind * 6:(joint_ind+1) * 6] = 0
        qpos[joint_ind * 7:(joint_ind+1) * 7] = [*drop_pos, 0, 0, 0, 0]

        if qpos is not None:
            self.sim.data.qvel[:] = qvel
            self.sim.data.qpos[:] = qpos
