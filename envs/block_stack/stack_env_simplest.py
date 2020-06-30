import os.path as osp
import numpy as np
from envs.block_stack.stack_xml import XML
import mujoco_py as mjc
from envs.block_stack.poly_gen import gen_polys

import gym.spaces as spaces
from envs.action_env import ActionEnv
import cv2


DROP_Z = 1.0
SMALL_RENDER_DIM = 84
DEF_LARGE_RENDER_DIM = 512
VEL_THRESHOLD = 0.05

EPISODE_STEPS = 10



class StackEnvSimplest(ActionEnv):
    def __init__(self, render_freq=25,
                 max_steps=100, asset_path='assets/stl/',
                 episode_steps=EPISODE_STEPS):
        super().__init__()
        self.render_freq = render_freq
        self.max_steps = max_steps

        self.max_episode_steps = episode_steps
        # Set to 0 in reset
        self.cur_episode_step = None
        self.num_penalty = 0
        self.noop = 0
        self.prev_height = 0.0
        # The prev height to keep track of for the render that is not set to 0
        # with the episode reset.
        self.render_height = 0.0

        self.debug_log = False

        self.asset_path = osp.join(osp.dirname(
            osp.realpath(__file__)), asset_path)

        self.all_polygons, self.shape_names, self.polygon_types = gen_polys(self.asset_path)
        self._log('with %i polygons' % len(self.all_polygons))

        self.placed_objs = []
        self.sim = None
        self.viewer = None
        self.usable_objs = []

        self.high_render = False
        self.high_render_dim = DEF_LARGE_RENDER_DIM
        self.observation_space = spaces.Box(low=0.0, high=255.0,
                                            shape=(SMALL_RENDER_DIM, SMALL_RENDER_DIM, 3), dtype=np.float32)

    def get_env_name(self):
        return 'stack'

    def _set_action_space(self, sub_split):
        super()._set_action_space(sub_split)

        # select any one of our tools
        space_dict = {'index': spaces.Discrete(len(self.aval_idx))}

        if self.args.stack_dim > 0:
            # placement between -1.0 and 1.0
            space_dict['pos'] = spaces.Box(low=-1.0, high=1.0, shape=(self.args.stack_dim,), dtype=np.float32)

        if self.args.separate_skip:
            space_dict['skip'] = spaces.Discrete(2)

        if len(space_dict) == 1:
            self.action_space = space_dict['index']
        else:
            self.action_space = spaces.Dict(space_dict)

    def set_args(self, args, set_eval):
        self.high_render = set_eval
        self.high_render_dim = args.high_render_dim
        self.render_freq = args.high_render_freq
        self.contacts_off = args.contacts_off
        super().set_args(args, set_eval)

    def seed(self, seed_val):
        np.random.seed(seed_val)

    def _drop(self, drop_pos, drop_idx):
        just_dropped_name = self.xml_names[drop_idx]
        self.placed_objs.append(just_dropped_name)
        qvel = self.sim.data.qvel
        qpos = self.sim.data.qpos
        for n in self.xml_names:
            joint_ind = self.sim.model._joint_name2id[n]
            vel = self.sim.data.qvel[joint_ind * 6:(joint_ind+1) * 6]
            pos = self.sim.data.qpos[joint_ind * 7:(joint_ind+1) * 7]
            # Zero out the velocity. This could help prevent NaN simulation
            # issues?
            qvel[joint_ind * 6:(joint_ind+1) * 6] = 0

        self._log('Dropping object with idx %s to position %s' %
                  (str(drop_idx), str(drop_pos)))
        just_dropped = self.xml_names[drop_idx]
        joint_ind = self.sim.model._joint_name2id[just_dropped]
        qvel[joint_ind * 6:(joint_ind+1) * 6] = 0
        qpos[joint_ind * 7:(joint_ind+1) * 7] = [*drop_pos, 0, 0, 0, 0]

        if self.contacts_off:
            self.sim.model.geom_contype[4 + joint_ind] = 1
            self.sim.model.geom_conaffinity[4 + joint_ind] = 1
            assert (self.sim.model.geom_condim[4+joint_ind] == 6)
            # self.sim.model.geom_condim[4 + joint_ind] = 6

        self.sim.data.qvel[:] = qvel
        self.sim.data.qpos[:] = qpos

    def get_place_polys(self):
        return []

    def _init_scene(self):
        self._log('loading scene')
        xml = XML(self.asset_path, self.shape_names,
                place2d=self.args.stack_dim < 2,
                contacts_off=self.contacts_off)

        for i, poly in enumerate(self.usable_objs):
            pos = [poly.pos[0], poly.pos[1], (i+1) * -3.0]
            self.start_pos[i] = pos
            xml.add_mesh(poly.ply, pos=pos, axangle=poly.angle,
                         scale=poly.scale, rgba=poly.rgba)

        self.scene_objs = []
        i_offset = len(self.usable_objs)
        scene_polys = []

        place_polys = self.get_place_polys()

        for i in range(len(place_polys)):
            rnd_poly = self.all_polygons[place_polys[i][0]]
            scene_polys.append(rnd_poly)

            pos = [rnd_poly.pos[0], rnd_poly.pos[1], (i+i_offset+1) * -3.0]
            self.start_pos[i + i_offset] = pos
            xml.add_mesh(rnd_poly.ply, pos=pos, axangle=rnd_poly.angle,
                         scale=rnd_poly.scale, rgba=rnd_poly.rgba)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        self.sim = mjc.MjSim(model)
        self.xml_names = xml.names

        if len(place_polys) > 0:
            self.scene_objs = self.xml_names[-len(place_polys):]
        else:
            self.scene_objs = []
        #self.sim.step()

        # Drop the scene objects.
        for i, scene_obj in enumerate(self.scene_objs):
            idx = self.sim.model.geom_name2id(scene_obj)
            joint_idx = self.sim.model._joint_name2id[scene_obj]
            rb_top_z = self._get_rbound_top()

            rb_new = self.sim.model.geom_rbound[idx]
            rb_drop_z = rb_top_z + rb_new
            correct_drop_z = rb_drop_z - scene_polys[i].subtract_dist

            place_pos = [place_polys[i][1], 0.0, correct_drop_z]
            self.sim.data.qpos[joint_idx*7:(joint_idx*7)+3] = place_pos
            if self.contacts_off:
                self.sim.model.geom_contype[4 + joint_idx] = 1
                self.sim.model.geom_conaffinity[4 + joint_idx] = 1
                assert (self.sim.model.geom_condim[4+joint_idx] == 6)
            self.placed_objs.append(scene_obj)

        # Settle
        step = 0
        for _ in range(self.args.stack_min_steps):
            self._internal_step()
            step += 1
        max_vel = np.abs(self.sim.data.qvel).max()
        vel_zero_count = int(max_vel < VEL_THRESHOLD)
        while vel_zero_count <= 2 and step < self.args.stack_max_steps:
            self._internal_step()
            max_vel = np.abs(self.sim.data.qvel[:, ]).max()
            if max_vel < VEL_THRESHOLD:
                vel_zero_count += 1
            else:
                vel_zero_count = 0
            step += 1

        # for _ in range(self.args.stack_min_steps):
        #     self._internal_step()

        self._log('xml names ' + str(self.xml_names))

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

    def _get_rbound_top(self):
        # Get highest rounding box top z-coordinate of all placed objects

        # If no objects has been placed
        if self.sim.data.qpos is None or len(self.placed_objs) == 0:
            return 0.0

        ymax_list = []
        for obj_name in self.xml_names:
            idx = self.sim.model.geom_name2id(obj_name)
            rbound = self.sim.model.geom_rbound[idx]
            ypos = self.sim.data.geom_xpos[idx, 2]
            ymax = ypos + rbound
            ymax_list.append(ymax)
        return max(ymax_list)


    def is_repeat(self, drop_obj_idx):
        just_dropped_name = self.xml_names[drop_obj_idx]
        return just_dropped_name in self.placed_objs

    def get_dropped_name(self, drop_obj_idx):
        return self.xml_names[drop_obj_idx]


    def step(self, a):
        SAFE_DIST = 0
        reward = 0
        done = False

        should_stop = False
        if self.args.separate_skip:
            should_stop = (a[1] == 1)
            # Remove the skip action
            a = [a[0], *a[2:]]

        if self.args.stack_dim == 0 and not self.args.separate_skip:
            drop_obj_idx = a
        else:
            drop_obj_idx = int(a[0])

        real_idx = self.aval_idx[drop_obj_idx]

        # Get highest rounding box top z-coordinate of all placed objects
        rb_top_z = self._get_rbound_top()

        poly = self.all_polygons[real_idx]

        # Allow to place the same object twice if rendering figures
        if should_stop:
            self.noop += 1
        elif self.is_repeat(drop_obj_idx):
            self._log('Cannot place object again')
            # This polygon has already been placed.
            reward = self.args.double_place_pen
            # Do not place anything more.
            self.num_penalty += 1
        else:
            # Place the new object below the table first
            init_drop_z = -1.0
            drop_obj_pos = [0.0, 0.0, init_drop_z]
            for i in range(self.args.stack_dim):
                drop_obj_pos[i] = a[i+1]

            # Place the new polygon
            self._drop(drop_obj_pos, drop_obj_idx)
            just_dropped_name = self.get_dropped_name(drop_obj_idx)

            # Get rounding box of newly placed object and change its z-coordinates
            idx = self.sim.model.geom_name2id(just_dropped_name)
            rb_new = self.sim.model.geom_rbound[idx]

            # correct z-pos in data.qpos. Subtract a safe subtract_dist from the bounding radius
            # for this specific type of object so the distance is minimized
            joint_ind = self.sim.model._joint_name2id[just_dropped_name]
            rb_drop_z = rb_top_z + SAFE_DIST + rb_new
            correct_drop_z = rb_drop_z - poly.subtract_dist
            self.sim.data.qpos[joint_ind * 7 + 2] = correct_drop_z

            #height_diff = highest - self.prev_height
            #reward = self.args.stack_reward * height_diff

        _, obs = self._settle_sim(render=self.high_render)
        top_height, mean_height = self._get_highest_obj()
        highest = top_height + 0.5
        self.prev_height = highest
        alternate_reward = mean_height * 0.5 + highest * 0.5
        self.render_height = alternate_reward if self.args.stack_mean_height else highest

        if highest > self.max_height:
            self.max_height = highest

        info = {
            'aval': self.aval_idx
        }

        self.cur_episode_step += 1
        if self.cur_episode_step >= self.max_episode_steps:
            self._log('Exceeded max episode count')
            done = True
        elif should_stop:
            self._log('Stop action taken')
            done = True

        if done:
            info['ep_final_height'] = highest
            info['ep_max_height'] = self.max_height
            info['ep_repeat'] = self.num_penalty
            info['ep_no_op'] = self.noop
            # reward is the final height of the tower.
            reward = alternate_reward if self.args.stack_mean_height else highest

        return obs, reward, done, info

    def _add_info_to_render(self, height, is_success, render_arr):
        if self.args.stack_no_text_render:
            return np.array(render_arr)
        height_txt = '%.2f' % height
        scale = self.high_render_dim // SMALL_RENDER_DIM

        def add_txt_fn(frame):
            ret_frame = cv2.putText(frame, height_txt,
                                (int(scale * 1), int(scale * 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.25,
                                (255, 255, 255), 1, cv2.LINE_AA)
            if is_success:
                ret_frame = cv2.putText(frame, 'Yes',
                                    (int(scale * 1), int(scale * 16)),
                                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.25,
                                    (46, 204, 113), 1, cv2.LINE_AA)
            return ret_frame

        return np.array([add_txt_fn(np.copy(frame)) for frame in render_arr])

    def _get_highest_obj(self):
        # Get every 3rd element (Z coord)
        num_objs = len(self.sim.model._joint_name2id)
        z_coords = np.array([self.sim.data.qpos[2 + i * 7] for i in range(num_objs)])
        p_coords = z_coords[z_coords > 0.0]
        p_mean = np.mean(p_coords) if len(p_coords) > 0 else 0.
        if len(z_coords) == 0:
            return (0.0, 0.0)
        return (max(z_coords), p_mean)

    def _internal_step(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        for name, idx in self.sim.model._joint_name2id.items():
            if name not in self.placed_objs:
                qvel[(idx*6):(idx*6)+6] = 0.0
                qpos[(idx*7):(idx*7)+3] = self.start_pos[idx]

            if self.args.constrain_physics:
                # Only have horizontal and vertical movement. Depth is Y axis.
                qvel[(idx*6)+1] = 0.0
                qpos[(idx*7)+1] = 0.0

        self.sim.data.qvel[:] = qvel
        self.sim.data.qpos[:] = qpos
        self.sim.step()

    def _settle_sim(self, vel_threshold=VEL_THRESHOLD, render=False, extra=0):
        step = 0
        imgs = []
        for _ in range(self.args.stack_min_steps + extra):
            self._internal_step()
            if render and step % self.render_freq == 0:
                imgs.append(self.render_img(self.args.stack_render_high))
            step += 1

        if self.sim.data.qvel is not None:
            max_vel = np.abs(self.sim.data.qvel).max()
            vel_zero_count = int(max_vel < vel_threshold)
            while vel_zero_count <=2 and step < self.args.stack_max_steps:
                self._internal_step()
                if render and step % self.render_freq == 0:
                    imgs.append(self.render_img(self.args.stack_render_high))

                max_vel = np.abs(self.sim.data.qvel[:, ]).max()
                if max_vel < vel_threshold:
                    vel_zero_count += 1
                else:
                    vel_zero_count = 0
                step += 1

        self.cur_render = imgs
        ret_img = self.render_img(False)

        return step, ret_img

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    def render(self, mode='rgb_array'):
        return self._add_info_to_render(self.render_height, False, self.cur_render)

    def set_fixed_action_space(self, args, action_set):
        sub_split = np.copy(action_set)
        # Set the shapes in the scene to be these shapes.
        self.usable_objs = [self.all_polygons[i] for i in sub_split]
        self.start_pos = {}
        self._init_scene()
        return sub_split

    def reset(self):
        self.cur_episode_step = 0
        self.num_penalty = 0
        self.noop = 0
        self.prev_height = 0.0
        self.placed_objs = []
        self.max_height = 0.

        if (not self.is_fixed_action_space):
            sub_split = np.copy(self.use_split)

            ply_types = set([self.all_polygons[i].ply for i in sub_split])
            # As a part of the environment sample some of these shapes for
            # usage during this episode.
            #rng = np.random.RandomState(42)
            np.random.shuffle(sub_split)

            sub_split = sub_split[:self.action_set_size]
            # Set the shapes in the scene to be these shapes.
            self.usable_objs = [self.all_polygons[i] for i in sub_split]
            self.start_pos = {}
            self._init_scene()

            self._set_action_space(sub_split)

        img = self.render_img(False)
        return img

    def get_aval(self):
        return self.aval_idx
