import os.path as osp
import numpy as np
from envs.block_stack.stack_xml import XML
import mujoco_py as mjc
from envs.block_stack.poly_gen import gen_polys

import gym.spaces as spaces
from envs.action_env import ActionEnv
import cv2
from envs.block_stack.stack_env_simplest import *
from envs.block_stack.stack_xml import get_xml_id


class StackEnvAll(StackEnvSimplest):
    def __init__(self, render_freq=25,
                 max_steps=100, asset_path='assets/stl/',
                 episode_steps=EPISODE_STEPS):
        super().__init__(render_freq, max_steps, asset_path, episode_steps)
        self.used_actions = []
        self.prev_names = []


    def is_repeat(self, drop_obj_idx):
        repeated = drop_obj_idx in self.used_actions
        return repeated

    def reset(self):
        self.used_actions = []
        self.prev_names = []
        self.xml_names = []
        self.ep_scene_polys = self.get_place_polys()
        self._drop(None, None, place_scene=True)
        return super().reset()

    def set_args(self, args, set_eval):
        super().set_args(args, set_eval)

    def get_dropped_name(self, drop_obj_idx):
        #real_idx = self.aval_idx[drop_obj_idx]
        #shape_type = self.all_polygons[real_idx].ply
        #use_name = get_xml_id(shape_type, self.prev_names)
        #self.prev_names.append(use_name)
        #return use_name
        return self.xml_names[-1]

    def _drop(self, drop_pos, drop_idx, place_scene=False):
        if drop_idx is not None:
            self.used_actions.append(drop_idx)

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
        xml = XML(self.asset_path, self.shape_names,
                place2d=self.args.stack_dim < 2,
                contacts_off=self.contacts_off)

        for i in range(len(self.ep_scene_polys)):
            place_dat = self.ep_scene_polys[i]
            rnd_poly = self.all_polygons[place_dat[0]]

            pos = [place_dat[1], 0.0, 0.5]
            xml.add_mesh(rnd_poly.ply, pos=pos, axangle=rnd_poly.angle,
                         scale=rnd_poly.scale, rgba=rnd_poly.rgba)

        for poly_i in [self.aval_idx[i] for i in self.used_actions]:
            poly = self.all_polygons[poly_i]

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

        if len(self.xml_names) > 0 and not place_scene:
            just_dropped = self.xml_names[-1]
            self._log('dropping ' + just_dropped)
            joint_ind = self.sim.model._joint_name2id[just_dropped]
            qvel[joint_ind * 6:(joint_ind+1) * 6] = 0
            qpos[joint_ind * 7:(joint_ind+1) * 7] = [*drop_pos, 0, 0, 0, 0]
            self.placed_objs.append(just_dropped)

        if qpos is not None:
            self.sim.data.qvel[:] = qvel
            self.sim.data.qpos[:] = qpos

        if place_scene:
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


    def _init_scene(self):
        pass

    def _internal_step(self):
        self.sim.step()

