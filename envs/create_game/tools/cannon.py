from .basic_obj import BasicObj, LINE_THICKNESS
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
from pymunk import Body
import pymunk
import pygame as pg
import numpy as np
from .img_tool import ImageTool
from .fixed_box import FixedRect

CANNON_WIDTH = 10.0
CANNON_HEIGHT = 7.3
FAN_HEIGHT = 10.0
FAN_WIDTH = 6.7

def cannon_begin_handler(arbiter, space, data):
    if arbiter.shapes[0].collision_type == MOVING_OBJ_COLLISION_TYPE:

        new_pos, launch_dir = arbiter.shapes[1].properties

        obj = arbiter.shapes[0]
        obj.body.position = new_pos[:]
        obj.body.velocity = pymunk.Vec2d(0., 0.)
        obj.body.angular_velocity = 0.
        obj.body.force = pymunk.Vec2d(0., 0.)
        obj.body.torque = 0.
        obj.body.angle = 0.
        obj.body.apply_impulse_at_local_point(launch_dir[:], (0, 0))
        return False
    return True

def fan_touching_handler(arbiter, space, data):
    if arbiter.shapes[0].collision_type == MOVING_OBJ_COLLISION_TYPE:
        launch_dir = arbiter.shapes[1].properties
        obj = arbiter.shapes[0]
        obj.body.apply_force_at_local_point(launch_dir[:], (0, 0))
        return False
    return True

class Cannon(FixedRect):
    # 1, 5 pi
    def __init__(self, pos, angle=np.pi/3, force=120.0, color='blue'):
        super().__init__(pos, CANNON_WIDTH, CANNON_HEIGHT)
        self.angle = angle
        self.color = color
        self.force = force
        self.launch_dir = force * np.array([np.cos(angle), np.sin(angle)])
        self.collision_type = 2

        self.img = ImageTool('cannon.png', angle, pos[:],
                use_shape=self.shape,
                debug_render=False)

        self.new_pos = [self.pos[0] + np.cos(angle) * (CANNON_WIDTH/2),
            self.pos[1] + np.sin(angle) * (CANNON_HEIGHT/2)]


    def add_to_space(self, space):
        cannon = self.img.get_shape()
        cannon.sensor=True
        cannon.collision_type = self.collision_type
        cannon.properties = (self.new_pos, self.launch_dir)
        self.shape = cannon
        space.add(cannon)
        self.attached_shapes.append(cannon)

        h = space.add_collision_handler(1, self.collision_type)
        h.begin = cannon_begin_handler

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)


class Fan(FixedRect):
    # 1, 5 pi
    def __init__(self, pos, angle=np.pi/3, force=120.0, color='blue'):
        super().__init__(pos, angle, FAN_WIDTH, FAN_HEIGHT)

        self.color = color
        self.launch_dir = force * np.array([np.cos(angle), np.sin(angle)])

        self.img = ImageTool('fan.png', angle, pos[:],
                use_shape=self.shape,
                debug_render=False)
        self.collision_type = 3


    def add_to_space(self, space):
        fan = self.img.get_shape()
        fan.sensor = True
        fan.collision_type = self.collision_type
        fan.properties = self.launch_dir
        self.shape = fan
        space.add(fan)
        self.attached_shapes.append(fan)

        h = space.add_collision_handler(1, self.collision_type)
        h.pre_solve = fan_touching_handler


    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)
