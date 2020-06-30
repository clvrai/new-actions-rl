from .basic_obj import BasicObj, LINE_THICKNESS
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
from pymunk import Body
import pymunk
import pygame as pg
import numpy as np
from .img_tool import ImageTool
from .ramp import Ramp

def touching_handler(arbiter, space, data):
    if arbiter.shapes[0].collision_type == MOVING_OBJ_COLLISION_TYPE:
        use_vel = arbiter.shapes[1].velocity
        obj = arbiter.shapes[0]
        obj.body.velocity = pymunk.Vec2d(use_vel, 0)
        return False
    return True


class Belt(Ramp):
    def __init__(self, pos, vel=5.0, length=20.0, color='black'):
        super().__init__(pos, length=length, angle=0.0)

        self.length = length
        self.color = color
        self.vel = vel
        self.collision_type = 4
        if vel < 0:
            self.png = 'belt_left.png'
        else:
            self.png = 'belt_right.png'

    def add_to_space(self, space):
        super().add_to_space(space, use_friction=False)
        self.img = ImageTool(self.png, 0.0,
                self.pos[:],
                use_shape=self.shape,
                debug_render=False)

        belt = self.img.get_shape()
        belt.sensor=True
        belt.velocity = self.vel

        belt.collision_type = self.collision_type

        h = space.add_collision_handler(1, self.collision_type)
        h.pre_solve = touching_handler


    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)


class RightBelt(Belt):
    def __init__(self, pos, vel=-4.0, length=20.0, color='black'):
         super().__init__(pos, vel=vel, length=length, color=color)
