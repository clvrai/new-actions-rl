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
        use_vel = arbiter.shapes[1].velocity[:]
        obj = arbiter.shapes[0]
        obj.body.velocity = pymunk.Vec2d(*(use_vel))
        return False
    return True


class Ladder(Ramp):
    def __init__(self, pos, vel=4.0, length=20.0, color='black'):
        self.angle = 95.0
        self.angle_rad = self.angle * np.pi / 180.0

        super().__init__(pos, length=length, angle=self.angle_rad)

        self.length = length
        self.color = color
        self.vel = vel
        self.collision_type = 7


    def add_to_space(self, space):
        super().add_to_space(space, use_friction=False)
        self.img = ImageTool('ladder.png', self.angle_rad,
                self.pos[:],
                use_shape=self.shape,
                debug_render=False)

        ladder = self.img.get_shape()
        ladder.sensor=True
        ladder.velocity = np.array([4.0 * self.vel * np.cos(self.angle_rad), self.vel * np.sin(self.angle_rad)])

        ladder.collision_type = self.collision_type

        h = space.add_collision_handler(1, self.collision_type)
        h.pre_solve = touching_handler


    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)


class DownLadder(Ladder):
    def __init__(self, pos, vel=-4.0):
        super().__init__(pos, vel=vel)



class FastLadder(Ladder):
    def __init__(self, pos, vel=4.0, length=15.0):
        super().__init__(pos, vel=vel, length=length)
