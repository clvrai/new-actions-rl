from .poly import FixedPoly
import numpy as np
import pymunk as pm
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
from .img_tool import ImageTool
import pygame as pg

def dist(a,b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def funnel_touching_handler(arbiter, space, data):
    if arbiter.shapes[0].collision_type == MOVING_OBJ_COLLISION_TYPE:
        locations = arbiter.shapes[1].locations

        d1 = dist(arbiter.shapes[0].body.position, locations[1])
        d2 = dist(arbiter.shapes[0].body.position, locations[2])
        d3 = dist(arbiter.shapes[0].body.position, locations[3])

        if d2 > d1 and d2 > d3:
            new_pos = locations[0]

            obj = arbiter.shapes[0]
            obj.body.position = new_pos[:]
            obj.body.velocity = pm.Vec2d(0., 0.)
            obj.body.angular_velocity = 0.
            obj.body.force = pm.Vec2d(0., 0.)
            obj.body.torque = 0.
            obj.body.angle = 0.
            return False
        else:
            return True
    return True

class Funnel(FixedPoly):
    # 1, 5 pi
    def __init__(self, pos, angle = np.pi/4, size=10.0, color='black'):
        super().__init__(pos, n_sides=3, angle = np.pi / 6 + angle, size=size, color=color)
        self.color = color
        self.nozzle_position = [self.pos[0] + self.vertices[1][0], self.pos[1] + self.vertices[1][1]]

        self.v_1 = np.array(self.pos) + np.array(self.vertices[0])
        self.v_2 = np.array(self.pos) + np.array(self.vertices[1])
        self.v_3 = np.array(self.pos) + np.array(self.vertices[2])

        pos1 = [self.nozzle_position[0], self.nozzle_position[1]]

        img_pos = [pos1[0] -  3. * size / 4 * np.sin(angle),
                pos1[1] + 3. * size / 4 * np.cos(angle)]

        self.img = ImageTool('funnel.png', angle, img_pos,
                use_shape=self.shape,
                debug_render=False)
        self.collision_type = 5


    def add_to_space(self, space):
        funnel = self.img.get_shape()
        funnel.collision_type = self.collision_type
        funnel.locations = [self.nozzle_position, self.v_1, self.v_2, self.v_3]
        self.shape = funnel
        space.add(funnel)
        self.attached_shapes.append(funnel)

        h = space.add_collision_handler(1, self.collision_type)
        h.pre_solve = funnel_touching_handler

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)
