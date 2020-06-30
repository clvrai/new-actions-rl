from .poly import FixedPoly
import numpy as np
import pymunk as pm
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
from .img_tool import ImageTool
from .funnel import dist
import pygame as pg

def bucket_touching_handler(arbiter, space, data):
    locations = arbiter.shapes[1].locations

    if arbiter.shapes[0].in_bucket:
        arbiter.shapes[0].body.position = locations[0][:]
        return False
    elif arbiter.shapes[0].collision_type == MOVING_OBJ_COLLISION_TYPE:

        d1 = dist(arbiter.shapes[0].body.position, locations[1])
        d2 = dist(arbiter.shapes[0].body.position, locations[2])
        d3 = dist(arbiter.shapes[0].body.position, locations[3])
        d4 = dist(arbiter.shapes[0].body.position, locations[4])

        if d1 < d2 and d1 < d3 and d4 < d2 and d4 < d3:
            new_pos = locations[0]
            obj = arbiter.shapes[0]
            obj.body.position = new_pos[:]
            obj.body.velocity = pm.Vec2d(0., 0.)
            obj.body.angular_velocity = 0.
            obj.body.force = pm.Vec2d(0., 0.)
            obj.body.torque = 0.
            obj.body.angle = 0.
            obj.in_bucket = True
            return False
        else:
            return True
    return True

class Bucket(FixedPoly):
    # 1, 5 pi
    def __init__(self, pos, angle = np.pi/4, size=10.0, color='black'):
        super().__init__(pos, n_sides=4, angle=(np.pi/4 + angle), size=size, color=color)

        self.color = color
        self.center_position = [self.pos[0], self.pos[1]]

        self.v_1 = np.array(self.pos) + np.array(self.vertices[0])
        self.v_2 = np.array(self.pos) + np.array(self.vertices[1])
        self.v_3 = np.array(self.pos) + np.array(self.vertices[2])
        self.v_4 = np.array(self.pos) + np.array(self.vertices[3])

        self.img = ImageTool('bucket.png', 0.0 + angle, pos[:],
                use_shape=self.shape,
                debug_render=False)

        self.collision_type = 6


    def add_to_space(self, space):
        bucket = self.img.get_shape()
        bucket.collision_type = self.collision_type
        bucket.locations = [self.center_position, self.v_1, self.v_2, self.v_3, self.v_4]
        self.shape = bucket
        space.add(bucket)
        self.attached_shapes.append(bucket)

        # Called when 1 (movable objects) collides with 3 (bucket)
        h = space.add_collision_handler(1, self.collision_type)
        h.pre_solve = bucket_touching_handler


    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)
