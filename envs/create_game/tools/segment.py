from .basic_obj import BasicObj
from .fixed_obj import FixedObj
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
import pymunk
import pygame as pg
from pygame import gfxdraw
import math

LINE_THICKNESS = 2.0

def seg_coll(arbiter, space, data):
    obj = arbiter.shapes[0]
    if isinstance(obj, pymunk.shapes.Circle) and obj.collision_type == MOVING_OBJ_COLLISION_TYPE:
        fric, should_damp_vel = arbiter.shapes[1].properties
        if fric <= 1.0 and fric >= 0:
            dampening = (1.0 - fric) + 1.0
        elif fric > 1.0 and fric <= 2.0:
            dampening = 1.0 - (fric - 1.0)

        # Only for rolling friction
            obj.body.angular_velocity *= dampening
            if should_damp_vel:
                obj.body.velocity = dampening * obj.body.velocity

FRIC_COLL_TYPE = 10

def register_fric_shape(shape, friction, space, should_damp_vel=False):
    shape.collision_type = FRIC_COLL_TYPE
    shape.properties = (friction, should_damp_vel)

    h = space.add_collision_handler(MOVING_OBJ_COLLISION_TYPE, FRIC_COLL_TYPE)
    h.post_solve = seg_coll

def get_polygon(left, right, thickness):
    center = ( (left[0] + right[0]) / 2., (left[1] + right[1]) / 2.)
    length = math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)
    angle = math.atan2(left[1] - right[1], left[0] - right[0])

    UL = (center[0] + (length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
          center[1] + (thickness / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
    UR = (center[0] - (length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
          center[1] + (thickness / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))
    BL = (center[0] + (length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
          center[1] - (thickness / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
    BR = (center[0] - (length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
          center[1] - (thickness / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))

    return (UL, UR, BR, BL)


"""
Segment is just like a wall. Immovable and reflects stuff
"""
class Segment(FixedObj):
    def __init__(self, start_pos, end_pos, mid_pos, friction=1.0,
            elasticity=0.0, color='black', thickness=None):
        super().__init__(mid_pos)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.mid_pos = mid_pos
        self.friction = friction
        if self.friction < 1.0 or self.friction > 2.0:
            raise ValueError('Invalid friction value')
        self.elasticity = elasticity
        # what colors am i allowed to use?
        # https://htmlcolorcodes.com/color-names/
        self.color = color
        self.thickness = thickness if thickness is not None else LINE_THICKNESS
        self.pointlist = get_polygon(start_pos, end_pos, self.thickness)

    def add_to_space(self, space, use_friction=True):
        self.shape = pymunk.Segment(space.static_body, self.start_pos, self.end_pos,
                radius=self.thickness // 2)
        self.shape.friction = self.friction
        self.shape.elasticity = self.elasticity

        space.add(self.shape)
        self.attached_shapes.append(self.shape)
        if use_friction:
            register_fric_shape(self.shape, self.friction, space)


    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        if anti_alias:
            pointlist = [scale * self.flipy(x) for x in self.pointlist]
            gfxdraw.filled_polygon(screen, pointlist, pg.Color(self.color))
            gfxdraw.aapolygon(screen, pointlist, pg.Color(self.color))
        else:
            p1 = scale * self.flipy(self.start_pos)
            p2 = scale * self.flipy(self.end_pos)
            pg.draw.lines(screen, pg.Color(self.color), False, (p1,
                p2), int(scale * self.thickness))

    def get_shape(self):
        return self.shape
