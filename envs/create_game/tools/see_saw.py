from .basic_obj import BasicObj, LINE_THICKNESS
from .ball import Ball
import pymunk
import pygame as pg
from pygame import gfxdraw
from pymunk.vec2d import Vec2d
import math
import numpy as np
from .segment import register_fric_shape, get_polygon
from ..constants import hinge_color
JOINT_RADIUS = 2.0
PLANK_THICKNESS = 1.0

class HingeSeg(BasicObj):
    def __init__(self, pos, length=24.0, friction=1.0, color=hinge_color,
        init_angle=0):
        super().__init__(pos)

        self.length = length
        self.init_angle = init_angle

        # This position is relative to the body itself
        self.temp_pos = (0,0)

        x = self.length * np.cos(init_angle) / 2
        y = self.length * np.sin(init_angle) / 2

        start_pos = [self.temp_pos[0] - x, self.temp_pos[1] - y]
        end_pos = [self.temp_pos[0] + x, self.temp_pos[1] + y]
        self.color = color

        rotation_center_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        rotation_center_body.position = self.pos

        self.body = pymunk.Body(10, 10000)
        self.body.position = self.pos
        self.seg = pymunk.Segment(self.body, start_pos, end_pos, radius=PLANK_THICKNESS // 2)
        self.friction = friction

        self.pin_joint = pymunk.PinJoint(self.body, rotation_center_body, (0,0), (0,0))

        self.shape = self.seg


    def add_to_space(self, space):
        space.add(self.seg, self.body, self.pin_joint)
        self.attached_shapes.append(self.seg)
        self.attached_bodies.append(self.body)
        self.attached_constraints.append(self.pin_joint)

        register_fric_shape(self.seg, self.friction, space, True)

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        body = self.seg.body
        start = body.position + self.seg.a.rotated(body.angle)
        end = body.position + self.seg.b.rotated(body.angle)

        if anti_alias:
            pointlist = get_polygon(start, end, PLANK_THICKNESS)
            pointlist = [scale * self.flipy(x) for x in pointlist]

            gfxdraw.filled_polygon(screen, pointlist, pg.Color(self.color))
            gfxdraw.aapolygon(screen, pointlist, pg.Color(self.color))
        else:
            p1 = scale * self.flipy(start)
            p2 = scale * self.flipy(end)
            pg.draw.lines(screen, pg.Color(self.color), False, (p1, p2), int(scale * PLANK_THICKNESS))


        top = scale * self.flipy(body.position)
        left = scale * self.flipy(Vec2d(body.position[0] - JOINT_RADIUS, body.position[1] - JOINT_RADIUS))
        right = scale * self.flipy(Vec2d(body.position[0] + JOINT_RADIUS, body.position[1] - JOINT_RADIUS))

        pointlist = [(int(top.x), int(top.y)),
                (int(left.x) - 1, int(left.y)),
                (int(right.x), int(right.y))]

        if anti_alias:
            gfxdraw.filled_polygon(screen, pointlist, pg.Color(self.color))
            gfxdraw.aapolygon(screen, pointlist, pg.Color(self.color))
        else:
            pg.draw.polygon(screen, pg.Color(self.color), pointlist)


class HingeSlideSeg(HingeSeg):
    def __init__(self, pos, max_angle=math.pi/6, on_left=False,
            length=24.0, friction=1.0, color=hinge_color, weighted=False):

        if weighted:
            init_angle = -max_angle if on_left else max_angle
        else:
            init_angle = 0
        super().__init__(pos, length, friction, color, init_angle=init_angle)
        self.on_left = on_left
        self.length = length

        # Joint Body
        offset = 1.0
        joint_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        joint_body.position = [self.pos[0] - offset, self.pos[1]]

        # Joint
        joint_limit = 2 * offset * math.sin(max_angle/2)
        self.joint = pymunk.SlideJoint(self.body, joint_body,
                (-offset * math.cos(init_angle), -offset * math.sin(init_angle)),
                (0,0), 0, joint_limit)

    def add_to_space(self, space):
        super().add_to_space(space)
        space.add(self.joint)
        self.attached_constraints.append(self.joint)


class SeeSaw(HingeSlideSeg):
    def __init__(self, pos, max_angle=math.pi/6, length=24.0, friction=1.0,
            on_left=False,
            ball_mass=0.1, ball_radius=1.5, ball_elasticity=1.0,
            ball_friction=1.0, color=hinge_color):

        super().__init__(pos, max_angle, on_left, length, friction, color,
            weighted=True)

        init_angle = max_angle if on_left else (np.pi - max_angle)

        self.weight = pymunk.Body(ball_mass, ball_mass)
        self.weight_dist = (3 * length / 8.0)

        self.weight.position = Vec2d(self.pos[0] + self.weight_dist * math.cos(init_angle),
            self.pos[1] - self.weight_dist * math.sin(init_angle))

        self.weight_radius = ball_radius
        self.pivot_joint = pymunk.PivotJoint(self.body, self.weight,
                self.weight.position)
        self.gear_joint = pymunk.GearJoint(self.body, self.weight,
                phase=0.0,
                ratio=1.0)



    def add_to_space(self, space):
        super().add_to_space(space)
        space.add(self.weight, self.pivot_joint, self.gear_joint)
        self.attached_bodies.append(self.weight)
        self.attached_constraints.extend([self.pivot_joint, self.gear_joint])


    def render(self, screen, scale=1, anti_alias=False):
        super().render(screen, scale)
        draw_radius = int(scale * self.weight_radius)

        center = scale * self.flipy(self.weight.position)

        if anti_alias:
            gfxdraw.filled_circle(screen, int(center.x), int(center.y), draw_radius, pg.Color(hinge_color))
            gfxdraw.aacircle(screen, int(center.x), int(center.y), draw_radius, pg.Color(self.color))
        else:
            pg.draw.circle(screen, pg.Color(hinge_color), (int(center.x), int(center.y)), draw_radius)
