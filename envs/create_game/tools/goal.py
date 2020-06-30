from .fixed_obj import FixedObj
import pymunk
import pygame as pg
from pygame import gfxdraw
from ..constants import goal_color
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
from .poly import Star

GOAL_RADIUS = 4.0
GOAL_OBJ_COLLISION_TYPE = 15

def goal_target_begin_handler(arbiter, space, data):
    if hasattr(arbiter.shapes[1], 'is_target') and arbiter.shapes[1].is_target and \
         hasattr(arbiter.shapes[0], 'is_goal') and arbiter.shapes[0].is_goal:
        arbiter.shapes[0].target_contact = True
    elif hasattr(arbiter.shapes[0], 'is_target') and arbiter.shapes[0].is_target and \
         hasattr(arbiter.shapes[1], 'is_goal') and arbiter.shapes[1].is_goal:
        arbiter.shapes[1].target_contact = True
    return True


class GoalObj(FixedObj):
    def __init__(self, pos, color=goal_color, radius=GOAL_RADIUS):
        super().__init__(pos)
        mass = 1.0
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        self.body = self._create_body(mass, inertia)
        self.body.position = self.pos
        self.shape = pymunk.Circle(self.body, radius, pymunk.Vec2d(0,0))
        self.radius = radius
        self.shape.sensor = True
        self.color = color
        self.collision_type = GOAL_OBJ_COLLISION_TYPE
        self.shape.is_goal = True

    def add_to_space(self, space):
        super().add_to_space(space)
        h = space.add_collision_handler(GOAL_OBJ_COLLISION_TYPE, MOVING_OBJ_COLLISION_TYPE)
        h.begin = goal_target_begin_handler

    def get_body(self):
        return self.body

    def get_shape(self):
        return self.shape

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1
        draw_radius = int(self.radius * scale)
        draw_pos = scale * self.flipy(self.body.position)
        draw_pos[0] = int(draw_pos[0])
        draw_pos[1] = int(draw_pos[1])

        if anti_alias:
            gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], draw_radius, pg.Color(self.color))
            gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], draw_radius, pg.Color(self.color))
        else:
            pg.draw.circle(screen, pg.Color(self.color), draw_pos, draw_radius)


class GoalStar(Star):
    def __init__(self, pos, color=goal_color, radius=GOAL_RADIUS):
        super().__init__(pos, n_sides=5, angle=0.0,
                size=GOAL_RADIUS, color=color)
        self.shape.sensor = True
        self.shape.target_contact = False
        self.shape.is_goal = True
        self.collision_type = GOAL_OBJ_COLLISION_TYPE

    def add_to_space(self, space):
        super().add_to_space(space)
        h = space.add_collision_handler(GOAL_OBJ_COLLISION_TYPE, MOVING_OBJ_COLLISION_TYPE)
        h.begin = goal_target_begin_handler
