import numpy as np
from pymunk import Vec2d
import pygame as pg

LINE_THICKNESS = 2.0


class BasicObj(object):
    def __init__(self, pos):
        self.pos = pos
        self.low = None
        self.high = None
        self.attached_shapes = []
        self.attached_bodies = []
        self.attached_constraints = []
        self.collision_type = 0
        self.settings = None

    def set_settings(self, settings):
        self.settings = settings

    def __str__(self):
        return str(self.__class__) + ": " + str(self.pos)

    def get_body(self):
        return self.shape.body

    def get_shape(self):
        return self.shape

    def add_to_space(self, space):
        self.shape.collision_type = self.collision_type
        self.shape.in_bucket = False
        space.add(self.get_body(), self.shape)
        self.attached_bodies.append(self.get_body())
        self.attached_shapes.append(self.shape)

    def remove_from_space(self, space):
        space.remove(
            [*self.attached_bodies,
            *self.attached_shapes,
            *self.attached_constraints]
            )

    def render(self, screen, scale=None, anti_alias=False):
        raise NotImplemented('not implemented')

    def flipy(self, p):
        assert self.settings is not None, 'Must set settings'
        """Convert chipmunk physics to pygame coordinates."""
        return Vec2d(p[0], -p[1] + self.settings.screen_height)

    def render_bb(self, screen, scale):
        shape = self.get_shape()
        shape.cache_bb()
        bb = shape.bb
        p1 = scale * self.flipy([bb.left, bb.top])
        draw_rect = (p1[0],
            p1[1],
            scale * (bb.right - bb.left),
            scale * (bb.top - bb.bottom))

        pg.draw.rect(screen, pg.Color('black'), draw_rect, 1)
