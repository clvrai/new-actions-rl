import pygame as pg
import numpy as np
import pymunk
import copy
import os.path as osp
from .img_loader import img_library
from .gravity_obj import GravityObj
from ..constants import asset_dir

class ImageTool(object):
    def __init__(self, image_loc, angle, pos, use_shape, debug_render=False,
        elasticity=None, friction=None):

        image_loc = osp.join(asset_dir, image_loc)
        assert image_loc in img_library
        self.img = pg.image.fromstring(*img_library[image_loc]).convert()
        # self.img = pg.image.load(image_loc).convert()
        colorkey = self.img.get_at((0,0))
        self.img.set_colorkey(colorkey, pg.RLEACCEL)
        self.img.set_alpha(255)

        self.use_shape = use_shape
        self.debug_render = debug_render
        self.angle = angle
        self.unscaled_img = self.img

        if elasticity is not None:
            self.use_shape.elasticity = elasticity

        if friction is not None:
            self.use_shape.friction = friction


    def get_shape(self):
        return self.use_shape


    def render(self, screen, scale, coord_convert):
        if scale is None:
            scale = 1

        if self.debug_render:
            if isinstance(self.use_shape, pymunk.Poly):
                pointlist = []
                for v in self.use_shape.get_vertices():
                    x, y = v.rotated(self.use_shape.body.angle) + self.use_shape.body.position
                    point = scale * coord_convert([x, y])
                    pointlist.append([int(point[0]), int(point[1])])

                pg.draw.polygon(screen, pg.Color('black'), pointlist, 0)
            elif isinstance(self.use_shape, pymunk.Segment):
                p1 = scale * coord_convert(self.use_shape.a)
                p2 = scale * coord_convert(self.use_shape.b)

                pg.draw.lines(screen, pg.Color('black'), False, (p1,
                    p2), int(scale * 2 * self.use_shape.radius))
            elif isinstance(self.use_shape, pymunk.Circle):
                draw_pos = scale * coord_convert(self.use_shape.body.position)
                draw_pos[0] = int(draw_pos[0])
                draw_pos[1] = int(draw_pos[1])
                pg.draw.circle(screen, pg.Color('black'), draw_pos,
                    int(self.use_shape.radius * scale))

        bb = self.use_shape.bb

        p1 = scale * coord_convert([bb.left, bb.top])
        draw_rect = (p1[0],
            p1[1],
            scale * (bb.right - bb.left),
            scale * (bb.top - bb.bottom))

        width = (bb.right - bb.left)
        height = (bb.top - bb.bottom)

        img = self.unscaled_img
        if self.use_shape.body.body_type != pymunk.Body.STATIC:
            angle = self.use_shape.body.angle
        else:
            angle = self.angle

        img = pg.transform.rotate(img, angle * (180.0 / np.pi))
        rect1 = img.get_rect()
        rect2 = img.get_bounding_rect()
        w = 1. * rect2[2] / rect1[2]
        h = 1. * rect2[3] / rect1[3]
        img = pg.transform.scale(img,
            [int(scale * width / w), int(scale * height / h)])

        # pg.draw.rect(screen, pg.Color('orange'), draw_rect, 1)

        screen.blit(img, draw_rect, area=img.get_bounding_rect())
