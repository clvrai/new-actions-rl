from .gravity_obj import GravityObj
import pymunk
import pygame as pg
from .img_tool import ImageTool


class BoxTool(GravityObj):
    # Elasticity of 1.0 means bouncy with any segments that are declared bouncy
    def __init__(self, pos, mass=10.0, size=5.0, elasticity=1.0, friction=1.0, color='slategray'):
        super().__init__(pos)

        moment = pymunk.moment_for_box(mass, (size, size))
        self.body = pymunk.Body(mass, moment)
        self.body.position = pymunk.Vec2d(pos[0], pos[1])

        self.shape = pymunk.Poly.create_box(self.body, (size, size))
        self.shape.elasticity = elasticity
        self.shape.friction = friction
        self.color = color
        self.size = size

        if color == 'box':
            self.img = ImageTool('box.png', angle=0, pos=pos[:],
                    use_shape=self.shape,
                    debug_render=False)

    def get_body(self):
        return self.body

    def render(self, screen, scale=1, anti_alias=False):
        if self.color == 'box':
            self.img.render(screen, scale, self.flipy)
        else:
            draw_pos = scale * self.flipy(self.body.position)

            draw_size = scale * self.size
            draw_rect = (draw_pos[0] - (draw_size / 2),
                         draw_pos[1] - (draw_size / 2),
                         draw_size,
                         draw_size)

            pg.draw.rect(screen, pg.Color(self.color), draw_rect)
