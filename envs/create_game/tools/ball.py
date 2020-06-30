from .gravity_obj import GravityObj, MOVING_OBJ_COLLISION_TYPE
from .goal import GOAL_RADIUS, goal_target_begin_handler
from pymunk import Body
import pymunk
import pygame as pg
from pygame import gfxdraw
from .img_tool import ImageTool
from ..constants import marker_color, target_color, goal_color


class Ball(GravityObj):
    # Elasticity of 1.0 means bouncy with any segments that are declared bouncy
    def __init__(self, pos, mass, radius, elasticity=1.0, friction=1.0,
            use_body=None, color='black'):
        super().__init__(pos)
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        if use_body is None:
            self.body = self._create_body(mass, inertia)
            self.body.position = self.pos
            self.shape = pymunk.Circle(self.body, radius, pymunk.Vec2d(0,0))
        else:
            print('setting from existing body')
            self.body = use_body
            self.shape = pymunk.Circle(self.body, radius, pymunk.Vec2d(*pos))
        self.shape.elasticity = elasticity
        self.shape.friction = friction
        self.radius = radius
        # what colors am i allowed to use?
        # https://htmlcolorcodes.com/color-names/
        self.color = color

        self.set_alpha = 255

        if color == 'basketball':
            img_file = 'basketball.png'
            self.img = ImageTool(img_file, angle=0, pos=pos[:],
                            use_shape=self.shape,
                             debug_render=False)

        self.is_trace = False
        self.scale_radius = 1
        self.segment = False
        self.prev_pos = None

    def get_body(self):
        return self.body

    def get_shape(self):
        return self.shape

    def render(self, screen, scale=1, anti_alias=False):
        if self.color == 'basketball':
            self.img.render(screen, scale, self.flipy)
        else:
            draw_pos = scale * self.flipy(self.body.position)
            draw_pos[0] = int(draw_pos[0])
            draw_pos[1] = int(draw_pos[1])

            if self.is_trace:
                draw_radius = max(2, int(scale * self.radius) // self.scale_radius)
                w, h = pg.display.get_surface().get_size()
                surface = pg.Surface((w, h)).convert_alpha()
                surface.set_colorkey((0,0,0))
                surface.set_alpha(self.set_alpha)

                if self.segment:
                    assert self.prev_pos is not None
                    prev_draw_pos = scale * self.flipy(self.prev_pos)
                    prev_draw_pos[0] = int(prev_draw_pos[0])
                    prev_draw_pos[1] = int(prev_draw_pos[1])
                    pg.draw.line(surface, pg.Color(self.color), prev_draw_pos,
                        draw_pos, 2)
                else:
                    if anti_alias:
                        gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], draw_radius, pg.Color(self.color))
                        gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], draw_radius, pg.Color(self.color))
                    else:
                        pg.draw.circle(surface, pg.Color(self.color), draw_pos, draw_radius)


                screen.blit(surface, (0,0))
            else:
                draw_radius = int(scale * self.radius)
                if anti_alias:
                    gfxdraw.filled_circle(screen, min(draw_pos[0], 32767),
                            min(draw_pos[1], 32767), draw_radius, pg.Color(self.color))
                    gfxdraw.aacircle(screen, min(draw_pos[0], 32767),
                            min(draw_pos[1], 32767), draw_radius, pg.Color(self.color))
                else:
                    pg.draw.circle(screen, pg.Color(self.color), draw_pos, draw_radius)


class MarkerBall(Ball):
    def __init__(self, pos, color=marker_color, radius=GOAL_RADIUS):
        super().__init__(pos, mass=10.0, radius=radius, color=color)


class TargetBall(Ball):
    def __init__(self, pos, color=target_color, radius=GOAL_RADIUS):
        super().__init__(pos, mass=10.0, radius=radius, color=color)
        self.shape.is_target = True

class BasketBall(Ball):
    def __init__(self, pos, color='basketball'):
        super().__init__(pos, mass=10.0, radius=3.5, color=color)
        self.shape.is_target = True

class TargetBallTest(Ball):
    def __init__(self, pos, color='orange'):
        super().__init__(pos, mass=10.0, radius=GOAL_RADIUS, color=color)
        self.shape.body.velocity = pymunk.Vec2d(-10., 10.)


class TargetBallTest2(Ball):
    def __init__(self, pos, color='brown'):
        super().__init__(pos, mass=10.0, radius=GOAL_RADIUS, color=color)
        self.shape.body.velocity = pymunk.Vec2d(25., -35.)




class GoalBall(Ball):
    def __init__(self, pos, color=goal_color, radius=GOAL_RADIUS):
        super().__init__(pos, mass=10.0, radius=radius, color=color)
        self.shape.target_contact = False
        self.shape.is_goal = True

    def add_to_space(self, space):
        super().add_to_space(space)

        h = space.add_collision_handler(MOVING_OBJ_COLLISION_TYPE, MOVING_OBJ_COLLISION_TYPE)
        h.begin = goal_target_begin_handler
