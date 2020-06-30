from .segment import Segment

ELASTICITY = 0.5

class Floor(Segment):
    def __init__(self, pos, length=16.0, friction=1.0, elasticity=ELASTICITY, color='STEELBLUE'
        , sensor=False, thickness=None):
        start_x = pos[0] - (length / 2)
        end_x = pos[0] + (length / 2)
        self.sensor = sensor

        super().__init__([start_x, pos[1]], [end_x, pos[1]], pos,
            friction, elasticity=elasticity, color=color, thickness=thickness)

    def add_to_space(self, space, use_friction=True):
        super().add_to_space(space, use_friction=use_friction)
        self.shape.sensor = self.sensor

class LongFloor(Floor):
    def __init__(self, pos, thickness=None, color='STEELBLUE'):
        super().__init__(pos, length=24.0, friction=1.0, thickness=thickness, color=color)


class MediumFloor(Floor):
    def __init__(self, pos, thickness=None, color='STEELBLUE'):
        super().__init__(pos, length=16.0, friction=1.0, thickness=thickness, color=color)


class ShortFloor(Floor):
    def __init__(self, pos, thickness=None, color='STEELBLUE'):
        super().__init__(pos, length=8.0, friction=1.0, thickness=thickness, color=color)


class VeryShortFloor(Floor):
    def __init__(self, pos, thickness=None, color='STEELBLUE'):
        super().__init__(pos, length=4.0, friction=1.0, thickness=thickness, color=color)
