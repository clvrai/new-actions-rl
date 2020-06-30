from .segment import Segment
from .floor import ELASTICITY
import numpy as np


class Ramp(Segment):
    def __init__(self, pos, length, angle, friction=1.0, elasticity=ELASTICITY,
        color='STEELBLUE', thickness=None):
        y = length * np.sin(angle)
        x = length * np.cos(angle)

        super().__init__([pos[0] - x/2, pos[1] - y/2],[pos[0] + x/2, pos[1] + y/2],
            pos, friction, elasticity, color=color, thickness=thickness)




class Ramp30(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=np.pi / 6.0, friction=1.0)

class Ramp60(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=np.pi / 3.0, friction=1.0)

class Ramp45(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=np.pi / 4.0, friction=1.0)


class ReverseRamp30(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=-np.pi / 6.0, friction=1.0)

class ReverseRamp60(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=-np.pi / 3.0, friction=1.0)

class ReverseRamp45(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=-np.pi / 4.0, friction=1.0)


MED_BOUNCE = 0.8
BUMP_COLOR = 'magenta'
class BumpRamp30(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=np.pi / 6.0, friction=1.0,
                color=BUMP_COLOR, elasticity=MED_BOUNCE)

class BumpRamp60(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=np.pi / 3.0, friction=1.0,
                color=BUMP_COLOR, elasticity=MED_BOUNCE)

class BumpRamp45(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=np.pi / 4.0, friction=1.0,
                color=BUMP_COLOR, elasticity=MED_BOUNCE)


class BumpReverseRamp30(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=-np.pi / 6.0, friction=1.0,
                color=BUMP_COLOR, elasticity=MED_BOUNCE)

class BumpReverseRamp60(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=-np.pi / 3.0, friction=1.0,
                color=BUMP_COLOR, elasticity=MED_BOUNCE)

class BumpReverseRamp45(Ramp):
    def __init__(self, pos):
        super().__init__(pos, length=16, angle=-np.pi / 4.0, friction=1.0,
                color=BUMP_COLOR, elasticity=MED_BOUNCE)


