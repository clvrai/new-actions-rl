from enum import Enum
from ..constants import *
from .floor import *
from .basic_obj import BasicObj
from .ball import *
from .goal import *
from .ramp import *
from .wall import *
from .trampoline import *
from .see_saw import *
from .puncher import *
from .box import *
from .belt import *
from .cannon import *
from .fixed_circle import *
from .fixed_box import *
from .no_op import NoOp
from .poly import *
from .funnel import *
from .bucket import *
from .ladder import *
from .basket import *


"""
Actions always have a range of -1 to 1.
This function helps convert between that and the real range of the screen.
"""
def convert_action(action, settings):
    x = (action[0] + 1) / 2
    y = (action[1] + 1) / 2
    ret_val = [x * settings.screen_width, y * settings.screen_height]
    return ret_val


class ToolTypes(Enum):
    LONG_FLOOR = 'long_floor'
    MEDIUM_FLOOR = 'medium_floor'
    SHORT_FLOOR = 'short_floor'
    FLOOR = 'floor'
    MARKER_BALL = 'marker_ball'
    TARGET_BALL = 'target_ball'
    TARGET_BALL_TEST = 'target_ball_test'
    TARGET_BALL_TEST2 = 'target_ball_test2'
    BASKET_BALL = 'basket_ball'
    NO_OP = 'no_op'
    RAMP = 'ramp'
    RAMP_30 = 'ramp_30'
    RAMP_45 = 'ramp_45'
    RAMP_60 = 'ramp_60'
    REVERSE_RAMP_30 = 'reverse_ramp_30'
    REVERSE_RAMP_45 = 'reverse_ramp_45'
    REVERSE_RAMP_60 = 'reverse_ramp_60'
    WALL = 'wall'
    WALL_ELASTIC = 'wall_elastic'
    TRAMPOLINE = 'trampoline'
    TRAMPOLINE_45 = 'trampoline_45'
    REVERSE_TRAMPOLINE_45 = 'reverse_trampoline_45'
    VERTICAL_TRAMPOLINE = 'vertical_trampoline'
    GOAL = 'goal'

    BUMP_RAMP_30         = 'bump_ramp_30'
    BUMP_RAMP_45         = 'bump_ramp_45'
    BUMP_RAMP_60         = 'bump_ramp_60'
    BUMP_REVERSE_RAMP_30 = 'bump_reverse_ramp_30'
    BUMP_REVERSE_RAMP_45 = 'bump_reverse_ramp_45'
    BUMP_REVERSE_RAMP_60 = 'bump_reverse_ramp_60'

    VERY_SHORT_FLOOR = 'very_short_floor'

    SEE_SAW = 'see_saw'
    HINGE_SEG = 'hinge_ramp'
    HINGE_SLIDE_SEG = 'hinge_slide_seg'
    PUNCHER = 'puncher'
    BOX = 'box'
    BALL = 'ball'
    BELT = 'belt'
    RIGHT_BELT = 'right_belt'
    CANNON = 'cannon'
    FAN = 'fan'
    FIXED_CIRCLE = 'fixed_circle'
    BOUNCY_CIRCLE = 'bouncy_circle'
    FIXED_BOX = 'fixed_box'
    BOUNCY_BOX = 'bouncy_box'
    FIXED_TRIANGLE = 'fixed_triangle'
    BOUNCY_TRIANGLE = 'bouncy_triangle'
    FIXED_SQUARE = 'fixed_square'
    BOUNCY_SQUARE = 'bouncy_square'
    FIXED_PENTAGON = 'fixed_pentagon'
    BOUNCY_PENTAGON = 'bouncy_pentagon'
    FIXED_HEXAGON = 'fixed_hexagon'
    BOUNCY_HEXAGON = 'bouncy_hexagon'
    FIXED_POLY = 'fixed_poly'
    FUNNEL = 'funnel'
    BUCKET = 'bucket'
    LADDER = 'ladder'
    DOWN_LADDER = 'down_ladder'
    FAST_LADDER = 'fast_ladder'
    BASKET = 'basket'
    GOAL_BALL = 'goal_ball'
    GOAL_STAR = 'goal_star'



"""
Class to ease creating all sorts of tools
"""
class ToolFactory(object):
    def __init__(self):
        self.def_map = {
                # Floors
                ToolTypes.LONG_FLOOR            : LongFloor,
                ToolTypes.MEDIUM_FLOOR          : MediumFloor,
                ToolTypes.SHORT_FLOOR           : ShortFloor,
                ToolTypes.VERY_SHORT_FLOOR      : VeryShortFloor,
                ToolTypes.FLOOR      : Floor,


                # Special
                ToolTypes.MARKER_BALL           : MarkerBall,
                ToolTypes.TARGET_BALL           : TargetBall,
                ToolTypes.TARGET_BALL_TEST      : TargetBallTest,
                ToolTypes.TARGET_BALL_TEST2     : TargetBallTest2,
                ToolTypes.BASKET_BALL           : BasketBall,
                ToolTypes.NO_OP                 : NoOp,
                ToolTypes.GOAL                  : GoalObj,

                # Ramps
                ToolTypes.RAMP                 : Ramp,
                ToolTypes.RAMP_30              : Ramp30,
                ToolTypes.RAMP_45              : Ramp45,
                ToolTypes.RAMP_60              : Ramp60,
                ToolTypes.REVERSE_RAMP_30      : ReverseRamp30,
                ToolTypes.REVERSE_RAMP_45      : ReverseRamp45,
                ToolTypes.REVERSE_RAMP_60      : ReverseRamp60,
                ToolTypes.BUMP_RAMP_30         : BumpRamp30,
                ToolTypes.BUMP_RAMP_45         : BumpRamp45,
                ToolTypes.BUMP_RAMP_60         : BumpRamp60,
                ToolTypes.BUMP_REVERSE_RAMP_30 : BumpReverseRamp30,
                ToolTypes.BUMP_REVERSE_RAMP_45 : BumpReverseRamp45,
                ToolTypes.BUMP_REVERSE_RAMP_60 : BumpReverseRamp60,

                # Walls
                ToolTypes.WALL                  : Wall,
                ToolTypes.WALL_ELASTIC          : WallElastic,

                # Trampolines
                ToolTypes.TRAMPOLINE            : MediumTrampoline,
                ToolTypes.TRAMPOLINE_45         : Trampoline45,
                ToolTypes.REVERSE_TRAMPOLINE_45 : ReverseTrampoline45,
                ToolTypes.VERTICAL_TRAMPOLINE   : VerticalTrampoline,

                # See-saws
                ToolTypes.SEE_SAW   : SeeSaw,
                ToolTypes.HINGE_SEG : HingeSeg,
                ToolTypes.HINGE_SLIDE_SEG: HingeSlideSeg,
                ToolTypes.PUNCHER   : Puncher,
                ToolTypes.BOX       : BoxTool,
                ToolTypes.BALL       : Ball,

                # Belt
                ToolTypes.BELT      : Belt,
                ToolTypes.RIGHT_BELT      : RightBelt,

                # Cannon, Fan
                ToolTypes.CANNON      : Cannon,
                ToolTypes.FAN      : Fan,

                # Shapes
                ToolTypes.FIXED_CIRCLE: FixedCircle,
                ToolTypes.BOUNCY_CIRCLE: BouncyCircle,
                ToolTypes.FIXED_BOX: FixedBox,
                ToolTypes.BOUNCY_BOX: BouncyBox,
                ToolTypes.FIXED_TRIANGLE:  FixedTriangle,
                ToolTypes.BOUNCY_TRIANGLE: BouncyTriangle,
                ToolTypes.FIXED_SQUARE:  FixedSquare,
                ToolTypes.BOUNCY_SQUARE: BouncySquare,
                ToolTypes.FIXED_PENTAGON:  FixedPentagon,
                ToolTypes.BOUNCY_PENTAGON: BouncyPentagon,
                ToolTypes.FIXED_HEXAGON:  FixedHexagon,
                ToolTypes.BOUNCY_HEXAGON: BouncyHexagon,
                ToolTypes.FIXED_POLY: FixedPoly,

                # Funnel
                ToolTypes.FUNNEL: Funnel,

                # Bucket
                ToolTypes.BUCKET: Bucket,

                # Ladder
                ToolTypes.LADDER      : Ladder,
                ToolTypes.DOWN_LADDER      : DownLadder,
                ToolTypes.FAST_LADDER      : FastLadder,

                # Basket
                ToolTypes.BASKET: Basket,

                # Moving Goal
                ToolTypes.GOAL_BALL: GoalBall,
                ToolTypes.GOAL_STAR: GoalStar
            }

    def set_settings(self, settings):
        self.settings = settings

    def create(self, tool_type, pos, add_params={}):
        pos = convert_action(pos, self.settings)
        obj = self.def_map[tool_type](pos, **add_params)
        obj.set_settings(self.settings)
        return obj

