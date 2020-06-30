from .tools.fixed_obj import FixedObj
from .tools.trampoline import Trampoline
from .tools.floor import Floor
from .tools.wall import Wall
from .tools.ramp import Ramp
from .tools.no_op import NoOp
from .tools.see_saw import *
from .tools.puncher import *
from .tools.box import *
from .tools.belt import *
from .tools.cannon import *
from .tools.fixed_circle import *
from .tools.fixed_box import *
from .tools.poly import *
from .tools.funnel import *
from .tools.bucket import *
from .tools.tool_factory import convert_action
from .tool_gen_filters import get_tools_from_filters, get_tool_prop
from .constants import fixed_color, bouncy_color, hinge_color

from collections import namedtuple, defaultdict

import numpy as np
import random
from copy import copy

def does_div(x, y, minn):
    r = (x - minn) / y
    n_r = round(r)
    return abs(n_r - r) < 1e-5

def get_all(arr, gap, minn):
    return [x for x in arr if does_div(x, gap, minn)]



class ToolType(object):
    def __init__(self, tool_id, tool_type, elasticity, angle, length, extra_info={}):
        self.tool_id = tool_id
        self.tool_type = tool_type
        self.elasticity = elasticity
        self.angle = angle
        self.length = length
        self.extra_info = extra_info

    # Need to edit this
    def to_gt(self, one_hot):
        #return [0,0]

        def_extra = defaultdict(float)
        for k,v in self.extra_info.items():
            def_extra[k] = float(v)

        return [*one_hot,
                 0 if self.elasticity is None else self.elasticity,
                 0 if self.angle is None else self.angle,
                 0 if self.length is None else self.length,
                 def_extra['friction'],
                 def_extra['max_angle'],
                 def_extra['ball_mass'],
                 def_extra['on_left'],
                 def_extra['force'],
                 def_extra['vel']]


    def __str__(self):
        out_str = self.tool_type
        if self.angle is not None:
            out_str += ' Angle: %.2f' % (self.angle * 180.0 / np.pi)
        if self.elasticity is not None:
            out_str += ' Elast: %.2f' % self.elasticity
        if self.length is not None:
            out_str += ' Len: %.2f' % self.length

        for k, v in self.extra_info.items():
            out_str += ' %s: %.2f' % (k, v)
        return out_str



#################################################################


class ToolGenerator:
    def __init__(self, gran_factor):
        self.tools = None

        seg_angle_gap = 15.0 / gran_factor

        id = 0

        '''
            Ramp TOOLS (Ramps, floors, trampolines, walls)
            NewMain: e = 3, a = 12, l = 5 => 180
        '''
        seg_length_gap = 4

        self.ramp_tools = []
        e_array = [0.1, 0.3, 0.5]
        seg_angle_array = np.arange(0, 180, seg_angle_gap,  dtype=np.float32)
        seg_length_array = np.arange(6, 22.001, seg_length_gap, dtype=np.float32)
        seg_fric_array = [1.0]
        self.seg_angle_arr = seg_angle_array

        for elasticity in e_array:
            for angle in seg_angle_array:
                for length in seg_length_array:
                    for fric in seg_fric_array:
                        tt = ToolType(id, 'Ramp',
                            elasticity,
                            angle * np.pi/180,
                            length, {
                                'friction': fric
                                }
                            )
                        self.ramp_tools.append(tt)
                        id += 1

        '''
            TRAMPOLINE TOOLS (Ramps, floors, trampolines, walls)
            NewMain: e = 3, a = 12, l = 5 => 180
        '''
        self.trampoline_tools = []
        e_array = [0.8, 1., 1.2]

        for elasticity in e_array:
            for angle in seg_angle_array:
                for length in seg_length_array:
                    tt = ToolType(id, 'Trampoline',
                        elasticity,
                        angle * np.pi/180,
                        length
                        )
                    self.trampoline_tools.append(tt)
                    id += 1

        '''
            CIRCLE TOOLS
            NewMain: e = 5, r = 4 => 20
        '''
        self.fixed_ball_tools = []
        e_array = [0.1, 0.2, 0.3, 0.4, 0.5]

        self.radius_gap_factor = 2.0
        self.radius_min = 3.0
        self.ball_radius_arr = np.arange(self.radius_min, 9.001,
            self.radius_gap_factor, dtype=np.float32) # => 4

        for elasticity in e_array:
            for radius in self.ball_radius_arr:
                tt = ToolType(
                    id,
                    'Fixed_Ball',
                    elasticity,
                    None,
                    radius
                )
                self.fixed_ball_tools.append(tt)
                id += 1
        '''
            BOUNCY Ball TOOLS
            NewMain: e = 5, r = 4 => 20
        '''
        self.bouncy_ball_tools = []
        e_array = [0.8, 0.9, 1., 1.1, 1.2]
        for elasticity in e_array:
            for radius in self.ball_radius_arr:
                tt = ToolType(
                    id,
                    'Bouncy_Ball',
                    elasticity,
                    None,
                    radius
                )
                self.bouncy_ball_tools.append(tt)
                id += 1

        '''
            BOX Tools
            e = 5, s = 4 => 20
        '''
        self.box_tools = []
        e_array = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.box_size_factor = 3.0
        self.box_size_min = 5.0
        self.box_size_arr = np.arange(self.box_size_min, 16.001, self.box_size_factor)

        for elasticity in e_array:
            for size in self.box_size_arr:
                tt = ToolType(
                    id,
                    'Fixed_Box',
                    elasticity,
                    None,
                    size
                )
                self.box_tools.append(tt)
                id += 1

        '''
            BOUNCY BOX Tools
            e = 5, s = 4 => 20
        '''
        self.bouncy_box_tools = []
        e_array = [0.8, 0.9, 1., 1.1, 1.2]

        for elasticity in e_array:
            for size in self.box_size_arr:
                tt = ToolType(
                    id,
                    'Bouncy_Box',
                    elasticity,
                    None,
                    size
                )
                self.bouncy_box_tools.append(tt)
                id += 1


        '''
            TRIANGLE Tools
            e = 3, s = 7, a = 8 => 168
        '''
        self.triangle_tools = []
        e_array = [0.1, 0.3, 0.5]
        self.triangle_size_min = 3.0
        triangle_size_array = np.arange(self.triangle_size_min, 9.001, 1.)

        triangle_angle_gap = 15.0 / gran_factor
        triangle_max_angle = 120
        self.triangle_angle_arr = np.arange(0, triangle_max_angle, triangle_angle_gap,  dtype=np.float32)

        for angle in self.triangle_angle_arr:
            for elasticity in e_array:
                for size in triangle_size_array:
                    tt = ToolType(
                        id,
                        'Fixed_Triangle',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.triangle_tools.append(tt)
                    id += 1

        '''
            BOUNCY TRIANGLE Tools
            e = 3, s = 7, a = 8 => 168
        '''
        self.bouncy_triangle_tools = []
        e_array = [0.8, 1., 1.2]

        for angle in self.triangle_angle_arr:
            for elasticity in e_array:
                for size in triangle_size_array:
                    tt = ToolType(
                        id,
                        'Bouncy_Triangle',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.bouncy_triangle_tools.append(tt)
                    id += 1



        '''
            SQUARE Tools
            e = 3, s = 7, a = (6-1)=5 => 105
        '''
        self.square_tools = []
        e_array = [0.1, 0.3, 0.5]
        self.square_size_factor = 3.0
        self.square_size_min = 3.0
        square_size_array = np.arange(self.square_size_min, 9.001, 1.)


        square_angle_gap = 15.0 / gran_factor
        square_max_angle = 90

        self.square_angle_arr = np.arange(0, square_max_angle, square_angle_gap,  dtype=np.float32).tolist()

        # Alternatively, combine this with boxes
        if 45. in self.square_angle_arr:
            self.square_angle_arr.remove(45.)


        for angle in self.square_angle_arr:
            for elasticity in e_array:
                for size in square_size_array:
                    tt = ToolType(
                        id,
                        'Fixed_Square',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.square_tools.append(tt)
                    id += 1

        '''
            BOUNCY SQUARE Tools
            e = 3, s = 7, a = 5 => 105
        '''
        self.bouncy_square_tools = []
        e_array = [0.8, 1., 1.2]

        for angle in self.square_angle_arr:
            for elasticity in e_array:
                for size in square_size_array:
                    tt = ToolType(
                        id,
                        'Bouncy_Square',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.bouncy_square_tools.append(tt)
                    id += 1



        '''
            PENTAGON Tools
            e = 3, s = 7, a = 5 => 105
        '''
        self.pentagon_tools = []
        e_array = [0.1, 0.3, 0.5]
        self.pentagon_size_factor = 3.0
        self.pentagon_size_min = 3.0
        pentagon_size_array = np.arange(self.pentagon_size_min, 9.001, 1.)

        pentagon_angle_gap = 15.0 / gran_factor
        pentagon_max_angle = 72
        self.pentagon_angle_arr = np.arange(0, pentagon_max_angle, pentagon_angle_gap,  dtype=np.float32)

        for angle in self.pentagon_angle_arr:
            for elasticity in e_array:
                for size in pentagon_size_array:
                    tt = ToolType(
                        id,
                        'Fixed_Pentagon',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.pentagon_tools.append(tt)
                    id += 1

        '''
            BOUNCY PENTAGON Tools
            e = 3, s = 7, a = 5 => 105
        '''
        self.bouncy_pentagon_tools = []
        e_array = [0.8, 1., 1.2]

        for angle in self.pentagon_angle_arr:
            for elasticity in e_array:
                for size in pentagon_size_array:
                    tt = ToolType(
                        id,
                        'Bouncy_Pentagon',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.bouncy_pentagon_tools.append(tt)
                    id += 1


        '''
            HEXAGON Tools
            e = 3, s = 7, a = 4 => 84
        '''
        self.hexagon_tools = []
        e_array = [0.1, 0.3, 0.5]
        self.hexagon_size_factor = 3.0
        self.hexagon_size_min = 3.0
        hexagon_size_array = np.arange(self.hexagon_size_min, 9.001, 1.)

        hexagon_angle_gap = 15.0 / gran_factor
        hexagon_max_angle = 60
        self.hexagon_angle_arr = np.arange(0, hexagon_max_angle, hexagon_angle_gap,  dtype=np.float32)

        for angle in self.hexagon_angle_arr:
            for elasticity in e_array:
                for size in hexagon_size_array:
                    tt = ToolType(
                        id,
                        'Fixed_Hexagon',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.hexagon_tools.append(tt)
                    id += 1

        '''
            BOUNCY HEXAGON Tools
            e = 3, s = 7, a = 4 => 84
        '''
        self.bouncy_hexagon_tools = []
        e_array = [0.8, 1., 1.2]

        for angle in self.hexagon_angle_arr:
            for elasticity in e_array:
                for size in hexagon_size_array:
                    tt = ToolType(
                        id,
                        'Bouncy_Hexagon',
                        elasticity,
                        angle * np.pi/180,
                        size
                    )
                    self.bouncy_hexagon_tools.append(tt)
                    id += 1

        '''
            HINGE tools
            Hinge   : f * l = 11 * 5 => 55
            Hinge_Constrained: l = 5, a = 6, f = 3 => 90
            See-Saw     : l = 5, a = 4, l/r = 2, ball_mass = 4, f = 1 => 160
        '''
        self.hinge_tools = []

        self.hinge_length_factor = 6.0
        self.hinge_length_min = 6.0
        self.hinge_length_arr = np.arange(self.hinge_length_min, 34.501, self.hinge_length_factor)
        hinge_fric_array = np.arange(1.0, 1.1001, 0.01)

        for hinge_length in self.hinge_length_arr:
            for fric in hinge_fric_array:
                tt = ToolType(
                    id,
                    'Hinge',
                    None,
                    None,
                    hinge_length, {
                        'friction': fric
                        }
                    )
                self.hinge_tools.append(tt)
                id += 1

        self.hinge_constrained_tools = []

        hinge_constrained_length_array = np.arange(12.0, 21.001, 2.0)
        self.hinge_angle_factor = 15.0
        self.hinge_angle_min = 5.0
        hinge_angle_gap = self.hinge_angle_factor / gran_factor
        self.hinge_angle_arr = np.arange(self.hinge_angle_min, 90.001, hinge_angle_gap)
        hinge_fric_array = np.arange(1.0, 1.1001, 0.05)

        for length in hinge_constrained_length_array:
            for angle in self.hinge_angle_arr:
                for fric in hinge_fric_array:
                    tt = ToolType(
                        id,
                        'Hinge_Constrained',
                        None,
                        None,
                        length, {
                            'max_angle':  angle * np.pi/180,
                            'friction': fric
                            }
                        )
                    self.hinge_constrained_tools.append(tt)
                    id += 1

        self.see_saw_tools = []
        see_saw_length_array = np.arange(12.0, 21.001, 2.0)
        ball_mass_array = np.arange(2.0, 14.001, 4.0) # 2, 6, 10, 14

        self.see_saw_angle_factor = 15.0
        self.see_saw_angle_min = 5.0
        see_saw_angle_gap = self.see_saw_angle_factor / gran_factor
        self.see_saw_angle_arr = np.arange(self.see_saw_angle_min,
            60.001,
            see_saw_angle_gap)
        # see_saw_fric_array = np.arange(1.0, 1.1001, 0.05)
        see_saw_fric_array = [1.0]

        for see_saw_length in see_saw_length_array:
            for angle in self.see_saw_angle_arr:
                for on_left in [True, False]:
                    for ball_mass in ball_mass_array:
                        for fric in see_saw_fric_array:
                            tt = ToolType(
                                id,
                                'See_Saw',
                                None,
                                None,
                                see_saw_length,
                                {
                                    'max_angle': angle * np.pi/180,
                                    'on_left': on_left,
                                    'ball_mass': ball_mass,
                                    'friction': fric
                                }
                            )
                            self.see_saw_tools.append(tt)
                            id += 1
        '''
            Cannon tools
            a = 24, force = 7 => 168
        '''
        self.cannon_tools = []

        cannon_angle_gap = 15.0 / gran_factor
        self.cannon_angle_arr = np.arange(0.0, 360.0, cannon_angle_gap)
        force_array = np.arange(60.0, 121.0, 10.0)

        for angle in self.cannon_angle_arr:
            for force in force_array:
                tt = ToolType(
                    id,
                    'Cannon',
                    None,
                    angle * np.pi / 180,
                    None,
                    {'force': force}
                )
                self.cannon_tools.append(tt)
                id += 1

        '''
            Fan tools
            a = 24, force = 7 => 168
        '''
        self.fan_tools = []
        force_array = np.arange(30.0, 121.0, 15.0)

        fan_angle_gap = 15.0 / gran_factor
        self.fan_angle_arr = np.arange(0.0, 360.0, fan_angle_gap)

        for angle in self.fan_angle_arr:
            for force in force_array:
                tt = ToolType(
                    id,
                    'Fan',
                    None,
                    angle * np.pi / 180,
                    None,
                    {'force': force}
                )
                self.fan_tools.append(tt)
                id += 1

        '''
            Belt tools
            l = 10, vel = 12 => 120
        '''
        self.belt_tools = []

        vel_gap = 1.0
        v_plus = np.arange(1.0, 6.001, vel_gap).tolist()
        v_neg = [-x for x in v_plus]
        velocity_array = v_plus + v_neg
        self.belt_vel_arr = velocity_array

        belt_length_array = np.arange(6.0, 24.001, 2.0)
        for length in belt_length_array:
            for vel in velocity_array:
                tt = ToolType(
                    id,
                    'Belt',
                    None,
                    None,
                    length,
                    {'vel': vel}
                )
                self.belt_tools.append(tt)
                id += 1

        '''
            Funnel tools
            l = 5, a = 24 => 120
        '''
        self.funnel_tools = []

        self.funnel_size_min = 3.0
        funnel_size_array = np.arange(self.funnel_size_min, 11.001, 2.)

        funnel_angle_gap = 15.0 / gran_factor
        self.funnel_angle_arr = np.arange(0.0, 360.0, funnel_angle_gap)

        for angle in self.funnel_angle_arr:
            for size in funnel_size_array:
                tt = ToolType(
                    id,
                    'Funnel',
                    None,
                    angle * np.pi / 180,
                    size
                )
                self.funnel_tools.append(tt)
                id += 1

        '''
            Bucket tools
            l = 5, a = 24 => 120
        '''
        self.bucket_tools = []

        self.bucket_size_min = 3.0
        bucket_size_array = np.arange(self.bucket_size_min, 11.001, 2.)

        bucket_angle_gap = 15.0 / gran_factor
        self.bucket_angle_arr = np.arange(0.0, 360.0, bucket_angle_gap)

        for angle in self.bucket_angle_arr:
            for size in bucket_size_array:
                tt = ToolType(
                    id,
                    'Bucket',
                    None,
                    angle * np.pi / 180,
                    size
                )
                self.bucket_tools.append(tt)
                id += 1


        self.misc_tools = []
        tt = ToolType(
            id,
            'no_op',
            None,
            None,
            None
        )
        self.misc_tools.append(tt)
        id += 1

        # print('ramp_tools', len(self.ramp_tools))
        # print('trampoline_tools', len(self.trampoline_tools))
        # print('fixed_ball_tools', len(self.fixed_ball_tools))
        # print('bouncy_ball_tools', len(self.bouncy_ball_tools))
        # print('box_tools', len(self.box_tools))
        # print('bouncy_box_tools', len(self.bouncy_box_tools))
        # print('triangle_tools', len(self.triangle_tools))
        # print('bouncy_triangle_tools', len(self.bouncy_triangle_tools))
        # print('square_tools', len(self.square_tools))
        # print('bouncy_square_tools', len(self.bouncy_square_tools))
        # print('pentagon_tools', len(self.pentagon_tools))
        # print('bouncy_pentagon_tools', len(self.bouncy_pentagon_tools))
        # print('hexagon_tools', len(self.hexagon_tools))
        # print('bouncy_hexagon_tools', len(self.bouncy_hexagon_tools))
        # print('hinge_tools', len(self.hinge_tools))
        # print('hinge_constrained_tools', len(self.hinge_constrained_tools))
        # print('see_saw_tools', len(self.see_saw_tools))
        # print('cannon_tools', len(self.cannon_tools))
        # print('fan_tools', len(self.fan_tools))
        # print('belt_tools', len(self.belt_tools))
        # print('funnel_tools', len(self.funnel_tools))
        # print('bucket_tools', len(self.bucket_tools))
        # print('misc_tools', len(self.misc_tools))

        # 180 (Ramp) + 180(Trampoline) + 15 (Ball) + 15(Bouncy Circle) +
        # 20 (Box) + 20 (Bouncy Box) +
        # Triangle (168 + 168) + Square (105 + 105) + Pentagon (105 + 105) + Hexagon (84 + 84)
        # 55 (Hinge) + 90 (Hinge Constrained)
        # 160 (See Saw) + 168 (Cannon) + 168 (Fan) + 120 (Belt)
        # Funnel (120) + Bucket (120)
        # + 1 = 2536 - changes
        self.tools = self.ramp_tools + self.trampoline_tools + \
            self.fixed_ball_tools + self.bouncy_ball_tools + \
            self.box_tools + self.bouncy_box_tools + \
            self.triangle_tools + self.bouncy_triangle_tools + \
            self.square_tools + self.bouncy_square_tools + \
            self.pentagon_tools + self.bouncy_pentagon_tools + \
            self.hexagon_tools + self.bouncy_hexagon_tools + \
            self.hinge_tools + self.hinge_constrained_tools + self.see_saw_tools + \
            self.cannon_tools + self.fan_tools + \
            self.belt_tools + \
            self.funnel_tools + self.bucket_tools + \
            self.misc_tools

        # Get the one hot encodings
        #keep_types = ['Ramp', 'Trampoline']
        distinct_types = list(set([t.tool_type for t in self.tools]))

        vec_len = len(distinct_types)
        #vec_len = len(keep_types)
        type_to_vec = {}
        for i, t in enumerate(distinct_types):
            type_to_vec[t] = np.zeros(vec_len)
            #if t in keep_types:
            #    i = keep_types.index(t)
            #    type_to_vec[t][i] = 1.0
            type_to_vec[t][i] = 1.0

        gt_embs = []
        for t in self.tools:
            gt_emb = t.to_gt(type_to_vec[t.tool_type])
            gt_embs.append(gt_emb)

        # Normalize (excluding the one hot portion)
        gt_embs = np.array(gt_embs)
        param_embs = gt_embs[:, vec_len:]

        param_std = np.std(param_embs, axis=0)
        assert not (param_std == 0).any(), 'All parameters should at least one non-zero val'

        gt_embs[:, vec_len:] = (param_embs - np.mean(param_embs, axis=0)) / param_std
        self.gt_embs = gt_embs

        self.colors = {
                'Ramp': fixed_color,
                'Trampoline': bouncy_color,
                'Fixed_Ball': fixed_color,
                'Bouncy_Ball': bouncy_color,
                'Fixed_Box': fixed_color,
                'Bouncy_Box': bouncy_color,
                'Fixed_Triangle': fixed_color,
                'Bouncy_Triangle': bouncy_color,
                'Fixed_Square': fixed_color,
                'Bouncy_Square': bouncy_color,
                'Fixed_Pentagon': fixed_color,
                'Bouncy_Pentagon': bouncy_color,
                'Fixed_Hexagon': fixed_color,
                'Bouncy_Hexagon': bouncy_color,
                'Hinge': hinge_color,
                'Hinge_Constrained': hinge_color,
                'See_Saw': hinge_color,
                'Cannon': 'blue',
                'Fan': 'blue',
                'Belt': 'blue',
                'Funnel': 'blue',
                'Bucket': 'blue',
                }

    def create_tool(self, params, pos):
        tool_type = params.tool_type

        if params.tool_type == 'Ramp':
            obj = Ramp(
                    pos,
                    length=params.length,
                    angle=params.angle,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type],
                    friction=params.extra_info['friction']
                    )
        elif params.tool_type == 'Trampoline':
            obj = Trampoline(
                    pos,
                    length=params.length,
                    angle=params.angle,
                    elasticity=params.elasticity,
                    )
        elif params.tool_type == 'Fixed_Ball':
            obj = FixedCircle(
                    pos,
                    radius=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Bouncy_Ball':
            obj = BouncyCircle(
                    pos,
                    radius=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Fixed_Box':
            obj = FixedBox(pos,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Bouncy_Box':
            obj = BouncyBox(pos,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Fixed_Triangle':
            obj = FixedTriangle(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Bouncy_Triangle':
            obj = BouncyTriangle(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Fixed_Square':
            obj = FixedSquare(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Bouncy_Square':
            obj = BouncySquare(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Fixed_Pentagon':
            obj = FixedPentagon(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Bouncy_Pentagon':
            obj = BouncyPentagon(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Fixed_Hexagon':
            obj = FixedHexagon(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Bouncy_Hexagon':
            obj = BouncyHexagon(pos,
                    angle=params.angle,
                    size=params.length,
                    elasticity=params.elasticity,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Hinge':
            obj = HingeSeg(pos,
                    length=params.length,
                    color=self.colors[params.tool_type],
                    friction=params.extra_info['friction']
                    )
        elif params.tool_type == 'Hinge_Constrained':
            obj = HingeSlideSeg(pos,
                    max_angle=params.extra_info['max_angle'],
                    length=params.length,
                    color=self.colors[params.tool_type],
                    friction=params.extra_info['friction']
                    )
        elif params.tool_type == 'See_Saw':
            obj = SeeSaw(pos,
                    max_angle=params.extra_info['max_angle'],
                    length=params.length,
                    on_left=params.extra_info['on_left'],
                    ball_mass=params.extra_info['ball_mass'],
                    color=self.colors[params.tool_type],
                    friction=params.extra_info['friction']
                    )
        elif params.tool_type == 'Cannon':
            obj = Cannon(
                    pos,
                    angle=params.angle,
                    force=params.extra_info['force'],
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Fan':
            obj = Fan(
                    pos,
                    angle=params.angle,
                    force=params.extra_info['force'],
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Belt':
            obj = Belt(
                    pos,
                    vel=params.extra_info['vel'],
                    length=params.length,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Funnel':
            obj = Funnel(
                    pos,
                    angle=params.angle,
                    size=params.length,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'Bucket':
            obj = Bucket(
                    pos,
                    angle=params.angle,
                    size=params.length,
                    color=self.colors[params.tool_type]
                    )
        elif params.tool_type == 'no_op':
            obj = NoOp()

        return obj


    def get_tool(self, tool_id, pos, settings):
        pos = convert_action(pos, settings)
        params = self.tools[tool_id]
        obj = self.create_tool(params, pos)
        obj.set_settings(settings)
        return obj


    def get_option_properties(args):
        for id in range(len(args.action_bank)):
            tool_type = self.tools[id].tool_type


    def get_train_test_split(self, args):
        train_tools = []
        test_tools = []
        split_type = args.split_type
        deterministic_split = args.deterministic_split


        if split_type is None:
            self_len = len(self.tools)

            rnd_order = np.arange(self_len)

            random.shuffle(rnd_order)

            mid_ind = int(self_len * 0.5)

            train_tools = rnd_order[:mid_ind]
            test_tools = rnd_order[mid_ind:]

        elif split_type == 'all_tools':
            train_tools = np.arange(len(self.tools))
            test_tools = np.arange(len(self.tools))


        elif args.split_type == 'selective_split':
            base_factor = args.custom_split_factor
            angle_gap = 15.0 * base_factor
            vel_gap = 1.0 * base_factor
            radius_gap = self.radius_gap_factor * base_factor
            size_gap = self.box_size_factor * base_factor
            hinge_length_gap = self.hinge_length_factor * base_factor
            hinge_angle_gap = self.hinge_angle_factor * base_factor
            see_saw_angle_gap = self.see_saw_angle_factor * base_factor

            angle_minn = 0.
            vel_minn = 1.0
            radius_minn = self.radius_min
            size_minn = self.box_size_min
            hinge_length_minn = self.hinge_length_min
            hinge_angle_minn = self.hinge_angle_min
            see_saw_angle_minn = self.see_saw_angle_min

            seg_angle_all = get_all(self.seg_angle_arr, angle_gap, angle_minn)

            ball_radius_all = get_all(self.ball_radius_arr, radius_gap, radius_minn)
            box_size_all = get_all(self.box_size_arr, size_gap, size_minn)

            triangle_angle_all = get_all(self.triangle_angle_arr, angle_gap, angle_minn)
            square_angle_all = get_all(self.square_angle_arr, angle_gap, angle_minn)
            pentagon_angle_all = get_all(self.pentagon_angle_arr, angle_gap, angle_minn)
            hexagon_angle_all = get_all(self.hexagon_angle_arr, angle_gap, angle_minn)

            cannon_angle_all = get_all(self.cannon_angle_arr, angle_gap, angle_minn)
            fan_angle_all = get_all(self.fan_angle_arr, angle_gap, angle_minn)

            hinge_length_all = get_all(self.hinge_length_arr, hinge_length_gap, hinge_length_minn)
            hinge_angle_all = get_all(self.hinge_angle_arr, hinge_angle_gap, hinge_angle_minn)
            see_saw_angle_all = get_all(self.see_saw_angle_arr, see_saw_angle_gap, see_saw_angle_minn)

            belt_vel_all = get_all(self.belt_vel_arr, vel_gap, vel_minn)

            funnel_angle_all = get_all(self.funnel_angle_arr, angle_gap, angle_minn)
            bucket_angle_all = get_all(self.bucket_angle_arr, angle_gap, angle_minn)

            seg_angle = [0.0, 45.0, 90.0, 135.0]
            ramp_elasticity = [0.3]
            trampoline_elasticity = [1.2]
            seg_length = [10., 18.]

            full_angle = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
            cannon_force = [120.0]

            see_saw_length = [12.0, 20.0]
            see_saw_angle = [20.0, 50.0]
            see_saw_ball_mass = [10.0]
            hinge_constrained_friction = [1.0]

            belt_vel = [-6.0, -3.0, 3.0, 6.0]
            belt_length = [12.0, 24.0]

            bucket_size = [7.]

            triangle_angle = [0.0, 30.0, 60.0, 90.0]
            triangle_size = [3.0, 6.0, 9.0]

            fan_force = [60.0, 120.0]
            hinge_friction = [1., 1.05]

            funnel_size = [5., 9.]

            train_names = {
                    'Ramp': {'angle': seg_angle, 'elasticity': ramp_elasticity,
                        'length': seg_length},  # 4 * 2 = 8
                    'Trampoline': {'angle': seg_angle, 'elasticity': trampoline_elasticity,
                        'length': seg_length},  # 4 * 2 = 8
                    'Fixed_Ball': {'elasticity': ramp_elasticity},   # 4
                    'Bouncy_Ball': {'elasticity': trampoline_elasticity},    # 4
                    'Cannon': {'angle': full_angle, 'force': cannon_force},    # 8 * 1 = 8
                    'See_Saw': {'max_angle': see_saw_angle, 'length': see_saw_length,
                        'ball_mass': see_saw_ball_mass}, # 8
                    'Hinge_Constrained': {'max_angle': see_saw_angle, 'length': see_saw_length,
                        'friction': hinge_constrained_friction},    # 4 (zero mass see-saw)
                    'Belt': {'vel': belt_vel, 'length': belt_length},   # 8
                    'Bucket': {'angle': full_angle, 'length': bucket_size},    # 8
                    'no_op': {},
                    }
            test_names = {
                    'Fixed_Triangle': {'angle': triangle_angle, 'elasticity': ramp_elasticity,
                        'length': triangle_size},  # 4 * 3 = 12
                    'Bouncy_Triangle': {'angle': triangle_angle, 'elasticity': trampoline_elasticity,
                        'length': triangle_size},  # 4 * 3 = 12
                    'Fan': {'angle': full_angle, 'force': fan_force},    # 16
                    'Hinge': {'friction': hinge_friction},  # 10
                    'Funnel': {'angle': full_angle, 'length': funnel_size},    # 8 * 2
                    'no_op': {},
                    }

            train_tools, test_tools = get_tools_from_filters(train_names, test_names,
                self.tools)


        elif args.split_type == 'selective_split_superset':
            train_names = {
                    'Ramp': {},
                    'Trampoline': {},
                    'Fixed_Ball': {},
                    'Bouncy_Ball': {},
                    'Cannon': {},
                    'See_Saw': {},
                    'Hinge_Constrained': {},
                    'Belt': {},
                    'Bucket': {},
                    'no_op': {},
                    }
            test_names = {
                    'Fixed_Triangle': {},
                    'Bouncy_Triangle': {},
                    'Fan': {},
                    'Hinge': {},
                    'Funnel': {},
                    'no_op': {},
                    }

            train_tools, test_tools = get_tools_from_filters(train_names, test_names,
                self.tools)


        elif split_type == 'tool_types_full':
            angle_gap = 15.0
            vel_gap = 1.0
            radius_gap = self.radius_gap_factor
            size_gap = self.box_size_factor
            hinge_length_gap = self.hinge_length_factor
            hinge_angle_gap = self.hinge_angle_factor
            see_saw_angle_gap = self.see_saw_angle_factor

            angle_minn = 0.
            vel_minn = 1.0
            radius_minn = self.radius_min
            size_minn = self.box_size_min
            hinge_length_minn = self.hinge_length_min
            hinge_angle_minn = self.hinge_angle_min
            see_saw_angle_minn = self.see_saw_angle_min

            seg_angle_all = get_all(self.seg_angle_arr, angle_gap, angle_minn)

            ball_radius_all = get_all(self.ball_radius_arr, radius_gap, radius_minn)
            box_size_all = get_all(self.box_size_arr, size_gap, size_minn)

            triangle_angle_all = get_all(self.triangle_angle_arr, angle_gap, angle_minn)
            square_angle_all = get_all(self.square_angle_arr, angle_gap, angle_minn)
            pentagon_angle_all = get_all(self.pentagon_angle_arr, angle_gap, angle_minn)
            hexagon_angle_all = get_all(self.hexagon_angle_arr, angle_gap, angle_minn)

            cannon_angle_all = get_all(self.cannon_angle_arr, angle_gap, angle_minn)
            fan_angle_all = get_all(self.fan_angle_arr, angle_gap, angle_minn)

            hinge_length_all = get_all(self.hinge_length_arr, hinge_length_gap, hinge_length_minn)
            hinge_angle_all = get_all(self.hinge_angle_arr, hinge_angle_gap, hinge_angle_minn)
            see_saw_angle_all = get_all(self.see_saw_angle_arr, see_saw_angle_gap, see_saw_angle_minn)

            belt_vel_all = get_all(self.belt_vel_arr, vel_gap, vel_minn)

            funnel_angle_all = get_all(self.funnel_angle_arr, angle_gap, angle_minn)
            bucket_angle_all = get_all(self.bucket_angle_arr, angle_gap, angle_minn)

            train_tools_filter = {
                    'Trampoline': {'angle': seg_angle_all},
                    'Ramp': {'angle': seg_angle_all},

                    'Fixed_Ball': {'length': ball_radius_all},
                    'Bouncy_Ball': {'length': ball_radius_all},

                    'Fixed_Triangle': {'angle': triangle_angle_all},
                    'Bouncy_Triangle': {'angle': triangle_angle_all},

                    'Fixed_Pentagon': {'angle': pentagon_angle_all},
                    'Bouncy_Pentagon': {'angle': pentagon_angle_all},

                    'Hinge': {'length': hinge_length_all},
                    'Hinge_Constrained': {'max_angle': hinge_angle_all},

                    'Fan': {'angle': fan_angle_all},
                    'Bucket': {'angle': bucket_angle_all},
                    'no_op': {},
                    }
            test_tools_filter = {
                    'Fixed_Box': {'length': box_size_all},
                    'Bouncy_Box': {'length': box_size_all},

                    'Fixed_Square': {'angle': square_angle_all},
                    'Bouncy_Square': {'angle': square_angle_all},

                    'Fixed_Hexagon': {'angle': hexagon_angle_all},
                    'Bouncy_Hexagon': {'angle': hexagon_angle_all},

                    'See_Saw': {'max_angle': see_saw_angle_all},

                    'Cannon': {'angle': cannon_angle_all},

                    'Belt': {'vel': belt_vel_all},
                    'Funnel': {'angle': funnel_angle_all},
                    'no_op': {},

                    }
            train_tools, test_tools = get_tools_from_filters(train_tools_filter,
                    test_tools_filter, self.tools)

        elif split_type == 'full_clean':
            train_names = {
                    'Ramp': {},
                    'Trampoline': {},
                    'Fixed_Ball': {},
                    'Bouncy_Ball': {},
                    'See_Saw': {},
                    'Hinge_Constrained': {},
                    'Cannon': {},
                    'Bucket': {},
                    'no_op': {},
                    }
            test_names = {
                    'Fixed_Triangle': {},
                    'Bouncy_Triangle': {},
                    'Hinge': {},
                    'Fan': {},
                    'Belt': {},
                    'Funnel': {},
                    'no_op': {},
                    }

            train_tools, test_tools = get_tools_from_filters(train_names, test_names,
                self.tools)

        elif split_type == 'all_clean':
            train_names = {
                    'Ramp': {},
                    'Trampoline': {},
                    'Fixed_Ball': {},
                    'Bouncy_Ball': {},
                    'See_Saw': {},
                    'Hinge_Constrained': {},
                    'Cannon': {},
                    'Bucket': {},
                    'Fixed_Triangle': {},
                    'Bouncy_Triangle': {},
                    'Hinge': {},
                    'Fan': {},
                    'Belt': {},
                    'Funnel': {},
                    'no_op': {},
                    }
            test_names = {
                    'Ramp': {},
                    'Trampoline': {},
                    'Fixed_Ball': {},
                    'Bouncy_Ball': {},
                    'See_Saw': {},
                    'Hinge_Constrained': {},
                    'Cannon': {},
                    'Bucket': {},
                    'Fixed_Triangle': {},
                    'Bouncy_Triangle': {},
                    'Hinge': {},
                    'Fan': {},
                    'Belt': {},
                    'Funnel': {},
                    'no_op': {},
                    }

            train_tools, test_tools = get_tools_from_filters(train_names, test_names,
                self.tools)

        elif split_type == 'tool_types_partial':
            angle_gap = 15.0
            vel_gap = 1.0
            radius_gap = self.radius_gap_factor
            size_gap = self.box_size_factor
            hinge_length_gap = self.hinge_length_factor
            hinge_angle_gap = self.hinge_angle_factor
            see_saw_angle_gap = self.see_saw_angle_factor

            angle_minn = 0.
            vel_minn = 1.0
            radius_minn = self.radius_min
            size_minn = self.box_size_min
            hinge_length_minn = self.hinge_length_min
            hinge_angle_minn = self.hinge_angle_min
            see_saw_angle_minn = self.see_saw_angle_min

            def get_train_test(arr, gap, minn, randomize=not deterministic_split,
                no_split=False):
                use_arr = [x for x in arr if does_div(x, gap, minn)]
                if no_split:
                    return use_arr
                elif randomize:
                    random.shuffle(use_arr)
                    mid_ind = (len(use_arr) + 1) // 2
                    train_vals = use_arr[:mid_ind]
                    test_vals = use_arr[mid_ind:]
                else:
                    train_vals = []
                    test_vals = []
                    # Alternate starting from the minimum value
                    for ind, x in enumerate(use_arr):
                        if ind % 2 == 0:
                            train_vals.append(x)
                        else:
                            test_vals.append(x)
                return train_vals, test_vals

            seg_angle_train, seg_angle_test = get_train_test(self.seg_angle_arr,
                    angle_gap, angle_minn)

            ball_radius_all = get_train_test(self.ball_radius_arr,
                    radius_gap, radius_minn, no_split=True)
            box_size_all = get_train_test(self.box_size_arr,
                    size_gap, size_minn, no_split=True)

            triangle_angle_train, triangle_angle_test = get_train_test(
                    self.triangle_angle_arr,
                    angle_gap, angle_minn)
            square_angle_train, square_angle_test = get_train_test(
                    self.square_angle_arr,
                    angle_gap, angle_minn)
            pentagon_angle_train, pentagon_angle_test = get_train_test(
                    self.pentagon_angle_arr,
                    angle_gap, angle_minn)
            hexagon_angle_train, hexagon_angle_test = get_train_test(
                    self.hexagon_angle_arr,
                    angle_gap, angle_minn)

            cannon_angle_train, cannon_angle_test = get_train_test(
                    self.cannon_angle_arr,
                    angle_gap, angle_minn)
            fan_angle_train, fan_angle_test = get_train_test(
                    self.fan_angle_arr,
                    angle_gap, angle_minn)

            hinge_length_all = get_train_test(
                    self.hinge_length_arr,
                    hinge_length_gap, hinge_length_minn,
                    no_split=True)
            hinge_angle_all = get_train_test(
                    self.hinge_angle_arr,
                    hinge_angle_gap, hinge_angle_minn,
                    no_split=True)
            see_saw_angle_all = get_train_test(
                    self.see_saw_angle_arr,
                    see_saw_angle_gap, see_saw_angle_minn,
                    no_split=True)

            belt_vel_train, belt_vel_test = get_train_test(self.belt_vel_arr,
                    vel_gap, vel_minn)

            funnel_angle_train, funnel_angle_test = get_train_test(
                    self.funnel_angle_arr,
                    angle_gap, angle_minn)
            bucket_angle_train, bucket_angle_test = get_train_test(
                    self.bucket_angle_arr,
                    angle_gap, angle_minn)

            train_tools_filter = {
                    'Trampoline': {'angle': seg_angle_train},
                    'Ramp': {'angle': seg_angle_train},

                    'Fixed_Ball': {'length': ball_radius_all},
                    'Bouncy_Ball': {'length': ball_radius_all},

                    'Fixed_Triangle': {'angle': triangle_angle_train},
                    'Bouncy_Triangle': {'angle': triangle_angle_train},

                    'Fixed_Square': {'angle': square_angle_train},
                    'Bouncy_Square': {'angle': square_angle_train},

                    'Fixed_Pentagon': {'angle': pentagon_angle_train},
                    'Bouncy_Pentagon': {'angle': pentagon_angle_train},

                    'Fixed_Hexagon': {'angle': hexagon_angle_train},
                    'Bouncy_Hexagon': {'angle': hexagon_angle_train},

                    'Hinge': {'length': hinge_length_all},
                    'Hinge_Constrained': {'max_angle': hinge_angle_all},

                    'Cannon': {'angle': cannon_angle_train},
                    'Fan': {'angle': fan_angle_train},

                    'Belt': {'vel': belt_vel_train},

                    'Funnel': {'angle': funnel_angle_train},
                    'Bucket': {'angle': bucket_angle_train},

                    'no_op': {},
                    }


            test_tools_filter = {
                    'Trampoline': {'angle': seg_angle_test},
                    'Ramp': {'angle': seg_angle_test},

                    'Fixed_Box': {'length': box_size_all},
                    'Bouncy_Box': {'length': box_size_all},

                    'Fixed_Triangle': {'angle': triangle_angle_test},
                    'Bouncy_Triangle': {'angle': triangle_angle_test},

                    'Fixed_Square': {'angle': square_angle_test},
                    'Bouncy_Square': {'angle': square_angle_test},

                    'Fixed_Pentagon': {'angle': pentagon_angle_test},
                    'Bouncy_Pentagon': {'angle': pentagon_angle_test},

                    'Fixed_Hexagon': {'angle': hexagon_angle_test},
                    'Bouncy_Hexagon': {'angle': hexagon_angle_test},

                    'See_Saw': {'max_angle': see_saw_angle_all},

                    'Cannon': {'angle': cannon_angle_test},
                    'Fan': {'angle': fan_angle_test},

                    'Belt': {'vel': belt_vel_test},

                    'Funnel': {'angle': funnel_angle_test},
                    'Bucket': {'angle': bucket_angle_test},

                    'no_op': {},
                    }
            train_tools, test_tools = get_tools_from_filters(train_tools_filter,
                    test_tools_filter, self.tools)
            print('Train, Test Num: ', len(train_tools), len(test_tools))

        elif split_type == 'gran_1':
            train_tools, test_tools = self.filter_gran_factor(1.0)
        elif split_type == 'gran_2':
            train_tools, test_tools = self.filter_gran_factor(1.5)
        elif split_type == 'gran_3':
            train_tools, test_tools = self.filter_gran_factor(3.0)

        elif split_type == 'all_gran':
            train_names = {
                    'Ramp': {},
                    'Trampoline': {},
                    'See_Saw': {},
                    'Hinge_Constrained': {},
                    'Cannon': {},
                    'Bucket': {},
                    'Fixed_Triangle': {},
                    'Bouncy_Triangle': {},
                    'Hinge': {},
                    'Fan': {},
                    'Funnel': {},
                    'no_op': {},
                    }
            test_names = train_names
            train_tools, test_tools = get_tools_from_filters(train_names, test_names,
                self.tools)
            train_tools = self.sub_filter_gran_factor(sub_gran_factor=5.0, tool_ids=train_tools)
            test_tools = train_tools


        elif 'analysis' in split_type:
            all_names = {
                    'Ramp': {},
                    'Trampoline': {},
                    'See_Saw': {},
                    'Hinge_Constrained': {},
                    'Cannon': {},
                    'Bucket': {},
                    'Fixed_Triangle': {},
                    'Bouncy_Triangle': {},
                    'Hinge': {},
                    'Fan': {},
                    'Funnel': {},
                    'no_op': {},
                    }
            test_names = all_names
            all_tools, test_tools = get_tools_from_filters(all_names, test_names,
                self.tools)
            all_tools = self.sub_filter_gran_factor(sub_gran_factor=5.0, tool_ids=all_tools)

            def extract_30(tool_set):
                select = []
                angle_minn = 0.0
                hinge_angle_minn = self.hinge_angle_min
                for tool_id in tool_set:
                    tool = self.tools[tool_id]
                    if tool.tool_type in ['Hinge'] or tool.tool_type in ['Hinge_Constrained'] or tool.tool_type in ['See_Saw']:
                        comp_angle = tool.extra_info['max_angle'] * 180.0 / np.pi
                        if does_div(comp_angle, 30., hinge_angle_minn):
                            select.append(tool_id)
                    elif tool.tool_type == 'no_op':
                        select.append(tool_id)
                    else:
                        comp_angle = tool.angle * 180.0 / np.pi
                        if does_div(comp_angle, 30., angle_minn):
                            select.append(tool_id)
                return np.array(select)

            train_tools = extract_30(all_tools)

            # Test tools should not be directly used in this setup. Instead
            # custom experimentation should be performed.
            test_tools = train_tools

        elif split_type == 'random_params':
            '''
                For each tool type, randomly split each configuration into train/test
            '''
            for tool in self.tools:
                if (tool.tool_type in train_tools_filter) and (tool.tool_type in test_tools_filter):
                    # Randomly split the tools here.
                    if np.random.random() < 0.5:
                        train_tools.append(tool)
                    else:
                        test_tools.append(tool)
                elif tool.tool_type in train_tools_filter:
                    train_tools.append(tool)
                elif tool.tool_type in test_tools_filter:
                    test_tools.append(tool)
        else:
            raise ValueError('Must be a valid split type')

        # check that there are no overlaps between the two sets
        if split_type != 'all_tools' and split_type != 'all_clean' and split_type != 'all_gran' and 'analysis' not in split_type:
            for train_tool in train_tools:
                if self.tools[train_tool].tool_type != 'no_op' and train_tool in test_tools:
                    raise ValueError('There is the same tool in both train and test set ' + str(train_tool))

        random.shuffle(train_tools)
        random.shuffle(test_tools)

        print('# train %i' % len(train_tools))
        print('# test %i' % len(test_tools))

        return np.array(train_tools), np.array(test_tools)


    def filter_gran_factor(self, gran_factor):
        angle_gap = 15.0 / gran_factor
        hinge_angle_gap = self.hinge_angle_factor / gran_factor

        angle_minn = 0.0
        hinge_angle_minn = self.hinge_angle_min

        inc_groups = defaultdict(list)
        for tool in self.tools:
            if tool.tool_type in ['Ramp', 'Trampoline']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_seg'].append(tool)
            elif tool.tool_type in ['Cannon']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_cannon'].append(tool)
            elif tool.tool_type in ['Fan']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_fan'].append(tool)
            elif tool.tool_type in ['Funnel']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_funnel'].append(tool)
            elif tool.tool_type in ['Bucket']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_bucket'].append(tool)
            elif tool.tool_type in ['Hinge_Constrained']:
                comp_angle = tool.extra_info['max_angle'] * 180.0 / np.pi
                if does_div(comp_angle, hinge_angle_gap, hinge_angle_minn):
                    inc_groups['max_angle_hinge'].append(tool)
            elif tool.tool_type in ['See_Saw']:
                comp_angle = tool.extra_info['max_angle'] * 180.0 / np.pi
                if does_div(comp_angle, hinge_angle_gap, hinge_angle_minn):
                    inc_groups['max_angle_seesaw'].append(tool)
            elif tool.tool_type in ['Fixed_Triangle', 'Bouncy_Triangle']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_triangle'].append(tool)
            elif tool.tool_type in ['Fixed_Square', 'Bouncy_Square']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_square'].append(tool)
            elif tool.tool_type in ['Fixed_Pentagon', 'Bouncy_Pentagon']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_pentagon'].append(tool)
            elif tool.tool_type in ['Fixed_Hexagon', 'Bouncy_Hexagon']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_hexagon'].append(tool)
            elif tool.tool_type == 'no_op':
                inc_groups['misc'].append(tool)
            else:
                # Ignore
                pass

        counts = [(g, len(t)) for g, t in inc_groups.items()]
        total = sum([x[1] for x in counts])
        for n, count in counts:
            print('%s: %i' % (n, count))
        print('Total # tools', total)

        train_tools = []
        test_tools = []
        for group_name, group_tools in sorted(inc_groups.items()):
            if group_name == 'misc':
                group_tool_ids = [g.tool_id for g in group_tools]
                train_tools.extend(group_tool_ids)
                test_tools.extend(group_tool_ids)
                continue
            else:
                prop_name = '_'.join(group_name.split('_')[:-1])
            tool_vals = [get_tool_prop(tool, prop_name) for tool in group_tools]
            # Only get the unique tool values
            unq_tool_vals = list(set(tool_vals))

            random.shuffle(unq_tool_vals)
            mid_ind = (len(unq_tool_vals) + 1) // 2
            # Randomly split each group.
            train_vals = unq_tool_vals[:mid_ind]
            test_vals = unq_tool_vals[mid_ind:]
            for i, tool_val in enumerate(tool_vals):
                if any([abs(tool_val - y) < 1e-5 for y in train_vals]):
                    train_tools.append(group_tools[i].tool_id)
                elif any([abs(tool_val - y) < 1e-5 for y in test_vals]):
                    test_tools.append(group_tools[i].tool_id)
                else:
                    raise ValueError('Tool val does not exist in the tool group')

        print('# train tools', len(train_tools))
        print('# test tools', len(test_tools))
        return train_tools, test_tools



    def sub_filter_gran_factor(self, sub_gran_factor, tool_ids):
        angle_gap = 15.0 / sub_gran_factor
        hinge_angle_gap = self.hinge_angle_factor / sub_gran_factor

        angle_minn = 0.0
        hinge_angle_minn = self.hinge_angle_min

        inc_groups = defaultdict(list)
        for tool_id in tool_ids:
            tool = self.tools[tool_id]
            if tool.tool_type in ['Ramp', 'Trampoline']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_seg'].append(tool)
            elif tool.tool_type in ['Cannon']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_cannon'].append(tool)
            elif tool.tool_type in ['Fan']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_fan'].append(tool)
            elif tool.tool_type in ['Funnel']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_funnel'].append(tool)
            elif tool.tool_type in ['Bucket']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_bucket'].append(tool)
            elif tool.tool_type in ['Hinge_Constrained']:
                comp_angle = tool.extra_info['max_angle'] * 180.0 / np.pi
                if does_div(comp_angle, hinge_angle_gap, hinge_angle_minn):
                    inc_groups['max_angle_hinge'].append(tool)
            elif tool.tool_type in ['See_Saw']:
                comp_angle = tool.extra_info['max_angle'] * 180.0 / np.pi
                if does_div(comp_angle, hinge_angle_gap, hinge_angle_minn):
                    inc_groups['max_angle_seesaw'].append(tool)
            elif tool.tool_type in ['Fixed_Triangle', 'Bouncy_Triangle']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_triangle'].append(tool)
            elif tool.tool_type in ['Fixed_Square', 'Bouncy_Square']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_square'].append(tool)
            elif tool.tool_type in ['Fixed_Pentagon', 'Bouncy_Pentagon']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_pentagon'].append(tool)
            elif tool.tool_type in ['Fixed_Hexagon', 'Bouncy_Hexagon']:
                comp_angle = tool.angle * 180.0 / np.pi
                if does_div(comp_angle, angle_gap, angle_minn):
                    inc_groups['angle_hexagon'].append(tool)
            elif tool.tool_type == 'no_op':
                inc_groups['misc'].append(tool)
            else:
                # Ignore
                pass

        counts = [(g, len(t)) for g, t in inc_groups.items()]
        total = sum([x[1] for x in counts])
        for n, count in counts:
            print('%s: %i' % (n, count))
        print('Total # tools', total)

        train_tools = []
        for group_name, group_tools in sorted(inc_groups.items()):
            if group_name == 'misc':
                group_tool_ids = [g.tool_id for g in group_tools]
                train_tools.extend(group_tool_ids)
                continue
            else:
                prop_name = '_'.join(group_name.split('_')[:-1])
            tool_vals = [get_tool_prop(tool, prop_name) for tool in group_tools]
            # Only get the unique tool values
            unq_tool_vals = list(set(tool_vals))
            for i, tool_val in enumerate(tool_vals):
                if any([abs(tool_val - y) < 1e-5 for y in unq_tool_vals]):
                    train_tools.append(group_tools[i].tool_id)
                else:
                    raise ValueError('Tool val does not exist in the tool group')

        print('# train tools', len(train_tools))
        return train_tools
