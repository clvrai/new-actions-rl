# Constants

import os.path as osp

# Ramp, circle, square, triangle, pentagon, hexagon, levers, see-saw
fixed_color = '0xfa8535'
bouncy_color = '0x086788'
hinge_color = '0xfa9149'

# Special Tool colors: trampoline, cannon, fan, belt, funnel, bucket (Just load in images)
special_tool_color = None

# marker_color = '0xb4f7bc'
marker_color = '0x79addc'

# target_color = '0xffc09f'
target_color = '0xf45b69'

subgoal_color = '0xd3feb9'

# goalobj, goalball, goalbasket
# goal_color = '0x79addc'
goal_color = '0x7cfd2e'

# brown-black
# sensor_wall_color = '0xf2f7fb'
sensor_wall_color = '0xffffff'

# placed_wall_color = '0x472d30'
placed_wall_color = 'LIGHTSTEELBLUE'

solid_frame_color = None
blank_frame_color = None

asset_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'assets')
