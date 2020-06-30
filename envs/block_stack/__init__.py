from gym.envs.registration import register
from rlf import register_env_interface
from envs.block_stack.stack_interface import StackInterface, StackPlayInterface

register_env_interface('^StackEnv-v0$', StackInterface)
register_env_interface('^StackEnvNew-v0$', StackInterface)
register_env_interface('^StackEnvSimplest(.*)?$', StackInterface)
register_env_interface('^StackCompressedEnv-v0$', StackInterface)
register_env_interface('^StackLoadEnv-v0$', StackInterface)
register_env_interface('^StackStartEnv-v0$', StackInterface)
register_env_interface('^BlockPlayImg-v0$', StackPlayInterface)

register(
    id='StackEnv-v0',
    entry_point='envs.block_stack.stack_env:StackEnv',
)

register(
        id='StackEnvNew-v0',
        entry_point='envs.block_stack.stack_env_new:StackEnvNew',
        )

register(
        id='StackEnvSimplest-v0',
        entry_point='envs.block_stack.stack_env_simplest:StackEnvSimplest',
        )
register(
        id='StackEnvSimplestAll-v0',
        entry_point='envs.block_stack.stack_env_simplest_all:StackEnvAll',
        )
register(
        id='StackEnvSimplestA-v0',
        entry_point='envs.block_stack.stack_tasks:TaskAShapes',
        )

register(
        id='StackEnvSimplestMoving-v0',
        entry_point='envs.block_stack.stack_tasks:MovingShapes',
        )

register(
        id='StackEnvSimplestMovingAll-v0',
        entry_point='envs.block_stack.stack_tasks:MovingShapesAll',
        )

register(
        id='StackEnvSimplestR-v0',
        entry_point='envs.block_stack.stack_tasks:RndShapes',
        )

register(
    id='StackCompressedEnv-v0',
    entry_point='envs.block_stack.stack_env:StackCompressedEnv',
)

register(
    id='StackLoadEnv-v0',
    entry_point='envs.block_stack.stack_env:StackLoadEnv',
)

register(
    id='StackStartEnv-v0',
    entry_point='envs.block_stack.stack_env:StackStartEnv',
)

register(
    id='SimpleStackEnv-v0',
    entry_point='envs.block_stack.simple_stack_env:SimpleStackEnv',
)

register(
    id='BlockPlayImg-v0',
    entry_point='envs.block_stack.block_play:BlockPlayImg',
)



