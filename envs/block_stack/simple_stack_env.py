from envs.block_stack.stack_env import StackEnv
import gym.spaces as spaces
import numpy as np


NOISE = 0.02

class SimpleStackEnv(StackEnv):
    def __init__(self):
        super().__init__()

    def step(self, a):
        noise = np.random.normal(0, NOISE, (2,))
        mod_action = np.array([a, *noise])
        return super().step(mod_action)


    def _set_action_space(self, sub_split):
        super()._set_action_space(sub_split)

        self.action_space = spaces.Discrete(len(self.aval_idx))
