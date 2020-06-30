import numpy as np
import copy
import torch
import math
import torch.optim as optim
from gym.spaces import Discrete, Box

from rlf.rl import utils
from rlf.rl.distributions import GeneralMixedDist
from rlf import BasePolicy
from method.models.main_method import extract_aval, init_extract_aval



class DistanceMethod(BasePolicy):
    def __init__(self, args, obs_space, action_space):
        self.args = args
        self.emb_mem = args.emb_mem
        self.dist_mem = args.dist_mem

        super().__init__(args, obs_space, action_space)

    def _get_dist(self, hidden_state_dim):
        if self.action_space.__class__.__name__ == "Discrete":
            return self._get_cont_policy(self.args.o_dim, use_gaussian=self.args.use_gaussian_distance)

        elif self.action_space.__class__.__name__ == "Box":
            return self._get_cont_policy(self.action_space.shape[0])

        elif self.action_space.__class__.__name__ == 'Dict' or self.action_space.__class__.__name__ == 'SampleDict':

            def get_dist(key, space):
                if isinstance(space, Discrete) and key == 'skip':
                    return (self._get_disc_policy(space.n), 1)
                elif isinstance(space, Discrete) and key == 'index':
                    return (self._get_cont_policy(self.args.o_dim, use_gaussian=self.args.use_gaussian_distance), self.args.o_dim)
                elif isinstance(space, Box):
                    return (self._get_cont_policy(space.shape[0]), space.shape[0])
                else:
                    raise ValueError('Unrecognized space')

            dist_set = [get_dist(key, space) for (key, space) in zip(self.action_space.spaces.keys(), self.action_space.spaces.values())]

            self.pos_idx = list(self.action_space.spaces.keys()).index('pos')
            self.index_idx = list(self.action_space.spaces.keys()).index('index')

            self.parts = [x[0] for x in dist_set]
            self.action_sizes = [x[1] for x in dist_set]

            return GeneralMixedDist(
                    self.parts,
                    self.action_sizes,
                    self.pos_idx,
                    self.args)
        else:
            raise NotImplemented('Unrecognized environment action space')


    def get_dim_add_input(self):
        return self.args.aval_actions.shape[-1]

    def get_add_input(self, extra, infos):
        return extract_aval(extra, infos)

    def get_init_add_input(self, args, evaluate=False):
        return init_extract_aval(self.args, evaluate=evaluate)

    def get_action(self, state, add_input, recurrent_hidden_state,
                   mask, args, network=None, num_steps=None):
        # Sample actions
        with torch.no_grad():
            parts = self.actor_critic.act(state, recurrent_hidden_state, mask,
                                     add_input=add_input,
                                     deterministic=self.args.deterministic_policy)
            value, full_action, action_log_prob, rnn_hxs, act_extra = parts

            if self.action_space.__class__.__name__ == "Discrete":
                o = full_action[:,:]
            elif self.action_space.__class__.__name__ == 'Dict' or self.action_space.__class__.__name__ == 'SampleDict':
                start_idx = sum(self.action_sizes[:self.index_idx])
                final_idx = start_idx + self.args.o_dim
                o = full_action[:, start_idx:final_idx]

            take_action, reward_effect, extra = self.get_discrete_action(o,
                    network, state, num_steps, add_input)


            if self.action_space.__class__.__name__ == 'Dict' or self.action_space.__class__.__name__ == 'SampleDict':
                if full_action.is_cuda:
                    take_action = take_action.cuda()
                if 'skip' in self.action_space.spaces.keys():
                    self.skip_idx = list(self.action_space.spaces.keys()).index('skip')
                    skip_start_idx = sum(self.action_sizes[:self.skip_idx])
                    skip_final_idx = skip_start_idx + self.action_sizes[self.skip_idx]

                    skip_action = full_action[:, skip_start_idx:skip_final_idx]
                    take_action = torch.cat([take_action.float(), skip_action], dim=-1)

                pos_start_idx = sum(self.action_sizes[:self.pos_idx])
                pos_final_idx = pos_start_idx + self.action_sizes[self.pos_idx]
                pos = full_action[:, pos_start_idx:pos_final_idx]

                if len(take_action.shape) < len(pos.shape):
                    take_action = take_action.unsqueeze(-1)
                take_action = torch.cat([take_action.float(), pos], dim=1)

            ac_outs = (value, full_action, action_log_prob,
                       recurrent_hidden_state)
            q_outs = (take_action, reward_effect, extra)
            return ac_outs, q_outs

    def get_discrete_action(self, z, network, state=None, num_steps=None,
                            aval_actions=None):
        sel_ind = self.emb_mem.sample_action(z.float(), 1,
                sample=self.args.distance_sample,
                aval_actions=aval_actions.long())

        return sel_ind, 0.0, {}
