import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import Box, Discrete, Dict
from rlf.rl.utils import init

import method.utils as meth_utils

import torch.distributions as D

from rlf import BasePolicy
from rlf.rl.distributions import FixedCategorical, Categorical


EPS = 1e-6

class OrderInvariantCategorical(nn.Module):
    def __init__(self, num_inputs, dist_mem, args):
        super().__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        hidden_dim = args.dist_hidden_dim
        self.args = args
        action_dim = args.o_dim if args.use_option_embs else args.z_dim
        self.dist_mem = dist_mem

        if args.dist_linear_action:
            self.action_linear = init_(nn.Linear(action_dim, hidden_dim))
        else:
            self.action_linear = nn.Sequential(
                init_(nn.Linear(action_dim, hidden_dim)), nn.ReLU(),
                init_(nn.Linear(hidden_dim, hidden_dim)))

        if args.dist_non_linear_final:
            self.linear = nn.Sequential(
                init_(nn.Linear(hidden_dim + num_inputs, hidden_dim)), nn.ReLU(),
                init_(nn.Linear(hidden_dim, 1)))
        else:
            self.linear = init_(nn.Linear(hidden_dim + num_inputs, 1))

        # self.linear = init_(nn.Linear(hidden_dim + num_inputs, 1))

    def forward(self, x, add_input):
        aval_actions = add_input.long()
        action_embs = self.dist_mem.get_action_embeddings(
            aval_actions, options=self.args.use_option_embs)

        act = self.action_linear(action_embs)
        x = torch.cat([x.view([x.shape[0], 1, x.shape[1]]).repeat(1, act.shape[1], 1), act], dim=-1)
        x = self.linear(x).squeeze(-1)
        if self.args.use_dist_double:
            x = x.double()
        return FixedCategorical(logits=x)

class RndCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super().__init__()

        self.args = args
        self.num_outputs = num_outputs


    def forward(self, x, add_input):
        return FixedCategorical(logits=torch.rand((x.shape[0], self.num_outputs)).cuda())


def extract_aval(extra, infos):
    aval = []
    for i, info in enumerate(infos):
        aval.append(info['aval'])

    aval = torch.FloatTensor(aval)
    return aval

def init_extract_aval(args, evaluate=False):
    init_aval = torch.LongTensor(args.aval_actions)
    init_aval = init_aval.unsqueeze(0).repeat(
        args.eval_num_processes if (evaluate and args.eval_num_processes is not None) else args.num_processes,
        1)
    return init_aval

class MainMethod(BasePolicy):
    def __init__(self, args, obs_space, action_space):
        self.args = args
        self.emb_mem = args.emb_mem
        self.dist_mem = args.dist_mem

        super().__init__(args, obs_space, action_space)

    def _get_disc_policy(self, num_outputs):
        if num_outputs == 2 and self.args.separate_skip:
            return Categorical(self.actor_critic.base.output_size,
                    num_outputs, self.args)

        if self.args.random_policy:
            return RndCategorical(self.actor_critic.base.output_size,
                    num_outputs, self.args)
        elif self.args.nearest_neighbor:
            return Categorical(self.actor_critic.base.output_size,
                    num_outputs, self.args)
        else:
            return OrderInvariantCategorical(self.actor_critic.base.output_size,
                    self.dist_mem, self.args)



    def _create_actor_critic(self):
        super()._create_actor_critic()

        if self.args.fine_tune:
            for name, module in self.actor_critic.named_modules():
                if isinstance(module, Categorical):
                    self.ignore_layers.append(name)

    def get_dim_add_input(self):
        return self.args.aval_actions.shape[-1]

    def get_add_input(self, extra, infos):
        return extract_aval(extra, infos)

    def get_init_add_input(self, args, evaluate=False):
        return init_extract_aval(args, evaluate=evaluate)

    def compute_fixed_action_set(self, take_action, aval_actions, args):
        if take_action.shape[-1] == 3:
            # This is a parameterized action space as with logic game or block
            # stacking.
            nn_result = self.emb_mem.nearest_neighbor_action(
                take_action[:, 0],
                args.training_fixed_action_set,
                aval_actions.long())
            if nn_result.shape[-1] == 1:
                nn_result = nn_result.squeeze(-1)
            take_action[:, 0] = nn_result
        else:
            take_action = self.emb_mem.nearest_neighbor_action(
                take_action.squeeze(-1),
                args.training_fixed_action_set,
                aval_actions.long())
            take_action = torch.LongTensor(
                np.round(take_action.cpu().numpy()))
        return take_action


    def get_action(self, state, add_input, recurrent_hidden_state,
                   mask, args, network=None, num_steps=None):
        # Sample actions
        with torch.no_grad():
            extra = {}
            parts = self.actor_critic.act(state, recurrent_hidden_state, mask,
                                     add_input=add_input,
                                     deterministic=args.deterministic_policy)
            value, action, action_log_prob, rnn_hxs, act_extra = parts
            if isinstance(act_extra, dict):
                act_extra = [act_extra]
            action_cpu = action.cpu().numpy()

            take_action = action_cpu
            if take_action.dtype == np.int64:
                take_action = torch.LongTensor(take_action)
            else:
                take_action = torch.Tensor(take_action)

            if args.load_fixed_action_set:
                take_action = self.compute_fixed_action_set(take_action,
                                                            add_input, args)

            if 'inferred_z' in act_extra:
                extra = {
                    **extra, **meth_utils.add_mag_stats(extra, act_extra[0]['inferred_z'])}

            entropy_reward = act_extra[0]['dist_entropy'].cpu() * args.reward_entropy_coef
            extra['alg_add_entropy_reward'] = entropy_reward.mean().item()
            extra['add_input'] = None

            add_reward = entropy_reward
            ac_outs = (value, action, action_log_prob, rnn_hxs)
            q_outs = (take_action, add_reward, extra)
            return ac_outs, q_outs

    def _get_action_emb(self, actions_idx):
        action_emb = self.dist_mem.get_action_embeddings(
            actions_idx[:, 0].long())
        if self.args.cuda:
            action_emb = action_emb.cuda()
        return action_emb

