import numpy as np
import torch
import torch.nn as nn
from method.utils import Conv2d3x3
from rlf.rl.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, args,
                 base=None,
                 dist_mem=None,
                 z_dim=None,
                 add_input_dim=0):
        super().__init__()
        self.obs_shape = obs_shape
        self.add_input_dim = add_input_dim

        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase_NEW
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError(
                    'Observation space is %s' % str(obs_shape))

        self.action_space = action_space
        self.args = args
        self.env_name = args.env_name

        use_action_output_size = 0

        self.base = base(obs_shape[0], add_input_dim,
                         action_output_size=use_action_output_size,
                         recurrent=args.recurrent_policy, hidden_size=args.state_encoder_hidden_size,
                         use_batch_norm=args.use_batch_norm, args=args)

    def clone_fresh(self):
        p = Policy(self.obs_shape, self.action_space, self.args,
                   type(self.base) if self.base is not None else None,
                   self.add_input_dim)

        if list(self.parameters())[0].is_cuda:
            p = p.cuda()

        return p

    def get_policies(self):
        return [self]

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_pi(self, inputs, rnn_hxs, masks, add_input=None):
        value, actor_features, rnn_hxs = self.base(
            inputs, rnn_hxs, masks, add_input)
        self.prev_actor_features = actor_features

        dist = self.dist(actor_features, add_input)
        return dist, value

    def act(self, inputs, rnn_hxs, masks, deterministic=False, add_input=None):
        dist, value = self.get_pi(
            inputs, rnn_hxs, masks, add_input)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        self.prev_action = action

        action_log_probs = dist.log_probs(action.double() if self.args.use_dist_double else action)
        dist_entropy = dist.entropy()

        if self.args.use_dist_double:
            if self.action_space.__class__.__name__ != "Discrete":
                action = action.float()
            action_log_probs = action_log_probs.float()
            dist_entropy = dist_entropy.float()
        if len(dist_entropy.shape) == 1:
            dist_entropy = dist_entropy.unsqueeze(-1)
        extra = {
            'dist_entropy': dist_entropy
        }

        return value, action, action_log_probs, rnn_hxs, extra

    def get_value(self, inputs, rnn_hxs, masks, action, add_input):
        value, actor_features, _ = self.base(inputs, rnn_hxs, masks, add_input)

        self.prev_actor_features = actor_features
        self.prev_action = action[:, :1].long()

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, add_input):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, add_input)

        dist = self.dist(actor_features, add_input)

        self.prev_actor_features = actor_features
        self.prev_action = action

        action_log_probs = dist.log_probs(action.double() if self.args.use_dist_double else action)
        dist_entropy = dist.entropy()

        if self.args.use_dist_double:
            action_log_probs = action_log_probs.float()
            dist_entropy = dist_entropy.float()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs



class CNNBase_NEW(NNBase):
    def __init__(self, num_inputs, add_input_dim,
                 action_output_size=32,
                 recurrent=False, hidden_size=64,
                 use_batch_norm=False, args=None):
        super().__init__(recurrent, hidden_size, hidden_size)
        self.args = args

        if self.args.env_name.startswith('MiniGrid'):
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=num_inputs,
                          out_channels=8, downsample=True),
                # shape is now (-1, 8, 5, 5)
                Conv2d3x3(in_channels=8, out_channels=8, downsample=True),
                # shape is now (-1, 8, 3, 3)
                Conv2d3x3(in_channels=8, out_channels=16, downsample=False),
                # shape is now (-1, 16, 3, 3)
            ])

            self.flat_size = 16 * 3 * 3

        else:
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=num_inputs,
                          out_channels=16, downsample=True),
                # shape is now (-1, 16, 42, 42)
                Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
                # shape is now (-1, 16, 21, 21)
                Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
                # shape is now (-1, 16, 11, 11)
                Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
                # shape is now (-1, 32, 6, 6)
                Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
                # shape is now (-1, 32, 3, 3)
            ])

            self.flat_size = 32 * 3 * 3

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.linear_layer = nn.Sequential(
            init_(nn.Linear(self.flat_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.nonlinearity = nn.ReLU()
        self.raw_state_emb = None
        self.hidden_size = hidden_size

        self.train()

    def forward(self, inputs, rnn_hxs, masks, action_pooled=None,
                add_input=None):
        if inputs.dtype == torch.uint8:
            inputs = inputs.float()
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.nonlinearity(x)
        x = x.view(-1, self.flat_size)
        x = self.linear_layer(x)

        self.raw_state_emb = x.clone()

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, add_input_dim,
                 action_output_size=32,
                 recurrent=False, hidden_size=64,
                 use_batch_norm=False, args=None):
        super().__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        # Action embedder
        self.actor_action = nn.Sequential(
            init_(nn.Linear(hidden_size + action_output_size, hidden_size)), nn.Tanh())
        # Action embedder - Critic
        self.critic_action = nn.Sequential(
            init_(nn.Linear(hidden_size + action_output_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(
            nn.Linear(hidden_size + action_output_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, add_input):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


