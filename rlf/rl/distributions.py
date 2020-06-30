import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlf.rl.utils import AddBias, init
from functools import partial
import numpy as np
from method.utils import EffectDiscrim

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal
FixedNormal.entropy_sum = True
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)
normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: (normal_entropy(self).sum(-1) if self.entropy_sum else normal_entropy(self).mean(-1))
FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Categorical, self).__init__()

        self.args = args

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, add_input):
        x = self.linear(x)
        if self.args.use_dist_double:
            x = x.double()
        return FixedCategorical(logits=x)



###################################
############# BETA ################
###################################

FixedBeta = torch.distributions.Beta
log_prob_beta = FixedBeta.log_prob
FixedBeta.scale_factor = 2.0
FixedBeta.mid_factor = 0.0
FixedBeta.log_probs = lambda self, actions: log_prob_beta(
    self, 0.5 + (actions - self.mid_factor)/(self.scale_factor + 1e-10)).sum(-1, keepdim=True)

beta_entropy = FixedBeta.entropy
FixedBeta.entropy = lambda self: beta_entropy(self).sum(-1)

FixedBeta.mode = lambda self: (self.mean - 0.5) * self.scale_factor + self.mid_factor

sample_beta = FixedBeta.sample
FixedBeta.sample = lambda self, num=[1]: self.scale_factor * (
    sample_beta(self, num).squeeze(0) - 0.5) + self.mid_factor


# http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
class Beta(nn.Module):
    def __init__(self, num_inputs, num_outputs, softplus=False, scale=2.0,
            mid=0.0, use_double=False):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.scale = scale
        self.mid = mid
        self.softplus = softplus
        self.use_double = use_double

        if softplus:
            self.softplus_fn = nn.Softplus(threshold=10)

        # Alpha and beta should always be positive
        self.fc_alpha = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_beta = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, add_input):
        # alpha, beta \in [1, inf] makes the distribution Uni-modal (as done in Chou et. al)
        if not self.softplus:
            alpha = 1. + torch.pow(self.fc_alpha(x), 2)
            beta = 1. + torch.pow(self.fc_beta(x), 2)
        else:
            alpha = 1 + self.softplus_fn(self.fc_alpha(x))
            beta = 1 + self.softplus_fn(self.fc_beta(x))
        if self.use_double:
            alpha = alpha.double()
            beta = beta.double()
        dist = FixedBeta(alpha, beta)
        dist.scale_factor = self.scale
        dist.mid_factor = self.mid
        return dist


###################################
############ Our Model ############
###################################
# Works the same as if rsample from this normal distribution
def reparameterize_gaussian(mean, logvar, cuda=True):
    std = torch.exp(0.5 * logvar)
    if cuda:
        eps = torch.randn(std.size()).cuda()
    else:
        eps = torch.randn(std.size())
    return mean + std * eps



###################################
######## Main Distribution ########
###################################
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, softplus=False, use_double=False,
        use_mean_entropy=False):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.use_double = use_double
        self.use_mean_entropy = use_mean_entropy
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))

        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x, add_input):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        if self.use_double:
            action_mean = action_mean.double()
            action_logstd = action_logstd.double()
        dist = FixedNormal(action_mean, action_logstd.exp())
        dist.entropy_sum = not self.use_mean_entropy
        return dist




class DiagGaussianVariance(nn.Module):
    def __init__(self, num_inputs, num_outputs, softplus=False, use_double=False,
        use_mean_entropy=False):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.softplus = softplus
        self.use_double = use_double
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.use_mean_entropy = use_mean_entropy

        if not softplus:
            self.fc_logstd = init_(nn.Linear(num_inputs, num_outputs))
        else:
            self.softplus_fn = nn.Softplus(threshold=10)
            self.fc_var = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, add_input):
        action_mean = self.fc_mean(x)
        if not self.softplus:
            action_logstd = self.fc_logstd(x)
            action_logstd = torch.clamp(action_logstd, min = -18.0, max = 2.0)
            if self.use_double:
                action_mean = action_mean.double()
                action_logstd = action_logstd.double()
            dist = FixedNormal(action_mean, torch.exp(action_logstd))
        else:
            action_var = self.softplus_fn(self.fc_var(x))
            if self.use_double:
                action_mean = action_mean.double()
                action_var = action_var.double()
            dist = FixedNormal(action_mean, action_var.sqrt())
        dist.entropy_sum = not self.use_mean_entropy
        return dist



class DistWrapper(torch.distributions.distribution.Distribution):
    def __init__(self, discs, conts, args=None):
        super().__init__()
        self.discs = discs
        self.conts = conts
        self.args = args
        self.cont_entropy_coef = args.cont_entropy_coef
        self.use_double = args.use_dist_double or args.use_double

    def mode(self):
        cont_modes = [c.mode().float() for c in self.conts]
        disc_modes = [d.mode().float() for d in self.discs]
        return torch.cat([*disc_modes, *cont_modes], dim=-1)

    def sample(self):
        cont_samples = [c.sample().float() for c in self.conts]
        disc_samples = [d.sample().float() for d in self.discs]
        return torch.cat([*disc_samples, *cont_samples], dim=-1)

    def log_probs(self, x):
        if self.use_double:
            x = x.double()

        cont_ac_start = len(self.discs)

        def get_cont_prob(c):
            # ONLY for gaussian distributions we clamp for Logic Game
            if isinstance(c, torch.distributions.Normal):
                cont_prob = torch.clamp(c.log_probs(x[:, cont_ac_start:]).float(), min = -1e5, max = 1e5)
            else:
                cont_prob = c.log_probs(x[:, cont_ac_start:]).float()
            return cont_prob

        cont_probs = [get_cont_prob(c) for c in self.conts]
        disc_probs = [d.log_probs(x[:, i:i+1]).float() for i, d in enumerate(self.discs)]

        log_probs = torch.cat([
            *disc_probs,
            *cont_probs], dim=-1)
        return log_probs.sum(-1).unsqueeze(-1)

    def __str__(self):
        return 'Cont: %s, Disc: %s' % (self.cont, self.disc)

    def entropy(self):
        cont_entropies = [self.cont_entropy_coef * c.entropy().float() for c in self.conts]
        disc_entropies = [d.entropy().float() for d in self.discs]
        entropy = torch.stack([*disc_entropies, *cont_entropies], dim=-1)
        return entropy.sum(-1).unsqueeze(-1)


class MixedDist(nn.Module):
    def __init__(self, disc_parts, cont_parts, args):
        super().__init__()

        self.cont_parts = nn.ModuleList(cont_parts)
        self.disc_parts = nn.ModuleList(disc_parts)

        self.cont_entropy_coef = args.cont_entropy_coef
        self.args = args

    def forward(self, x, add_input):
        conts = [cont(x, add_input) for cont in self.cont_parts]
        discs = [disc(x, add_input) for disc in self.disc_parts]
        return DistWrapper(discs, conts, args=self.args)


class GeneralDistWrapper(torch.distributions.distribution.Distribution):
    def __init__(self, dists, action_sizes, pos_idx, args=None):
        super().__init__()
        self.dists = dists
        self.action_sizes = action_sizes
        self.pos_idx = pos_idx

        self.args = args
        self.cont_entropy_coef = args.cont_entropy_coef
        self.use_double = args.use_dist_double

    def mode(self):
        modes = [d.mode().float() for d in self.dists]
        return torch.cat(modes, dim=-1)

    def sample(self):
        samples = [d.sample().float() for d in self.dists]
        return torch.cat(samples, dim=-1)

    def log_probs(self, x):
        if self.use_double:
            x = x.double()

        index = 0
        probs = []
        for i, d in enumerate(self.dists):
            next_index = index + self.action_sizes[i]
            if isinstance(d, torch.distributions.Normal):
                pr = torch.clamp(d.log_probs(x[:, index:next_index]).float(), min = -1e5, max = 1e5)
            else:
                pr = d.log_probs(x[:, index:next_index]).float()
            probs.append(pr)
            index = next_index
        probs = torch.cat(probs, dim=-1)
        return probs.sum(-1).unsqueeze(-1)

    def __str__(self):
        return 'Distributions: %s' % (self.dists)

    def entropy(self):
        entropies = [d.entropy().float() for d in self.dists]
        entropies[self.pos_idx] *= self.cont_entropy_coef
        entropy = torch.stack(entropies, dim=-1)
        return entropy.sum(-1).unsqueeze(-1)


class GeneralMixedDist(nn.Module):
    def __init__(self, parts, action_sizes, pos_idx, args):
        super().__init__()

        self.parts = nn.ModuleList(parts)
        self.action_sizes = action_sizes
        self.pos_idx = pos_idx
        self.args = args

    def forward(self, x, add_input):
        dists = [part(x, add_input) for part in self.parts]
        return GeneralDistWrapper(dists, self.action_sizes, self.pos_idx, args=self.args)


# Here we need another class for defining the distribution
class ConditionedAuxWrapper(torch.distributions.distribution.Distribution):
    def __init__(self, args, disc, alpha, beta, cont_entropy_coef=1e-1):
        super().__init__()
        self.args = args
        self.disc = disc
        self.cont = None
        self.cont_entropy_coef = cont_entropy_coef
        self.alpha = alpha
        self.beta = beta
        self.use_double = self.args.use_dist_double or self.args.use_double

    def mode(self):
        disc_mode = self.disc.mode()
        self.disc_sample = disc_mode        
        dc = torch.stack([disc_mode, disc_mode], -1)

        alpha = self.alpha.gather(1, dc).squeeze(1)
        beta = self.beta.gather(1, dc).squeeze(1)

        if self.args.use_beta:
            self.cont = FixedBeta(alpha, beta)
            self.cont.scale_factor = 2.0
            self.cont.mid_factor = 0.0
        else:
            self.cont = FixedNormal(alpha, beta)
        cont_mode = self.cont.mode()

        return torch.cat([disc_mode.float(), cont_mode.float()], dim=-1)

    def sample(self):
        disc_sample = self.disc.sample()
        self.disc_sample = disc_sample
        dc = torch.stack([disc_sample, disc_sample], -1)

        alpha = self.alpha.gather(1, dc).squeeze(1)
        beta = self.beta.gather(1, dc).squeeze(1)
        if self.args.use_beta:
            self.cont = FixedBeta(alpha, beta)
            self.cont.scale_factor = 2.0
            self.cont.mid_factor = 0.0
        else:
            self.cont = FixedNormal(alpha, beta)
        cont_sample = self.cont.sample()

        return torch.cat([disc_sample.float(), cont_sample.float()], dim=-1)

    def log_probs(self, x):
        discrete_action = x[:, :1].long()
        if self.use_double:
            x = x.double()

        if self.cont is None:
            dc = torch.stack([discrete_action, discrete_action], -1)
            alpha = self.alpha.gather(1, dc).squeeze(1)
            beta = self.beta.gather(1, dc).squeeze(1)
            if self.args.use_beta:
                self.cont = FixedBeta(alpha, beta)
                self.cont.scale_factor = 2.0
                self.cont.mid_factor = 0.0
            else:
                self.cont = FixedNormal(alpha, beta)
        else:
            assert (self.disc_sample == discrete_action).all()

        cont_prob = self.cont.log_probs(x[:, 1:])

        log_probs = torch.cat([
            self.disc.log_probs(x[:, :1]).float(),
            cont_prob.float()], dim=-1)

        return log_probs.sum(-1).unsqueeze(-1)

    def entropy(self):
        disc_ent = self.disc.entropy()
        # Should this be the average entropy over all the distributions?
        assert self.cont is not None
        cont_ent = self.cont.entropy()

        if len(disc_ent.shape) == 1:
            disc_ent = disc_ent.unsqueeze(-1)
            cont_ent = cont_ent.unsqueeze(-1)

        entropy = torch.cat([disc_ent.float(), self.cont_entropy_coef * cont_ent.float()], dim=-1)
        return entropy.sum(-1).unsqueeze(-1)

# This should be of the form GaussianVariance/Ordered invariant categorical
# Assumes Only one continuous auxiliary output and one discrete selection (no auxiliary discrete)
class ConditionedAuxDist(nn.Module):
    def __init__(self, state_size, cont_output_size, dist_mem, args, use_double=False):
        super().__init__()
        self.args = args
        self.use_double = use_double
        self.cont_entropy_coef = args.cont_entropy_coef

        # Discrete
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        hidden_dim = args.dist_hidden_dim
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
                init_(nn.Linear(hidden_dim + state_size, hidden_dim)), nn.ReLU(),
                init_(nn.Linear(hidden_dim, 1)))
        else:
            self.linear = init_(nn.Linear(hidden_dim + state_size, 1))


        # Continuous
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0))
        if args.use_beta:
            if args.softplus:
                self.softplus_fn = nn.Softplus(threshold=10)
            if args.conditioned_non_linear:
                self.fc_alpha = nn.Sequential(
                    init_(nn.Linear(hidden_dim + state_size, hidden_dim)), nn.ReLU(),
                    init_(nn.Linear(hidden_dim, cont_output_size)))
                self.fc_beta = nn.Sequential(
                    init_(nn.Linear(hidden_dim + state_size, hidden_dim)), nn.ReLU(),
                    init_(nn.Linear(hidden_dim, cont_output_size)))
            else:
                self.fc_alpha = init_(nn.Linear(hidden_dim + state_size, cont_output_size))
                self.fc_beta = init_(nn.Linear(hidden_dim + state_size, cont_output_size))
        else:
            if args.conditioned_non_linear:
                self.fc_mean = nn.Sequential(
                    init_(nn.Linear(hidden_dim + state_size, hidden_dim)), nn.ReLU(),
                    init_(nn.Linear(hidden_dim, cont_output_size)))
            else:
                self.fc_mean = init_(nn.Linear(hidden_dim + state_size, cont_output_size))
            # self.logstd = nn.Parameter(torch.randn(n_cont))
            if not args.softplus:
                if args.conditioned_non_linear:
                    self.fc_logstd = nn.Sequential(
                        init_(nn.Linear(hidden_dim + state_size, hidden_dim)), nn.ReLU(),
                        init_(nn.Linear(hidden_dim, cont_output_size)))
                else:
                    self.fc_logstd = init_(nn.Linear(hidden_dim + state_size, cont_output_size))
            else:
                self.softplus_fn = nn.Softplus(threshold=10)
                if args.conditioned_non_linear:
                    self.fc_var = nn.Sequential(
                        init_(nn.Linear(hidden_dim + state_size, hidden_dim)), nn.ReLU(),
                        init_(nn.Linear(hidden_dim, cont_output_size)))
                else:
                    self.fc_var = init_(nn.Linear(hidden_dim + state_size, cont_output_size))


    def forward(self, x, add_input):
        # cont_out should have one distribution for each of the discrete actions
        # Note: we don't need to store aval_actions in storage buffer if we have all the actions anyway
        aval_actions = add_input.long()
        action_embs = self.dist_mem.get_action_embeddings(
            aval_actions, options=self.args.use_option_embs)
        act = self.action_linear(action_embs)
        x = torch.cat([x.view([x.shape[0], 1, x.shape[1]]).repeat(1, act.shape[1], 1), act], dim=-1)

        # Discrete
        probs = self.linear(x).squeeze(-1)
        if self.args.use_double:
            probs = probs.double()
        disc_dist = FixedCategorical(logits=probs)

        # Continuous
        if self.args.use_beta:
            if not self.args.softplus:
                alpha = 1. + torch.pow(self.fc_alpha(x), 2)
                beta = 1. + torch.pow(self.fc_beta(x), 2)
            else:
                alpha = 1 + self.softplus_fn(self.fc_alpha(x))
                beta = 1 + self.softplus_fn(self.fc_beta(x))
            if self.use_double:
                alpha = alpha.double()
                beta = beta.double()
            return ConditionedAuxWrapper(self.args, disc_dist, alpha, beta, cont_entropy_coef=self.args.cont_entropy_coef)
        else:
            mean = self.fc_mean(x)
            if self.use_double:
                mean = mean.double()
            if not self.args.softplus:
                action_logstd = self.fc_logstd(x)
                action_logstd = torch.clamp(action_logstd, min = -18.0, max = 2.0)
                if self.use_double:
                    action_logstd = action_logstd.double()
                std = action_logstd.exp()
            else:
                action_var = self.softplus_fn(self.fc_var(x))
                if self.use_double:
                    action_var = action_var.double()
                std = action_var.sqrt()
            return ConditionedAuxWrapper(self.args, disc_dist, mean, std, cont_entropy_coef=self.args.cont_entropy_coef)

