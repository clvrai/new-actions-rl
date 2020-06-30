import numpy as np
import torch
import random
import os
import os.path as osp
from method.embedder.utils import (scatter_contexts, tensor_gaussian_log_likelihood,
    tensor_kl_diagnormal_diagnormal, tensor_gaussian_log_likelihood_per_dim, tensor_kl_diagnormal_diagnormal_dim)
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class DistributionMemory(object):
    def __init__(self, cuda=False, n_distributions=1, args=None):
        self.mem = []
        self.cuda = cuda
        self.mem_keys = None
        self.mem_size = 0
        self.n_distributions = n_distributions
        self.args = args
        self.r = None

    def randomize_embs(self):
        # Means
        min_mean = self.mem_keys[:, 0].min(0)[0]
        max_mean = self.mem_keys[:, 0].max(0)[0]

        # Logvars
        min_logvar = self.mem_keys[:, 1].min()
        max_logvar = self.mem_keys[:, 1].max()

        rnd_mean = torch.rand(self.mem_keys.shape[0], self.mem_keys.shape[-1])
        rnd_logvar = torch.rand(self.mem_keys.shape[0], self.mem_keys.shape[-1])

        if self.args.cuda:
            rnd_mean = rnd_mean.cuda()
            rnd_logvar = rnd_logvar.cuda()

        self.mem_keys[:, 0] = min_mean + (max_mean - min_mean) * rnd_mean
        self.mem_keys[:, 1] = min_logvar + (max_logvar - min_logvar) * rnd_logvar


    def no_var(self):
        eps = 0.00001
        self.mem_keys[:, 1] = eps
        for i in range(len(self.mem)):
            self.mem[i][0][1] = eps


    def sample_action(self, z, K, sample=False, aval_actions=None):
        if self.args is not None:
            assert z.shape[-1] == self.args.z_dim, 'using z dim %i expected %i' % (z.shape[-1], self.args.z_dim)
        mean = self.mem_keys.select(1, 0)[aval_actions]
        logvar = self.mem_keys.select(1, 1)[aval_actions]
        z2 = z.view(z.shape[0], 1, z.shape[1])
        logp = tensor_gaussian_log_likelihood(z2, mean, logvar)

        # For diagnosing
        self.std_sel_p = torch.mean(torch.std(logp.exp(), dim=-1))

        if sample:
            a = torch.multinomial(logp.exp(), num_samples=K, replacement=False)
        else:
            _, a = torch.topk(logp, K, dim=1, largest=True)

        return a


    def sample_action_many_z(self, z, K, sample=False, aval_actions=None,
            num_z=1):
        mean = self.mem_keys.select(1, 0)[aval_actions]
        logvar = self.mem_keys.select(1, 1)[aval_actions]

        z2 = z.clone()

        z2 = z2.view(z.shape[0], num_z, 1, self.args.z_dim)
        mean = mean.view(mean.shape[0], 1, mean.shape[1], mean.shape[2])
        logvar = logvar.view(logvar.shape[0], 1, logvar.shape[1], logvar.shape[2])

        logp = tensor_gaussian_log_likelihood(z2, mean, logvar)

        logp = logp.view(z.shape[0], num_z, -1)
        logp = logp.double()

        prob = F.softmax(logp, dim=-1)
        prob = prob.mean(1)
        self.std_sel_p = torch.mean(torch.std(prob, dim=-1))

        if sample:
            a = torch.multinomial(prob, num_samples=K, replacement=False)
        else:
            _, a = torch.topk(prob, K, dim=1, largest=True)

        # Note: a here is the index of action in dist_mem.mem_keys
        # However, the actual action discrete value may be something else

        return a


    def get_action_embeddings(self, aval_actions, options=True):
        if options:
            all_embs = self.option_embs
        else:
            all_embs = self.mem_keys.select(1, 0)

        return all_embs[aval_actions]

    def normalize_mem(self):
        use_embs = self.option_embs
        # use_embs = (use_embs - use_embs.mean(0)) / use_embs.std(0)
        use_embs = use_embs / ((use_embs**2).sum(1).sqrt().unsqueeze(-1))
        self.option_embs = use_embs

    def kl_action_logits(self, z_mean, z_logvar, aval_actions):
        # z should be p distribution (original) in D(P || Q)
        mean = self.mem_keys.select(1, 0)[aval_actions]
        logvar = self.mem_keys.select(1, 1)[aval_actions]

        z_mean = z_mean.view(z_mean.shape[0], 1, z_mean.shape[1])
        z_logvar = z_logvar.view(z_logvar.shape[0], 1, z_logvar.shape[1])

        # meand = mean.double()
        # logvard = logvar.double()
        # z_meand = z_mean.double()
        # z_logvard = z_logvar.double()
        # kl_distances = -tensor_kl_diagnormal_diagnormal(meand, logvard, z_meand, z_logvard)
        # sel_logits = kl_distances.float()
        # return sel_logits

        kl_distances = -tensor_kl_diagnormal_diagnormal(mean, logvar, z_mean, z_logvar)
        return kl_distances


    def kl_action_logits_dim(self, z_mean, z_logvar, aval_actions):
        # z should be p distribution (original) in D(P || Q)
        mean = self.mem_keys.select(1, 0)[aval_actions]
        logvar = self.mem_keys.select(1, 1)[aval_actions]

        z_mean = z_mean.view(z_mean.shape[0], 1, z_mean.shape[1])
        z_logvar = z_logvar.view(z_logvar.shape[0], 1, z_logvar.shape[1])

        kl_distances = -tensor_kl_diagnormal_diagnormal_dim(mean, logvar, z_mean, z_logvar)
        return kl_distances


    def effect_kl(self, z_mean, z_logvar, aval_actions):
        # z should be p distribution (original) in D(P || Q)
        mean = self.mem_keys.select(1, 0)[aval_actions]
        logvar = self.mem_keys.select(1, 1)[aval_actions]

        kl_distances = tensor_kl_diagnormal_diagnormal(mean, logvar, z_mean, z_logvar)
        return kl_distances


    def get_action_logits(self, z, aval_actions):
        mean = self.mem_keys.select(1, 0)[aval_actions]
        logvar = self.mem_keys.select(1, 1)[aval_actions]
        logp = tensor_gaussian_log_likelihood(z.view(z.shape[0], 1, z.shape[1]), mean, logvar)
        return logp


    def get_action_logits_per_action(self, z, aval_actions):
        mean = self.mem_keys.select(1, 0)[aval_actions, :]
        logvar = self.mem_keys.select(1, 1)[aval_actions, :]
        logp = tensor_gaussian_log_likelihood(z, mean, logvar)
        return logp

    def get_action_logits_per_action_dim(self, z, aval_actions):
        mean = self.mem_keys.select(1, 0)[aval_actions, :]
        logvar = self.mem_keys.select(1, 1)[aval_actions, :]
        logp = tensor_gaussian_log_likelihood_per_dim(z, mean, logvar)
        return logp


    def get_action_dim_probs(self, z, aval_actions):
        mean = self.mem_keys.select(1, 0)[aval_actions]
        logvar = self.mem_keys.select(1, 1)[aval_actions]
        z = z.view(z.shape[0], 1, z.shape[1])
        z_prob_dim = tensor_gaussian_log_likelihood_per_dim(z, mean, logvar)
        return z_prob_dim

    def store_model(self, model):
        # Store the latent distribution function p(z|None, o, s0)
        self.model = model

    def store_embs(self, mem_keys):
        self.option_embs = mem_keys
        if self.args.use_double:
            self.option_embs = self.option_embs.double()

    def sample_from_embs(self, z, state, sample=False):
        # only feed in the observation that the
        mean, logvar = self.model.get_z_log_likelihood(z, state, self.option_embs)
        z = z.view(z.shape[0], 1, z.shape[1])

        logp = tensor_gaussian_log_likelihood(z, mean, logvar)
        action_distribution = torch.distributions.Categorical(logits=logp)

        if sample:
            a = action_distribution.sample()
        else:
            a = action_distribution.mode()

        return a


    def get_key_dim(self):
        return self.args.z_dim


    def add_distribution(self, dist, val):
        self.mem.append((dist, val),)
        if len(np.array(dist).shape) == 0:
            dist = np.expand_dims(np.array(dist), axis=0)

        new_key = torch.tensor(dist).unsqueeze(0)

        if self.cuda:
            new_key = new_key.cuda()

        if self.mem_keys is None:
            self.mem_keys = new_key
        else:
            self.mem_keys = torch.cat((self.mem_keys, new_key), 0)

        self.mem_size += 1

    def __len__(self):
        return self.mem_size

    def sample_values(self, size):
        return np.array([random.choice(self.mem)[1] for _ in range(size)])


    def save_distributions(self, save_file):
        if not osp.exists('method/embedder/saved_distributions'):
            os.makedirs('method/embedder/saved_distributions')
        path = osp.join('method/embedder/saved_distributions', save_file + '.emb')

        torch.save({
            'mem': self.mem,
            'mem_keys': self.mem_keys,
            'mem_size': self.mem_size
            }, path)


    def load_gt(self, env_name, use_cuda, gt_embs):
        # Also append with a fixed variance
        eps = -2.
        logvar = torch.zeros(gt_embs.shape)
        logvar[:] = eps
        gt_embs = torch.stack([gt_embs, logvar], dim=1)
        if use_cuda:
            gt_embs = gt_embs.cuda()

        self.mem_keys = gt_embs
        self.mem_size = self.mem_keys.shape[0]
        self.mem = [[gt_embs[i].cpu().numpy(), i] for i in range(len(gt_embs))]


    def load_distributions(self, load_file):
        path = osp.join('method/embedder/saved_distributions', load_file + '.emb')
        print('loading dists from ' + path)
        ckpt = torch.load(path)

        self.mem = ckpt['mem']
        self.mem_keys = ckpt['mem_keys']
        self.mem_size = ckpt['mem_size']
        if self.args.use_double:
            self.mem_keys = self.mem_keys.double()

        if self.cuda:
            self.mem_keys = self.mem_keys.cuda()
        else:
            self.mem_keys = self.mem_keys.cpu()

    def set_max_range(self, args):
        means = self.mem_keys.narrow(1, 0, 1).squeeze()
        std = self.mem_keys.narrow(1, 1, 1).squeeze().exp().sqrt()

        if self.args.constrained_effects:
            minn = means.min(0)[0]
            maxx = means.max(0)[0]
            mid = (minn + maxx) / 2.
            scale = args.emb_margin * (maxx - minn) # 1.5 is added to have some space from the boundaries
        else:
            minn = (means - args.max_std_width * std).min(0)[0]
            maxx = (means + args.max_std_width * std).max(0)[0]
            mid = (minn + maxx) / 2.
            scale = maxx - minn

        args.discrete_beta_mid = mid
        args.discrete_beta_scale = scale

