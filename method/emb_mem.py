import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import os.path as osp
from method.embedder.utils import scatter_contexts
from method.embedder.utils import (tensor_gaussian_log_likelihood,
    tensor_kl_diagnormal_diagnormal)
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict



class EmbeddingMemory(object):
    def __init__(self, cuda=False, args=None):
        self.mem = []
        self.mem_logvar = []

        self.cuda = cuda

        self.mem_keys = None
        self.mem_logvar_keys = None

        self.mem_size = 0
        self.mem_logvar_size = 0

        self.temperature = 1.
        self.topk = False
        self.visit = 0
        self.extract_disc = False
        self.fixed_ac_set = None
        self.args = args
        self.emb_map = None
        self.action_groups = None

    def should_compute_clusterings(self):
        return self.action_groups is None

    def compute_clusterings(self, action_set_indices):
        self.action_groups = defaultdict(list)

        rng = np.random.RandomState(123)
        samples = self.mem_keys.cpu().numpy()[action_set_indices]

        n_elements = samples.shape[0] // self.args.n_clusters + 1

        knn = MiniBatchKMeans(self.args.n_clusters, init='k-means++',
                random_state=rng)

        clusters = knn.fit_transform(samples)
        weights = clusters.min(axis=-1) - clusters.max(axis=-1)
        order = np.argsort(weights)
        count = [0 for _ in range(self.args.n_clusters)]
        for el in order:
            dists = np.argsort(clusters[el])
            for clust in dists:
                if count[clust] < n_elements:
                    self.action_groups[clust].append(action_set_indices[el])
                    count[clust] += 1
                    break

    def get_key_dim(self):
        return self.mem[0][0].shape[0]


    def load_gt(self, env_name, use_cuda, args, gt_embs):
        if use_cuda:
            gt_embs = gt_embs.cuda()

        self.mem_keys = gt_embs
        self.mem_size = self.mem_keys.shape[0]
        self.mem = [[gt_embs[i].cpu().numpy(), i] for i in range(len(gt_embs))]


    def load_embeddings(self, load_file):
        path = osp.join('method/embedder/saved_embeddings', load_file + '.emb')
        if not self.cuda:
            ckpt = torch.load(path, map_location=torch.device('cpu'))
        else:
            print('loading embeddings from ' + path)
            ckpt = torch.load(path)
        self.mem = ckpt['mem']
        self.mem_keys = ckpt['mem_keys']
        self.mem_size = ckpt['mem_size']
        if self.args.load_emb_logvar:
            self.mem_logvar = ckpt['mem_logvar']
            self.mem_logvar_keys = ckpt['mem_logvar_keys']
            self.mem_logvar_size = ckpt['mem_logvar_size']


    def replace(self, dist_mem):
        # May need change for double
        means = dist_mem.mem_keys.select(1, 0)

        self.mem = []
        for m in dist_mem.mem:
            self.mem.append((m[0][0], m[1]))

        self.mem_keys = means
        self.mem_size = dist_mem.mem_size


    def add_embedding(self, emb, val):
        self.mem.append((emb, val),)
        if len(np.array(emb).shape) == 0:
            emb = np.expand_dims(np.array(emb), axis=0)

        new_key = torch.Tensor(emb).unsqueeze_(0)

        if self.cuda:
            new_key = new_key.cuda()

        if self.mem_keys is None:
            self.mem_keys = new_key
        else:
            self.mem_keys = torch.cat((self.mem_keys, new_key), 0)

        self.mem_size += 1

    def add_emb_logvar(self, emb_logvar, val):
        self.mem_logvar.append((emb_logvar, val),)
        if len(np.array(emb_logvar).shape) == 0:
            emb_logvar = np.expand_dims(np.array(emb_logvar), axis=0)

        new_key = torch.Tensor(emb_logvar).unsqueeze_(0)

        if self.cuda:
            new_key = new_key.cuda()

        if self.mem_logvar_keys is None:
            self.mem_logvar_keys = new_key
        else:
            self.mem_logvar_keys = torch.cat((self.mem_logvar_keys, new_key), 0)

        self.mem_logvar_size += 1

    def normalize_mem(self, set_embs):
        if set_embs is None:
            x = self.mem_keys.cpu().numpy()
            # mean = np.mean(x, axis=0)
            # std = np.std(x, axis=0)
            # x = (x - mean) / std
            x = x / ((x**2).sum(1).sqrt().unsqueeze(-1))
        else:
            x = set_embs

        self.mem_keys = torch.Tensor(x)

        if self.cuda:
            self.mem_keys = self.mem_keys.cuda()

        for i in range(len(self.mem)):
            self.mem[i] = (x[i], self.mem[i][1])


    def __len__(self):
        return self.mem_size


    def sample_values(self, size):
        return np.array([random.choice(self.mem)[1] for _ in range(size)])


    def convert_ind(self, ind):
        if len(ind.shape) > 1 and ind.shape[1] > 1:
            return ind[:, 0].long()
        else:
            return ind


    def get_values(self, ind, aval):
        use_ind = self.convert_ind(ind)
        use_ind = torch.gather(aval, 1, use_ind.unsqueeze(-1))

        values = np.array([self.mem[i][1] for i in use_ind])
        if values.dtype == np.int64:
            values = torch.LongTensor(values)
        else:
            values = torch.FloatTensor(values)
        return values

    def get_for_real_ind(self, ind):
        if self.emb_map is None:
            self.emb_map = {}
            for i, (e, k) in enumerate(self.mem):
                self.emb_map[k] = i
        return self.emb_map[ind]

    def get_embeddings(self, ind, aval):
        use_ind = self.convert_ind(ind)
        use_ind = torch.gather(aval, 1, use_ind.unsqueeze(-1))

        embs = np.array([self.mem[i][0] for i in use_ind])
        embs = torch.FloatTensor(embs)
        if self.cuda:
            embs = embs.cuda()
        return embs


    def nearest_neighbor_action(self, action, fixed_action_set, aval_actions):
        if self.fixed_ac_set is None:
            self.fixed_ac_set = torch.LongTensor(fixed_action_set)
            if self.cuda:
                self.fixed_ac_set = self.fixed_ac_set.cuda()

        ac = action.clone().long()
        fixed_idx = self.fixed_ac_set[ac].squeeze(-1)

        embs = self.mem_keys[fixed_idx, :]
        if len(embs.shape) == 1:
            embs = embs.unsqueeze(0)
        return self.sample_action(embs, 1, False, aval_actions).squeeze(0).float()



    def sample_action(self, z, K, sample=False, aval_actions=None):
        z2 = z.clone()
        z2 = z2.unsqueeze(1)
        # use_idx, inverse_lookup = torch.unique(aval_actions, sorted=False, return_inverse=True)
        # embs = self.mem_keys[use_idx, :]

        embs = self.mem_keys[aval_actions]

        if self.args.cosine_distance:
            cos = torch.nn.CosineSimilarity(dim=-1)
            inv_distance = cos(z2, embs)
        else:
            inv_distance = -torch.sum(torch.pow(embs - z2, 2), dim=-1)

        # sel_logp = torch.gather(inv_distance, 1, inverse_lookup)

        if sample:
            a = torch.multinomial(F.softmax(inv_distance, dim=-1), num_samples=K, replacement=False)
        else:
            _, a = torch.topk(inv_distance, K, dim=1, largest=True)

        return a


    def get_action_logits_distance(self, z, aval_actions):
        use_idx, inverse_lookup = torch.unique(aval_actions, sorted=False, return_inverse=True)
        embs = self.mem_keys[use_idx, :]
        z = z.unsqueeze(1)

        if self.args.cosine_distance:
            cos = torch.nn.CosineSimilarity(dim=-1)
            inv_distance = cos(z, embs)
        else:
            inv_distance = -torch.sum(torch.pow(embs - z, 2), dim=-1).sqrt()

        sel_logp = torch.gather(inv_distance, 1, inverse_lookup)
        return sel_logp

    def get_action_logits_distribution(self, z, aval_actions):
        use_idx, inverse_lookup = torch.unique(aval_actions, sorted=False, return_inverse=True)
        mean = self.mem_keys[use_idx, :]
        logvar = self.mem_logvar_keys[use_idx, :]
        z = z.view(z.shape[0], 1, z.shape[1])
        logp = tensor_gaussian_log_likelihood(z, mean, logvar)
        sel_logp = torch.gather(logp, 1, inverse_lookup)
        return sel_logp


    def get_closest_k(self, lookup_k, aval, K):
        batch_size, D = lookup_k.shape[0], lookup_k.shape[-1]
        # batch size * Dimensions
        lookup = lookup_k.clone()

        lookup.unsqueeze_(1)
        lookup = lookup.expand(batch_size, self.mem_size, D)

        # Memory Key Tensor expanded from N * D
        mem_key = self.mem_keys
        mem_key = mem_key.expand(batch_size, self.mem_size, D)

        dist = torch.sum(torch.pow(mem_key - lookup, 2), dim=-1)
        dist = torch.gather(dist, 1, aval)

        val, ind = torch.topk(dist, K, dim=1, largest=False)

        result = self.mem_keys[ind]

        return result, ind



    def sample_k(self, lookup_k, aval, K, do_gumbel_softmax=False, num_steps=None):
        batch_size, D = lookup_k.shape[0], lookup_k.shape[-1]
        # batch size * Dimensions
        lookup = lookup_k.clone()

        lookup.unsqueeze_(1)
        lookup = lookup.expand(batch_size, self.mem_size, D)

        # Memory Key Tensor expanded from N * D
        mem_key = self.mem_keys
        mem_key = mem_key.expand(batch_size, self.mem_size, D)

        dist = torch.sum(torch.pow(mem_key - lookup, 2), dim=-1)

        if not self.topk:

            if not do_gumbel_softmax:
                probs = torch.nn.Softmax()(dist)
            else:
                probs = torch.nn.functional.gumbel_softmax(dist, tau=self.temperature)

            self.visit += 1

            self.temperature = 0.9 + (1 - 0.9) * np.exp(-self.visit/1000.)

            if self.temperature < 0.91:
                self.topk = True

            ind_k = torch.multinomial(probs, K, replacement=False)
        else:
            val, ind_k = torch.topk(dist, K, dim=1, largest=False)

        result = self.mem_keys[ind_k]

        return result, ind_k



    def visualize_embeddings(self, out_dir, value_label, label_list, label_type='trial'):
        if not osp.exists(out_dir + 'figures/'):
            os.makedirs(out_dir + 'figures/')
        # show coloured by labels
        path = out_dir + 'figures/' + label_type + '.pdf'

        scatter_contexts(self.mem_keys.cpu().numpy(), value_label,
                         label_list, savepath=path)


    def save_embeddings(self, save_file):
        if not osp.exists('method/embedder/saved_embeddings'):
            os.makedirs('method/embedder/saved_embeddings')
        path = osp.join('method/embedder/saved_embeddings', save_file + '.emb')

        torch.save({
            'mem': self.mem,
            'mem_keys': self.mem_keys,
            'mem_size': self.mem_size,
            'mem_logvar': self.mem_logvar,
            'mem_logvar_keys': self.mem_logvar_keys,
            'mem_logvar_size': self.mem_logvar_size,
            }, path)


    def randomize_embeddings(self):
        mem_keys = []
        old_mem = list(self.mem)

        minn = self.mem_keys.min(0)[0].cpu().numpy()
        maxx = self.mem_keys.max(0)[0].cpu().numpy()

        for i in range(self.mem_size):
            emb, val = self.mem[i]
            emb = np.random.uniform(low=minn, high=maxx, size=emb.shape)
            self.mem[i] = (emb, val)

            if len(np.array(emb).shape) == 0:
                emb = np.expand_dims(np.array(emb), axis=0)

            mem_keys.append(torch.Tensor(emb))

        self.mem_keys = torch.stack(mem_keys, dim=0)
        if self.cuda:
            self.mem_keys = self.mem_keys.cuda()

        self.extract_disc = False

    def set_max_range(self, args):
        embeddings = self.mem_keys

        minn = embeddings.min(0)[0]
        maxx = embeddings.max(0)[0]

        mid = (minn + maxx) / 2.
        scale = args.emb_margin * (maxx - minn) # 1.5 is added to have some space from the boundaries

        args.discrete_beta_mid = mid
        args.discrete_beta_scale = scale

