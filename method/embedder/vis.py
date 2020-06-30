import numpy as np
import random
import os
import os.path as osp
from method.embedder.utils import scatter_contexts
from sklearn.manifold import TSNE
import torch
from envs.gym_minigrid.action_sets import get_option_properties

def vis_embs(dist_mem, emb_mem, n_dists, exp_type, render, save_prefix, args,
        viz_folder='./method/embedder/visualization/', use_idx=None):
    viz_folder = osp.join(viz_folder, args.load_emb_model_file.split('.')[0])
    if not osp.exists(viz_folder):
        os.makedirs(viz_folder)

    means = dist_mem.mem_keys.select(1, 0)
    logvars = dist_mem.mem_keys.select(1, 1)
    opt_embs = emb_mem.mem_keys

    if 'MiniGrid' in args.env_name:
        mean_labels, label_list = get_option_properties(args, quadrant=True)
    else:
        mean_labels, label_list = args.env_interface.get_env_option_names()

    if use_idx is not None:
        means = means[use_idx]
        logvars = logvars[use_idx]
        opt_embs = opt_embs[use_idx]
        mean_labels = np.array(mean_labels)[use_idx]

    if n_dists is not None:
        # Limit the number of distributions
        random.seed(args.seed)
        select_indices = random.sample(list(range(means.shape[0])),
                min(n_dists, means.shape[0]))
        means = means[select_indices]
        logvars = logvars[select_indices]
        opt_embs = opt_embs[select_indices]
        mean_labels = np.array(mean_labels)[select_indices]
    print('-' * 20)
    print('Across %i dists' % means.shape[0])
    print('opt embs mean  %.7f' % torch.mean(opt_embs))
    print('opt embs std %.7f' % torch.std(opt_embs))
    print('-' * 20)
    perplexity = 10
    if render:
        print('Computing option embeddings TSNE')
        # Compute TSNE
        tsne4 = TSNE(n_components=2, perplexity=perplexity, verbose=1)
        tsne4_results = tsne4.fit_transform(opt_embs.data.cpu().numpy())

        # Visualize
        save_path = osp.join(viz_folder, save_prefix + '_embs' +
                ('_test_split' if args.test_split else ''))
        print('Saving option embeddings to %s' % save_path)
        scatter_contexts(tsne4_results, np.array(mean_labels), label_list,
                savepath=save_path + '.pdf', no_label=True)
        scatter_contexts(tsne4_results, np.array(mean_labels), label_list,
                savepath=save_path + '_leg.pdf', no_label=False)

