import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import os.path as osp
from tqdm import tqdm
from arguments import get_args
from rlf.rl import algo, utils
from method.emb_mem import EmbeddingMemory
from method.dist_mem import DistributionMemory
from method.embedder.embedder import Embedder
from rlf.rl.logger import Logger
import copy

from main import ExpRunSettings
from rlf.rl import utils

# Embedding Specific
from method.embedder.htvae import HTVAE
from method.embedder.embedder import Embedder

if __name__ == "__main__":
    run_settings = ExpRunSettings()
    args = run_settings.get_args()
    log_args = run_settings.get_set_args()

    log_dir = os.path.expanduser(args.log_dir)
    trial_log_dir = log_dir + "_trial"
    utils.cleanup_log_dir(trial_log_dir)

    # Set Seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    args.grid_playing = True

    if args.save_dataset:
        args.both_train_test = True
    env_trans_fn = run_settings.get_env_trans_fn(args)
    emb_mem = EmbeddingMemory(cuda=args.cuda, args=args)

    real_name = args.env_name

    data_folder = args.play_data_folder
    model_folder = args.emb_model_folder
    logger = Logger(log_args, './data/embedder/logs/')
    logger.set_prefix(args)

    # Data Generation Phase:
    if args.save_dataset:
        train_option_ids = args.overall_aval_actions

        embedder = Embedder(
            args, trial_log_dir,
            env_trans_fn, data_folder,
            option_ids=train_option_ids
            )

        print('No data to load... Generating Dataset')
        embedder.generate_dataset(args, trial_log_dir,
                env_trans_fn)

        print('Loading data... ')
        embedder.load_data_params()

        print('Successfully Loaded data params... ')
        args.load_dataset = True


    # Training Phase
    if args.train_embeddings or args.resume_emb_training:
        args.train_split = True
        args.test_split = False

        train_option_ids = args.overall_aval_actions

        embedder = Embedder(
            args, trial_log_dir,
            env_trans_fn, data_folder,
            option_ids=train_option_ids
            )

        # If no model to load, then generate training data
        if args.load_emb_model_file is None:
            # Generate Training Data
            if not args.load_dataset:
                print('No data to load... Generating Dataset')
                embedder.generate_dataset(args, trial_log_dir,
                        env_trans_fn)

        print('Loading data... ')
        embedder.load_data_params()

        print('Loaded data params... ')
        embedder.prepare_model(args, model_folder, logger, method=args.emb_method)

    # Testing Phase
    elif args.test_embeddings:
        test_option_ids = args.overall_aval_actions
        embedder = Embedder(args, trial_log_dir,
            env_trans_fn, data_folder,
            option_ids=test_option_ids)

        # Generate Test Data
        if not args.load_dataset:
            print('No data to load... Generating Dataset')
            embedder.generate_dataset(args, trial_log_dir,
                    env_trans_fn)

        print('Loading data...')
        embedder.load_data_params()

        assert args.load_emb_model_file is not None
        # Only loading model in testing phase
        embedder.prepare_model(args, model_folder, logger, method=args.emb_method)

        dist_mem = DistributionMemory(cuda=args.cuda, args=args)
        embedder.eval_dists_from_ids(dist_mem, None, 1)
        embedder.eval_embs_from_ids(emb_mem, None, dist_mem=dist_mem)

        embedder.visualize_trajectory_embeddings(args.emb_method,
            reconstruction=False, emb_mem=emb_mem, dist_mem=dist_mem)



