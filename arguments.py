import argparse
from rlf.args import add_args, str2bool

import torch

def set_if_none(args, var, value):
    vars(args)[var] = value if vars(args)[var] is None else vars(args)[var]


def grid_args(args):
    if args.load_embeddings_file is None and args.save_embeddings_file is None:
        args.load_embeddings_file = 'gw_st'

    set_if_none(args, 'exp_type', 'rnd')

    set_if_none(args, 'onehot_state', True)
    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'sample_clusters', False)
    set_if_none(args, 'action_set_size', 50)
    set_if_none(args, 'test_action_set_size', 50)
    set_if_none(args, 'o_dim', 16)
    set_if_none(args, 'z_dim', 16)
    set_if_none(args, 'num_env_steps', 40000000)

    if args.distance_based:
        set_if_none(args, 'num_steps', 256)
        set_if_none(args, 'use_mean_entropy', False)
    else:
        set_if_none(args, 'num_steps', 512)

    set_if_none(args, 'entropy_coef', 5e-2)
    set_if_none(args, 'num_eval', 20)
    set_if_none(args, 'eval_interval', 25)

    set_if_none(args, 'load_all_data', True)

    args.up_to_option = '5'
    args.grid_flatten = True
    args.no_frame_stack = True


def grid_play_args(args):

    set_if_none(args, 'exp_type', 'rnd')

    args.up_to_option = '5'
    args.trajectory_len = 6
    if args.train_embeddings and args.n_trajectories == 1024:
        args.n_trajectories = 672

    set_if_none(args, 'emb_epochs', 10000)
    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'o_dim', 16)
    set_if_none(args, 'z_dim', 16)
    set_if_none(args, 'onehot_state', True)
    set_if_none(args, 'emb_save_interval', 100)

    args.num_eval = 20
    args.action_random_sample = False
    set_if_none(args, 'load_all_data', True)
    if args.prefix == '':
        args.prefix = 'gw_train_embs'
    args.play_grid_size = 80
    args.grid_flatten = False
    args.no_frame_stack = True


def create_args(args):
    set_if_none(args, 'num_processes', 32)

    set_if_none(args, 'exp_type', 'NewMain')
    set_if_none(args, 'sample_clusters', False)
    set_if_none(args, 'split_type', 'gran_1')
    set_if_none(args, 'action_set_size', 50)
    set_if_none(args, 'test_action_set_size', 50)
    set_if_none(args, 'entropy_coef', 5e-3)
    set_if_none(args, 'num_env_steps', 60000000)
    set_if_none(args, 'lr_env_steps', 100000000)
    if args.distance_based:
        set_if_none(args, 'num_steps', 256)
        set_if_none(args, 'use_mean_entropy', True)
    else:
        set_if_none(args, 'num_steps', 384)
    set_if_none(args, 'num_eval', 20)

    set_if_none(args, 'o_dim', 128)
    set_if_none(args, 'z_dim', 128)
    if args.load_embeddings_file is None and args.save_embeddings_file is None:
        args.load_embeddings_file = 'create_g1_len7'

    if args.play_env_name is not None:
        if args.play_env_name.startswith('State'):
            args.load_all_data = True
        else:
            set_if_none(args, 'image_resolution', 48)
            args.load_all_data = False
            args.hidden_dim_traj = 128
            args.encoder_dim = 128
            args.hidden_dim_option = 128
    else:
        set_if_none(args, 'image_resolution', 64)

def create_play_args(args):
    set_if_none(args, 'exp_type', 'NewMain')
    if args.split_type is None:
        args.split_type = 'full_clean'
    if args.env_name.startswith('State'):
        set_if_none(args, 'emb_epochs', 10000)
        if args.train_embeddings and args.n_trajectories == 1024:
            args.n_trajectories = 464
        args.emb_batch_size = 128
        args.load_all_data = True
        args.prefix = 'StateCreate'
    else:
        set_if_none(args, 'emb_epochs', 5000)
        if args.train_embeddings and args.n_trajectories == 1024:
            args.n_trajectories = 45
        set_if_none(args, 'image_resolution', 48)

        args.emb_batch_size = 32
        args.load_all_data = False
        args.hidden_dim_traj = 128
        args.encoder_dim = 128
        args.hidden_dim_option = 128
        args.prefix = 'VideoCreate'

    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'o_dim', 128)
    set_if_none(args, 'z_dim', 128)
    args.action_random_sample = False

def stack_args(args):
    set_if_none(args, 'exp_type', 'rnd')
    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'eval_num_processes', 8)
    set_if_none(args, 'action_set_size', 20)
    set_if_none(args, 'test_action_set_size', 20)
    set_if_none(args, 'separate_skip', True)

    if args.env_name == 'StackEnvSimplestMovingAll-v0':
        args.contacts_off = False

    set_if_none(args, 'o_dim', 128)
    set_if_none(args, 'z_dim', 128)
    if args.load_embeddings_file is None and args.save_embeddings_file is None:
        args.load_embeddings_file = 'stack_rnd'

    set_if_none(args, 'num_env_steps', 3000000)
    set_if_none(args, 'lr_env_steps', 100000000)
    set_if_none(args, 'entropy_coef', 1e-2)
    set_if_none(args, 'sample_clusters', False)
    if args.distance_based:
        set_if_none(args, 'num_steps', 128)
    else:
        set_if_none(args, 'num_steps', 128)

    args.n_clusters = 10

    set_if_none(args, 'eval_interval', 25)
    args.save_interval = 25

    set_if_none(args, 'image_resolution', 84)

    args.load_all_data = False
    args.hidden_dim_traj = 128
    args.encoder_dim = 128
    args.hidden_dim_option = 128
    args.input_channels = 3
    args.emb_mlp_decoder = True

def stack_play_args(args):
    set_if_none(args, 'exp_type', 'rnd')
    set_if_none(args, 'o_dim', 128)
    set_if_none(args, 'z_dim', 128)
    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'emb_epochs', 5000)

    if args.train_embeddings and args.n_trajectories == 1024:
        args.n_trajectories = 52
    set_if_none(args, 'image_resolution', 84)

    args.emb_batch_size = 32
    set_if_none(args, 'load_all_data', False)
    args.hidden_dim_traj = 128
    args.encoder_dim = 128
    args.hidden_dim_option = 128
    args.prefix = 'ImageStack'
    args.input_channels = 3
    args.emb_mlp_decoder = True

def reco_args(args):
    set_if_none(args, 'exp_type', 'rnd')

    set_if_none(args, 'sample_clusters', False)
    if args.eval_only and args.reco_include_mu and (args.test_split or args.eval_split):
        set_if_none(args, 'action_set_size', 25)
    else:
        set_if_none(args, 'action_set_size', 500)
    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'eval_num_processes', 8)
    set_if_none(args, 'num_eval', 20)
    set_if_none(args, 'entropy_coef', 1e-2)
    set_if_none(args, 'test_action_set_size', 50)
    set_if_none(args, 'num_steps', 256)
    if args.reco_include_mu:
        set_if_none(args, 'num_env_steps', 80000000)
    else:
        set_if_none(args, 'num_env_steps', 40000000)
    set_if_none(args, 'eval_interval', 20)

    if args.fixed_action_set and not args.nearest_neighbor:
        set_if_none(args, 'reco_special_fixed_action_set_size', 2000)

    args.no_frame_stack = True
    args.gt_embs = True


def env_specific_args(args):
    if args.env_name.startswith('MiniGrid-Empty'):
        grid_play_args(args)
    elif args.env_name.startswith('MiniGrid'):
        grid_args(args)
    elif args.env_name.startswith('CreateLevel'):
        create_args(args)
    elif 'Create' in args.env_name:
        create_play_args(args)
    elif args.env_name.startswith('BlockPlayImg'):
        stack_play_args(args)
    elif args.env_name.startswith('Stack'):
        stack_args(args)
    elif args.env_name.startswith('Reco'):
        reco_args(args)


def general_args(args):
    set_if_none(args, 'emb_save_interval', 10)
    set_if_none(args, 'num_eval', 5)
    set_if_none(args, 'eval_interval', 50)
    set_if_none(args, 'load_all_data', False)
    set_if_none(args, 'image_resolution', 64)
    set_if_none(args, 'separate_skip', False)
    if args.distance_based:
        set_if_none(args, 'use_mean_entropy', True)
    else:
        set_if_none(args, 'use_mean_entropy', False)


def fixed_action_settings(args):
    if args.fixed_action_set:
        args.half_tool_ratio = None
        args.action_set_size = None



def get_args(arg_str=None):
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--fine-tune', action='store_true', default=False)
    parser.add_argument('--conditioned-aux', action='store_true', default=False)
    parser.add_argument('--conditioned-non-linear', type=str2bool, default=False)
    parser.add_argument('--load-best-name', type=str, default=None)

    parser.add_argument('--usage-policy', action='store_true', default=False)
    parser.add_argument('--usage-loss-coef', type=float, default=1.0)
    parser.add_argument('--normalize-embs', action='store_true', default=False)
    parser.add_argument('--only-vis-embs', action='store_true', default=False)
    parser.add_argument('--random-policy', action='store_true', default=False)
    parser.add_argument('--separate-skip', type=str2bool, default=None)
    parser.add_argument('--pac-bayes', action='store_true', default=False)
    parser.add_argument('--pac-bayes-delta', type=float, default=0.1)
    parser.add_argument('--complexity-scale', type=float, default=10.0)

    parser.add_argument('--create-len-filter', type=float, default=None)


    ########################################################
    # Action sampling related
    parser.add_argument('--sample-clusters', type=str2bool, default=None)
    parser.add_argument('--gt-clusters', action='store_true', default=False)
    parser.add_argument('--strict-gt-clusters', action='store_true', default=False)
    parser.add_argument('--n-clusters', type=int, default=10)

    parser.add_argument('--analysis-angle', type=int, default=None)
    parser.add_argument('--analysis-emb', type=float, default=None)
    parser.add_argument('--train-mix-ratio', type=float, default=None)
    ########################################################



    ########################################################
    # Reco related
    parser.add_argument('--reco-n-prods', type=int, default=10000)
    parser.add_argument('--reco-special-fixed-action-set-size', type=int, default=None)
    parser.add_argument('--reco-max-steps', type=int, default=100)
    parser.add_argument('--reco-normalize-beta', type=str2bool, default=True)
    parser.add_argument('--reco-include-mu', type=str2bool, default=False)
    parser.add_argument('--reco-change-omega', type=str2bool, default=True)
    parser.add_argument('--reco-random-product-view', type=str2bool, default=True)
    parser.add_argument('--reco-normal-time-generator', type=str2bool, default=True)
    parser.add_argument('--reco-deterministic', type=str2bool, default=False)
    parser.add_argument('--reco-no-repeat', type=str2bool, default=False)
    parser.add_argument('--reco-prod-dim', type=int, default=16)
    parser.add_argument('--reco-num-flips', type=int, default=0)
    ########################################################


    ########################################################
    # Stack env related
    parser.add_argument('--stack-reward', type=float, default=1.0)
    parser.add_argument('--stack-no-text-render', type=str2bool, default=True)
    parser.add_argument('--stack-mean-height', type=str2bool, default=True)
    parser.add_argument('--stack-dim', type=int, default=1)
    parser.add_argument('--constrain-physics', action='store_true', default=False)
    parser.add_argument('--contacts-off', type=str2bool, default=True)
    parser.add_argument('--double-place-pen', type=float, default=0.0)
    parser.add_argument('--stack-min-steps', type=int, default=2)
    parser.add_argument('--stack-max-steps', type=int, default=300)
    parser.add_argument('--stack-render-high', type=str2bool, default=True)
    ########################################################

    ########################################################
    # Entropy Reward related
    parser.add_argument('--reward-entropy-coef', type=float, default=0.0)
    ########################################################

    parser.add_argument('--gran-factor', type=float, default=1.0)

    parser.add_argument('--distance-effect', action='store_true', default=False)
    parser.add_argument('--distance-sample', action='store_true', default=False)
    parser.add_argument('--only-pos', action='store_true', default=False)


    ########################################################
    # Action set generation args
    parser.add_argument('--action-seg-loc', type=str, default='./data/action_segs')
    parser.add_argument('--action-random-sample', type=str2bool,
            default=True, help='Randomly sample actions or not.')

    parser.add_argument('--action-set-size', type=int, default=None)
    parser.add_argument('--test-action-set-size', type=int, default=None)
    parser.add_argument('--play-data-folder', type=str, default='./method/embedder/data')
    parser.add_argument('--emb-model-folder', type=str, default='./data/embedder/trained_models/')
    parser.add_argument('--create-play-len', type=int, default=7)
    parser.add_argument('--create-play-run-steps', type=int, default=3)
    parser.add_argument('--create-play-colored', action='store_true', default=False)
    parser.add_argument('--create-play-fixed-tool', action='store_true', default=False)
    parser.add_argument('--play-env-name', type=str, default=None)
    parser.add_argument('--image-resolution', type=int, default=None,
        help='If lower image resolution is to be used in play data')
    parser.add_argument('--image-mask', type=str2bool, default=True)

    parser.add_argument('--input-channels', type=int, default=1,
        help='No. of input channels for HTVAE')

    parser.add_argument('--test-split', action='store_true', default=False)
    parser.add_argument('--eval-split', action='store_true', default=False)
    parser.add_argument('--eval-split-ratio', type=float, default=0.5,
            help='Fraction of action set that is eval')

    parser.add_argument('--both-train-test', type=str2bool, default=False)
    parser.add_argument('--fixed-action-set', action='store_true', default=False)

    parser.add_argument('--load-fixed-action-set', action='store_true', default=False,
        help='For nearest neighbor lookup of discrete policy at evaluation')

    parser.add_argument('--num-z', type=int, default=1)

    parser.add_argument('--weight-decay', type=float, default=0.)

    parser.add_argument('--decay-clipping', action='store_true', default=False)


    ########################################################
    # Method specific args
    ########################################################
    parser.add_argument('--latent-dim', type=int, default=1)
    parser.add_argument('--action-proj-dim', type=int, default=1)
    parser.add_argument('--load-only-actor', type=str2bool, default=True)
    parser.add_argument('--sample-k', action='store_true', default=False)
    parser.add_argument('--do-gumbel-softmax', action='store_true', default=False)
    parser.add_argument('--discrete-fixed-variance', type=str2bool, default=False)
    parser.add_argument('--use-batch-norm', type=str2bool, default=False)

    parser.add_argument('--gt-embs', action='store_true', default=False)


    parser.add_argument(
        '--cont-entropy-coef',
        type=float,
        default=1e-1,
        help='scaling continuous entropy coefficient term further (default: 0.1)')

    # Discrete Beta settings
    parser.add_argument('--discrete-beta', type=str2bool, default=False)
    parser.add_argument('--max-std-width', type=float, default=3.0)
    parser.add_argument('--constrained-effects', type=str2bool, default=True)
    parser.add_argument('--bound-effect', action='store_true', default=False)

    parser.add_argument('--emb-margin', type=float, default=1.1)

    parser.add_argument('--nearest-neighbor', action='store_true', default=False)
    parser.add_argument('--combined-dist', action='store_true', default=False)
    parser.add_argument('--combined-add', action='store_true', default=False)

    parser.add_argument('--no-frame-stack', action='store_true', default=False)

    parser.add_argument('--dist-hidden-dim', type=int, default=64)
    parser.add_argument('--dist-linear-action', type=str2bool, default=True)
    parser.add_argument('--dist-non-linear-final', type=str2bool, default=True)

    parser.add_argument('--exp-logprobs', action='store_true', default=False)
    parser.add_argument('--kl-pen', type=float, default=None)
    parser.add_argument('--cat-kl-loss', type=float, default=None)

    parser.add_argument('--reparam', type=str2bool, default=True)
    parser.add_argument('--no-var', action='store_true', default=False)
    parser.add_argument('--z-mag-pen', type=float, default=None)

    # Distance Model specific
    parser.add_argument('--distance-based', action='store_true', default=False)
    parser.add_argument('--cosine-distance', action='store_true', default=False)

    # Gridworld specific
    parser.add_argument('--up-to-option', type=str, default=None)
    parser.add_argument('--no-diag', type=str2bool, default=True)
    parser.add_argument('--option-penalty', type=float, default=0.0)
    parser.add_argument('--grid-flatten', type=str2bool, default=True)
    parser.add_argument('--grid-playing', action='store_true', default=False)
    parser.add_argument('--play-grid-size', type=int, default=80)
    parser.add_argument('--onehot-state', type=str2bool, default=None)

    parser.add_argument('--not-upto', type=str2bool, default=True)
    parser.add_argument('--orig-crossing-env', type=str2bool, default=False)
    parser.add_argument('--max-grid-steps', type=int, default=50)
    parser.add_argument('--grid-subgoal', type=str2bool, default=True)
    parser.add_argument('--grid-fixed-rivers', type=str2bool, default=False)
    parser.add_argument('--grid-safe-wall', type=str2bool, default=True)

    # Video specific
    parser.add_argument('--vid-dir', type=str, default='./data/vids')
    parser.add_argument('--obs-dir', type=str, default='./data/obs')
    parser.add_argument('--should-render-obs', type=str2bool, default=False)
    parser.add_argument('--result-dir', type=str, default='./data/results')
    parser.add_argument('--vid-fps', type=float, default=5.0)
    parser.add_argument('--eval-only', action='store_true', default=False)
    parser.add_argument('--evaluation-mode', type=str2bool, default=False)

    parser.add_argument('--high-render-dim', type=int, default=256, help='Dimension to render evaluation videos at')
    parser.add_argument('--high-render-freq', type=int, default=50)
    parser.add_argument('--no-test-eval', action='store_true', default=False)
    parser.add_argument('--num-render', type=int, default=None)
    parser.add_argument('--num-eval', type=int, default=None)
    parser.add_argument('--render-info-grid', action='store_true', default=False)
    parser.add_argument('--deterministic-policy', action='store_true', default=False)

    parser.add_argument('--debug-render', action='store_true', default=False)
    parser.add_argument('--render-gifs', action='store_true', default=False)
    parser.add_argument('--verbose-eval', action='store_true', default=True)


    # CREATE specific
    parser.add_argument('--half-tools', type=str2bool, default=True)
    parser.add_argument('--half-tool-ratio', type=float, default=0.5)
    parser.add_argument('--marker-reward', type=str, default='reg',
                    help='Type of reward given for the marker ball [reg, dir]')
    parser.add_argument('--create-target-reward', type=float, default=1.0)
    parser.add_argument('--create-sec-goal-reward', type=float, default=2.0)

    parser.add_argument('--run-interval', type=int, default=10)
    parser.add_argument('--render-high-res', action='store_true', default=False)
    parser.add_argument('--render-ball-traces', action='store_true', default=False)
    parser.add_argument('--render-text', action='store_true', default=False)
    parser.add_argument('--render-changed-colors', action='store_true', default=False)

    # Mega render args
    parser.add_argument('--render-mega-res', action='store_true', default=False)
    parser.add_argument('--render-mega-static-res', action='store_true', default=False)
    parser.add_argument('--mega-res-interval', type=int, default=4)
    parser.add_argument('--anti-alias-blur', type=float, default=0.0)

    parser.add_argument('--render-result-figures', action='store_true', default=False)
    parser.add_argument('--render-borders', action='store_true', default=False)

    parser.add_argument('--success-failures', action='store_true', default=False)
    parser.add_argument('--success-only', action='store_true', default=False)

    parser.add_argument('--exp-type', type=str, default=None,
        help='Type of experiment')
    parser.add_argument('--split-type', type=str, default=None,
        help='Type of Splitting for New tools for create game')
    parser.add_argument('--deterministic-split', action='store_true', default=False)

    # Create environment specific
    parser.add_argument('--create-max-num-steps', type=int, default=30,
        help='Max number of steps to take in create game (Earlier default 25)')
    parser.add_argument('--create-permanent-goal', type=str2bool, default=True)
    parser.add_argument('--large-steps', type=int, default=40,
        help='Large steps (simulation gap) for create game (Earlier default 40)')
    parser.add_argument('--skip-actions', type=int, default=1,
        help='No. of actions to skip over for create game')
    parser.add_argument('--play-large-steps', type=int, default=30,
        help='Large steps (simulation gap) for create game play env')
    parser.add_argument('--no-overlap-env', type=str2bool, default=False)
    parser.add_argument('--threshold-overlap', type=str2bool, default=True)


    ########################################################
    # Embedding specific
    ########################################################

    parser.add_argument('--use-random-embeddings', action='store_true', default=False)
    parser.add_argument('--verify-embs', action='store_true', default=False)
    parser.add_argument('--n-distributions', type=int, default=1)


    parser.add_argument('--use-action-trajectory', action='store_true', default=False)
    parser.add_argument('--emb-batch-size', type=int, default=128)
    parser.add_argument('--trajectory-len', type=int, default=None)
    parser.add_argument('--n-trajectories', type=int, default=1024)
    parser.add_argument('--o-dim', type=int, default=None,
                    help='dimension of action (a or o) embeddings (Earlier default: 3)')
    parser.add_argument('--z-dim', type=int, default=None,
                    help='dimension of (trajectory) z variables (default: 5)')
    parser.add_argument('--print-vars', action='store_true', default=False,
                    help='whether to print all learnable parameters for sanity check '
                         '(default: False)')
    parser.add_argument('--emb-epochs', type=int, default=None)
    parser.add_argument('--emb-viz-interval', type=int, default=5,
                    help='number of epochs between visualizing action space '
                         '(default: -1 (only visualize last epoch))')
    parser.add_argument('--emb-save-interval', type=int, default=None,
                        help='number of epochs between saving model '
                             '(default: -1 (save on last epoch))')
    parser.add_argument('--n-hidden-traj', type=int, default=3,
                    help='number of hidden layers in modules outside option network '
                         '(default: 3)')
    parser.add_argument('--hidden-dim-traj', type=int, default=64,
                    help='dimension of hidden layers in modules outside option network '
                         '(default: 128)')
    parser.add_argument('--encoder-dim', type=int, default=64,
                    help='size of LSTM encoder output (default: 128)')
    parser.add_argument('--emb-learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
    parser.add_argument('--emb-use-batch-norm', type=str2bool, default=True)
    parser.add_argument('--emb-use-radam', type=str2bool, default=True)
    parser.add_argument('--emb-schedule-lr', type=str2bool, default=False)
    parser.add_argument('--deeper-encoder', type=str2bool, default=False,
                    help='whether or not to use deep convolutional encoder and decoder')
    parser.add_argument('--save-dataset', action='store_true', default=False)
    parser.add_argument('--load-dataset', type=str2bool, default=True)
    parser.add_argument('--load-all-data', type=str2bool, default=None)
    parser.add_argument('--save-emb-model-file', type=str, default=None)
    parser.add_argument('--load-emb-model-file', type=str, default=None)

    parser.add_argument('--train-embeddings', action='store_true', default=False)
    parser.add_argument('--test-embeddings', action='store_true', default=False)
    parser.add_argument('--resume-emb-training', action='store_true', default=False)

    parser.add_argument('--emb-method', type=str, default='htvae')
    parser.add_argument('--shared-var', type=str2bool, default=True)

    parser.add_argument('--load-emb-logvar', type=str2bool, default=True)

    parser.add_argument(
        '--emb-log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')

    parser.add_argument('--start-option', type=int, default=None)

    parser.add_argument('--save-embeddings-file', type=str,default=None)
    parser.add_argument('--load-embeddings-file', type=str,default=None)
    parser.add_argument('--skip-tsne', action='store_true', default=False)
    parser.add_argument('--num-distributions', type=int, default=400)
    parser.add_argument('--plot-samples', type=int, default=300)

    # htvae specific
    parser.add_argument('--n-hidden-option', type=int, default=3,
                    help='number of hidden layers in option network modules '
                     '(default: 3)')
    parser.add_argument('--hidden-dim-option', type=int, default=128,
                    help='dimension of hidden layers in option network (default: 128)')
    parser.add_argument('--n-stochastic', type=int, default=1,
                        help='number of z variables in hierarchy (default: 1)')
    parser.add_argument('--htvae-clip-gradients', type=str2bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')

    parser.add_argument('--emb-non-linear-lstm', type=str2bool, default=True)
    parser.add_argument('--emb-mlp-decoder', type=str2bool, default=False)
    parser.add_argument('--effect-only-decoder', type=str2bool, default=False)
    parser.add_argument('--concat-oz', type=str2bool, default=False)
    parser.add_argument('--no-initial-state', type=str2bool, default=False)

    #### Action Input to Policy specific
    parser.add_argument('--use-option-embs', type=str2bool, default=True)

    parser.add_argument('--action-base-output-size', type=int, default=64,
                        help='Dimensionality of action base output (Earlier 32 default)')
    parser.add_argument('--action-base-hidden-size', type=int, default=128,
                        help='Dimensionality of action base hidden layers (Earlier 32 default)')
    parser.add_argument('--state-encoder-hidden-size', type=int, default=64,
                        help='Dimensionality of state encoder hidden layers')

    add_args(parser)

    if arg_str is not None:
        args = parser.parse_args(arg_str.split(' '))
    else:
        args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.eval_only and args.num_eval == 10:
        args.num_eval = 100

    if args.save_dataset:
        args.load_dataset = False

    args.train_split = (not args.test_split and not args.eval_split)

    env_specific_args(args)
    general_args(args)

    fixed_action_settings(args)
    args.training_action_set_size = args.action_set_size

    if args.fine_tune:
        args.test_action_set_size = args.action_set_size
        args.nearest_neighbor = True
        args.fixed_action_set = True
        args.action_random_sample = False
        args.test_split = True
        args.train_split = False
        args.both_train_test = False

    if args.eval_interval == -1:
        args.eval_interval = None

    return args
