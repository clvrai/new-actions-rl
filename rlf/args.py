
def str2bool(v):
    return v.lower() == 'true'


def add_args(parser):
    parser.add_argument('--wand', action='store_true', default=False)
    parser.add_argument('--use-double', action='store_true', default=False)
    parser.add_argument('--use-dist-double', type=str2bool, default=True)
    parser.add_argument('--use-mean-entropy', type=str2bool, default=None)

    parser.add_argument('--sync-host', type=str, default='')
    parser.add_argument('--sync-port', type=str, default='22')

    ########################################################
    # Distribution args
    ########################################################
    parser.add_argument('--use-beta', type=str2bool, default=True)
    parser.add_argument('--use-gaussian-distance', type=str2bool, default=True)
    parser.add_argument('--softplus', type=str2bool, default=True)
    parser.add_argument('--fixed-variance', type=str2bool, default=False)

    ########################################################
    ## PPO / A2C specific args
    ########################################################

    ## **
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    ## **
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--rl-use-radam', type=str2bool, default=False)
    parser.add_argument(
        '--lr-env-steps', type=int, default=None, help='only used for lr schedule')

    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    ## **
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    ## **
    parser.add_argument(
        '--use-gae',
        type=str2bool,
        default=True,
        help='use generalized advantage estimation')
    ## **
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    ## **
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=None,
        help='entropy term coefficient (old default: 0.01)')
    ## **
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    ## **
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    ## **
    parser.add_argument(
        '--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    ## **
    parser.add_argument(
        '--num-processes',
        type=int,
        default=None,
        help='how many training CPU processes to use (default: 32)')
    parser.add_argument(
        '--eval-num-processes',
        type=int,
        default=None,
        help='how many training CPU processes to use (default: None)')
    ## **
    parser.add_argument(
        '--num-steps',
        type=int,
        default=None,
        help='number of forward steps in A2C/PPO (old default: 128)')
    ## **
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    ## **
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=4,
        help='number of batches for ppo (default: 4)')
    ## **
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.1,
        help='ppo clip parameter (old default: 0.2)')
    parser.add_argument(
        '--prefix',
        default='',
        help='prefix of log dir')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 1)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=50,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--backup',
        type=str,
        default=None,
        help='whether to backup or not. Specify your username (default: None)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    ## **
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=None,
        help='number of environment steps to train (default: 1e8)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./data/trained_models/',
        help='directory to save agent trained models (default: ./data/trained_models/)')

    parser.add_argument(
        '--load-file',
        default='',
        help='.pt weights file')
    parser.add_argument(
        '--resume',
        default=False,
        action='store_true',
        help='Resume training')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    ## **
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    ## **
    parser.add_argument(
        '--recurrent-policy',
        type=str2bool,
        default=False,
        help='use a recurrent policy')
    ## **
    parser.add_argument(
        '--use-linear-lr-decay',
        type=str2bool,
        default=True,
        help='use a linear schedule on the learning rate')

