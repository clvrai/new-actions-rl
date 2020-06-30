
import sys
sys.path.insert(0, '.')
import libtmux
import argparse
import os
import os.path as osp
import time
import datetime
import warnings
import subprocess
from subprocess import PIPE
import uuid
from rlf.exp_mgr import config_mgr

parser = argparse.ArgumentParser()
parser.add_argument('--sess-id', type=int, default=-1,
        help='tmux session id to connect to. If unspec will run in current window')
parser.add_argument('--sess-name', default=None, type=str,
        help='tmux session name to connect to')
parser.add_argument('--cmd', type=str, default=None, help='list of commands to run')
parser.add_argument('--no-conda', action='store_true', default=False,
        help='If not to use conda, and switch to virtual env')
parser.add_argument('--virtual-env', default='~/env', type=str, help='Address of virtual environment')
parser.add_argument('--cd', default='1', type=str,
        help='String of CUDA_VISIBLE_DEVICES=(example: \"1 2\")')
parser.add_argument('--env', default=None, type=str)
parser.add_argument('--type', default='policy', type=str, help='"emb" or "policy"')
parser.add_argument('--save-name', default='def', type=str, help='Name of the file to save')
parser.add_argument('--no-wand', action='store_true', default=False)
parser.add_argument('--lvl', default=None, type=str, help='Create game level')
parser.add_argument('--test-runs', default=None, type=str, help='')
parser.add_argument('--train-eval', action='store_true', default=False)
parser.add_argument('--render-high', action='store_true', default=False)

cmd_folder = config_mgr.get_prop('cmds_loc')

def add_on_args(spec_args):
    spec_args = ['"' + x + '"' if ' ' in x else x for x in spec_args]
    return ((' '.join(spec_args)))

def get_cmds(cmd_loc, train_type, spec_args, args):
    try:
        open_cmd = osp.join(cmd_folder, cmd_loc + '.cmd')
        print('opening', open_cmd)
        with open(open_cmd) as f:
            cmds = f.readlines()
    except:
        raise ValueError('Must place %s command in %s' % (cmd_loc, cmd_folder))

    # pay attention to comments
    cmds = list(filter(lambda x: not (x.startswith('#') or x == '\n'), cmds))
    cmds = [cmd.rstrip() + " " for cmd in cmds]

    # Check if any commands are references to other commands
    all_ref_cmds = []
    for i, cmd in enumerate(cmds):
        if cmd.startswith('R:'):
            ref_cmd_loc = cmd.split(':')[1]
            ref_cmds = get_cmds(ref_cmd_loc.rstrip(), train_type, spec_args, args)
            all_ref_cmds.extend(ref_cmds)
    cmds = list(filter(lambda x: not x.startswith('R:'), cmds))

    if train_type != 'emb' and train_type != 'policy':
        raise ValueError('Invalid args type!')

    cmds = [cmd + add_on_args(spec_args) for cmd in cmds]
    if args.lvl is not None:
        cmds = [cmd % args.lvl for cmd in cmds]

    cmds.extend(all_ref_cmds)
    return cmds


def get_tmux_window(args):
    server = libtmux.Server()

    if args.sess_name is None:
        tmp = server.list_sessions()
        sess = server.get_by_id('$%i' % args.sess_id)
    else:
        sess = server.find_where({ "session_name": args.sess_name })
    if sess is None:
        raise ValueError('invalid session id')

    return sess.new_window(attach=False, window_name="auto_proc")


def learn_policy(args, rest):
    USE_WAND = not args.no_wand
    cmds = get_cmds(args.cmd, args.type, rest, args)
    add_on = ''
    #if '--prefix' not in rest:
    #    add_on = '--prefix NONE'

    add_on += ' --render-changed-colors'

    if args.sess_id == -1:
        if len(cmds) == 1:
            exec_cmd = cmds[0]
            exec_cmd = 'CUDA_VISIBLE_DEVICES=' + args.cd + ' ' + exec_cmd + ' ' + add_on
            if USE_WAND:
                exec_cmd += ' --wand'
            print('executing ', exec_cmd)
            os.system(exec_cmd)
            return
        else:
            raise ValueError('Must specify tmux session id')

    new_window = get_tmux_window(args)

    exp_files = []
    for cmd in cmds:
        cmd += ' ' + add_on
        if USE_WAND:
            cmd = cmd + ' --wand'
        print('running full command %s' % cmd)

        # Send the keys to run the command
        last_pane = new_window.attached_pane
        last_pane.send_keys(cmd, enter=False)
        pane = new_window.split_window(attach=False)
        pane.set_height(height=50)
        if args.no_conda:
            pane.send_keys('source ' + args.virtual_env + '/bin/activate')
        else:
            pane.send_keys('source deactivate')
            pane.send_keys('source activate tor')
        pane.enter()
        pane.send_keys('export CUDA_VISIBLE_DEVICES=' + args.cd)
        pane.enter()
        pane.send_keys(cmd)
        pane.enter()

        #exp_files.append(exp_file)

    print('everything should be running...')

def learn_embs(args, rest, extra_args):
    assert args.cmd is None, 'Cannot specify an command for embedding learning.'
    if not args.env.startswith('MiniGrid'):
        env_name = args.env + 'Play-v0'
    else:
        env_name = args.env
    import pdb; pdb.set_trace()
    cmd = 'python method/embedder/option_embedder.py --env-name "%s"' % env_name

    cmd += ' ' + extra_args + ' ' + ' '.join(rest)

    print('executing ', cmd)
    os.system(cmd)


args, rest = parser.parse_known_args()

if __name__ == '__main__':
    if args.type == 'policy':
        learn_policy(args, rest)
    elif args.type == 'eval':
        open_file = osp.join(cmd_folder, args.cmd + '.cmd')
        with open(open_file, 'r') as f:
            cmds = f.readlines()
        cmds = list(filter(lambda x: not (x.startswith('#') or x == '\n'), cmds))
        cmds = [cmd.rstrip() + " " for cmd in cmds]
        # we will only run one evaluation command
        cmd = cmds[0]
        cmd = 'CUDA_VISIBLE_DEVICES=' + args.cd + ' ' + cmd
        cmd = cmd % args.lvl
        cmd += ' --prefix "eval"'
        cmd += ' --num-eval 50'
        cmd += ' --eval-only'
        if args.render_high:
            cmd += ' --render-high-res'
            cmd += ' --render-ball-traces'
        cmd += ' ' + ' '.join(rest)
        if not args.train_eval:
            cmd += ' --test-split'

        all_runs = args.test_runs.split(',')
        results = {}
        for run in all_runs:
            use_cmd = cmd
            # Get the most trained model.
            max_count = -1
            use_file = None
            search_dir = 'trained_models/LogicLevel%s-v0/%s' % (args.lvl, run)
            locs = config_mgr.get_prob('backup_locs')[0]
            for f in os.listdir(search_dir):
                iter_count = int(f.split('_')[1].split('.')[0])
                if iter_count > max_count:
                    use_file = f
                    max_count = iter_count
            use_model_file = osp.join(search_dir, 'model_%i.pt' % max_count)

            use_cmd += ' --load-file "%s"' % use_model_file
            print('Executing ', use_cmd )
            tmp_name = 'tmp_' + run + '_' + str(uuid.uuid4()) + '.txt'

            os.system(use_cmd + ' > ' + tmp_name)
            mean_reward = -1.0
            with open(tmp_name, 'r') as f:
                for l in f.readlines():
                    if 'ep_goal_hit' in l:
                        mean_reward = l.split('ep_goal_hit: mean = ')[1].split(',')[0]
                        mean_reward = float(mean_reward.rstrip())
                        break
                    if 'Rendered frames' in l:
                        print('')
                        print(l)
                        print('')
            #os.remove(tmp_name)
            results[run] = mean_reward

        print('')
        print('-' * 20)
        print('Results')
        for run_name, mean_reward in results.items():
            print('%s: %.5f' % (run_name, mean_reward))

    else:
        raise ValueError()

