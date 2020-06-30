import glob
import os
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import moviepy.editor as mpy
from rlf.rl.envs import VecNormalize
from PIL import Image

# Get a render frame function (Mainly for transition)
def get_render_frame_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].unwrapped.render_frame
    elif hasattr(venv, 'venv'):
        return get_render_frame_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_frame_func(venv.env)

    return None

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None



# def get_vec_normalize2(venv):
#     from transition.ppo.envs import VecNormalize
#     if isinstance(venv, VecNormalize):
#         return venv
#     elif hasattr(venv, 'venv'):
#         return get_vec_normalize2(venv.venv)

#     return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    final_lr = initial_lr * 0.1
    init_lr = (initial_lr - final_lr)
    lr = final_lr + init_lr - (init_lr * (min(1., epoch / (float(total_num_epochs)))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def save_mp4(frames, vid_dir, name, fps=60.0, no_frame_drop=False):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if no_frame_drop:
        def f(t):
            idx = min(int(t * fps), len(frames)-1)
            return frames[idx]

        if not osp.exists(vid_dir):
            os.makedirs(vid_dir)


        vid_file = osp.join(vid_dir, name + '.mp4')
        if osp.exists(vid_file):
            os.remove(vid_file)

        video = mpy.VideoClip(f, duration=len(frames)/fps)
        video.write_videofile(vid_file, fps, verbose=False, logger=None)

    else:
        drop_frame = 1.5
        def f(t):
            frame_length = len(frames)
            new_fps = 1./(1./fps + 1./frame_length)
            idx = min(int(t*new_fps), frame_length-1)
            return frames[int(drop_frame*idx)]

        if not osp.exists(vid_dir):
            os.makedirs(vid_dir)


        vid_file = osp.join(vid_dir, name + '.mp4')
        if osp.exists(vid_file):
            os.remove(vid_file)

        video = mpy.VideoClip(f, duration=len(frames)/fps/drop_frame)
        video.write_videofile(vid_file, fps, verbose=False, logger=None)

def render_obs(obs, obs_name, args):
    if not osp.exists(args.obs_dir):
        os.makedirs(args.obs_dir)

    for i in range(obs.shape[1]):
        o = obs.cpu().numpy()[0, i]
        o = np.stack((o,) * 3, axis=-1)
        if o.dtype == np.float32:
            o = (o * 255.0).astype('uint8')
        Image.fromarray(o).save(osp.join(args.obs_dir, '%s_%i.jpeg' % (obs_name, i)))
