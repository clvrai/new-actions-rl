from moviepy.editor import VideoFileClip
import moviepy.editor as mpy
import os
import os.path as osp
import numpy as np
import cv2
import shutil
import imageio
from tqdm import tqdm

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

SAVE_DIR = '/home/aszot/result-output/'
if osp.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
LOAD_DIR = '/home/aszot/ss-result-vids/'

def save_vid_clip(path):
    clip = VideoFileClip(path)
    clip_dat = np.array([x for x in clip.iter_frames()]).astype(np.uint8)
    name = path.split('/')[-1]

    task = name.split('-')[1]
    case = name.split('-')[-1].split('.')[0]
    mode = name.split('-')[4].split('_')[-1]

    if task == 'GW':
        put_title = 'Grid World'
        mode = name.split('_')[-1].split('.')[0]
    elif task == 'BS':
        put_title = 'Shape Stacking'
        mode = name.split('_')[-1].split('.')[0]
    else:
        put_title = 'CREATE ' + task

    if case == 'success':
        put_subtitle = ' Success Examples'
    elif case == 'failure':
        put_subtitle = ' Failure Examples'
    else:
        case = 'reg'
        put_subtitle = ''

    if mode == 'test':
        put_subtitle = 'Generalization' + put_subtitle
    else:
        put_subtitle = 'Training' + put_subtitle


    title_frames = []
    s = clip_dat.shape[1]
    num_seconds = 3
    fps = 10

    for _ in range(int(num_seconds * fps)):
        f = np.zeros(clip_dat.shape[1:]).astype(np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = clip_dat.shape[1] / 1024

        textsize = cv2.getTextSize(put_title, font, 3*scale, 2)[0]
        center = ((f.shape[1] - textsize[0]) // 2, (f.shape[0] - textsize[1]) // 2)

        offset_y = int(0.1 * clip_dat.shape[1])
        cv2.putText(f, put_title,
                (center[0], center[1] - offset_y),
                font, 3*scale, (255, 255, 255), 2, cv2.LINE_AA)

        textsize = cv2.getTextSize(put_subtitle, font, 1.5*scale, 1)[0]
        center = ((f.shape[1] - textsize[0]) // 2, (f.shape[0] - textsize[1]) // 2)
        cv2.putText(f, put_subtitle, (center[0], center[1] + offset_y),
                font, 1.5*scale, (255, 255, 255), 1, cv2.LINE_AA)

        render_txt = 'Generalization to New Actions in Reinforcement Learning'
        textsize = cv2.getTextSize(render_txt, font, 1*scale, 2)[0]

        cv2.putText(f, render_txt, ((f.shape[1] - textsize[0]) // 2, center[1] + (4*offset_y)),
                font, 1*scale, (255, 255, 255), 1, cv2.LINE_AA)

        title_frames.append(f.astype(np.uint8))
    title_frames = np.array(title_frames)

    total_clip_dat = np.concatenate([title_frames, clip_dat], axis=0).astype(np.uint8)
    #total_clip_dat = total_clip_dat[:1500]
    #save_name = task + '-' + mode + '-' + case
    save_name = name.split('.')[0] + '_title'
    print('Trying to save', total_clip_dat.shape)
    save_path = osp.join(SAVE_DIR, save_name + '.mp4')
    imageio.mimwrite(save_path, total_clip_dat, fps=fps)
    print('saved ', save_path)
    #save_mp4(total_clip_dat, SAVE_DIR, save_name)
    #print('saved ', save_name)

load_files = list(os.listdir(LOAD_DIR))
sorted(load_files)
for i, f in tqdm(enumerate(load_files)):
    print('%i for %s' % (i, f))
    save_vid_clip(osp.join(LOAD_DIR, f))




