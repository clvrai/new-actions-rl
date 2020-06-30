import torch
import numpy as np
from torch.utils import data
import os
import os.path as osp
import time
from rlf.rl.utils import save_mp4
from rlf.baselines.common.tile_images import tile_images
from imageio import mimsave


class Option():
    def __init__(self, option_id):
        # Note: Environment just deals with option ids
        # Environment maintains an action bank itself
        self.option_id = option_id
        self.states = None
        self.actions = None
        self.data = None
        self.datapoint_size = None
        self.renderings = None

    def add(self, info, device):
        # obs is of shape num_processes * obs_shape
        states = []
        actions = []
        for x in info:
            states.append(x['states'])
            actions.append(x['actions'])

        states = torch.tensor(np.array(states)).to(device).float()
        actions = torch.tensor(np.array(actions)).to(device).float()

        if self.states is None:
            self.states = states
            self.actions = actions
        else:
            self.states = torch.cat([self.states, states], dim=0)
            self.actions = torch.cat([self.actions, actions], dim=0)

    def viz(self, viz_folder, option_id, trial, add_name, args, frames=None):
        if frames is None:
            frames = 255.0 * np.concatenate(
                self.states.cpu().numpy(), axis = 0).reshape(-1, 84, 84, 2)

            frames = np.expand_dims(frames[:,:,:,0], -1).repeat(3, axis=-1)

        #frames = self.renderings.cpu().numpy()
        #frames = frames[0]
        #frames = np.array([tile_images(frames[:, i]) for i in range(frames.shape[1])])

        save_name = 'option_video_%i_%i_%s' % (option_id, trial, add_name)
        save_mp4(frames, viz_folder, save_name,
                fps=5.0, no_frame_drop=True)
        print('saved to %s' % save_name)


    def viz_gif(self, viz_folder, option_id, trial, add_name, args, frames=None):

        res = args.image_resolution

        frames = self.states.view(-1, res, res, 3).cpu().numpy()
        frames = (frames + 0.5) * 255.0
        frames = frames.astype('uint8')

        save_name = 'option_video_%i_%s' % (option_id, add_name)

        mimsave(osp.join(viz_folder, save_name + '.gif'),
            frames,
            'GIF',
            duration=0.15)

        print('saved to %s' % save_name)


    def prepare_option_data(self, use_action_trajectory=False):
        if not use_action_trajectory:
            self.data = self.states
        else:
            # Concatenate (s,a) pairs. Note for the final state -> action = -1
            self.data = torch.cat([self.states, self.actions], dim=-1)

        self.datapoint_size = self.data.shape[-1]

        # To save space let's keep it on the CPU.
        self.data = self.data.cpu()
        del self.states
        del self.actions


    def save_dataset(self, data_folder, env_name, num_separations=1):
        new_dir = osp.join(data_folder, env_name)
        if not osp.exists(new_dir):
            os.makedirs(new_dir)
        new_file = osp.join(new_dir, 'option_' + str(self.option_id) + '.npy')
        np.save(new_file, self.data.cpu().numpy())
        return


    def load_data_params(self, data_folder, env_name):
        load_dir = osp.join(data_folder, env_name)
        opt_data = np.load(osp.join(
            load_dir, 'option_' + str(self.option_id) + '.npy'))
        return 1, opt_data.shape[0], opt_data.shape[1], opt_data.shape[-1]

        # load_dir = osp.join(data_folder, env_name + '_' + str(self.option_id))
        # files = os.listdir(load_dir)
        # data = np.load(osp.join(load_dir, files[0]))
        # # Note: datapoint_size is ignored for logic game (conv)
        # return len(files), data.shape[0], data.shape[1], data.shape[-1]


class OptionDataset(data.Dataset):
    def __init__(self, option_ids, data_folder,
            env_name, n_trajectories, args=None):

        self.args = args
        # Note: n_trajectories < #trajectories per file
        self.option_ids = option_ids
        self.data_folder = data_folder
        self.env_name = env_name
        self.n_trajectories = n_trajectories

        self.global_mean = None
        self.global_std = None

        if args.load_all_data:
            self.data = []
            self.opt_ids = []
            load_dir = osp.join(self.data_folder, self.env_name)
            for option_id in self.option_ids:
                file = osp.join(load_dir, 'option_' + str(option_id) + '.npy')
                opt_data = np.load(file)
                self.data.append(opt_data)
            self.data = np.array(self.data)
            if self.args.emb_method == 'tvae' and not self.args.test_embeddings:
                self.data = self.data.reshape(-1, *self.data.shape[2:])

            if 'BlockPlay' in self.env_name:
                self.data = self.data - 0.5

    def data_modifier(self, batch):
        if self.args.env_name.startswith('MiniGrid') and self.args.onehot_state:
            batch = np.eye(self.args.play_grid_size)[batch.astype(int)]
        return batch.astype(np.float32)


    def __getitem__(self, item):
        option_id = self.option_ids[item]

        if self.args.load_all_data:
            if self.args.emb_method == 'tvae' and not self.args.test_embeddings:
                choice_traj = np.random.choice(self.data.shape[0],
                    size = min(self.n_trajectories, self.data.shape[0]))
                return (option_id, self.data_modifier(self.data[choice_traj]))
            else:
                assert self.data[item].shape[0] >= self.n_trajectories - self.args.num_processes
                choice_traj = np.random.choice(self.data[item].shape[0],
                    size=min(self.n_trajectories, self.data[item].shape[0]) ,
                    replace=False)
                return (option_id, self.data_modifier(self.data[item][choice_traj]))
        else:
            load_dir = osp.join(self.data_folder, self.env_name)

            file = osp.join(load_dir, 'option_' + str(option_id) + '.npy')
            opt_data = np.load(file)
            if 'BlockPlay' in self.env_name:
                opt_data = opt_data - 0.5

            choice_traj = np.random.choice(opt_data.shape[0],
                size=min(self.n_trajectories, opt_data.shape[0]) ,
                replace=False)

            return (option_id, self.data_modifier(opt_data[choice_traj]))


    def __len__(self):
        return len(self.option_ids)


