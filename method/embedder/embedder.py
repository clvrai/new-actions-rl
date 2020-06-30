from method.embedder.utils import scatter_contexts
from sklearn.manifold import TSNE
from torch.utils import data
from rlf.rl.utils import save_mp4
from method.embedder.option import OptionDataset, Option
from method.embedder.tvae import TVAE
from method.embedder.htvae import HTVAE
from rlf.rl.envs import make_vec_envs
from gym.spaces import Discrete
import matplotlib as mpl
import imageio
from tqdm import tqdm
import os.path as osp
import shutil
import os
import random
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import gym
import sys
import os
import os.path as osp
from tqdm import tqdm
import pymunkoptions
pymunkoptions.options["debug"] = False
from rlf.rl import utils

from torch.utils import data

# Embedding Specific
from method.embedder.htvae import HTVAE

from method.embedder.vis import vis_embs
from method.radam import RAdam

from envs.gym_minigrid.action_sets import convert_gridworld, get_option_properties

sys.path.append('.')
sys.path.append('..')
sys.path.append('...')


mpl.use('Agg')


# Embedding Specific

def extract_dists(args, dist_mem, emb_mem, env_trans_fn,
                  real_name, load_all=False):
    if args.save_embeddings_file is None:
        if args.load_embeddings_file is None:
            return
        emb_mem.load_embeddings(args.load_embeddings_file)
        dist_mem.load_distributions(args.load_embeddings_file)  # Same Name
        if args.verify_embs:
            print('Verifying!')
            vis_embs(dist_mem, emb_mem, args.num_distributions, args.exp_type,
                     render=args.debug_render, save_prefix='verify', args=args,
                     use_idx=args.overall_aval_actions)
        return

    # NOTE: We might want to set Embedder.option_ids here
    log_dir = os.path.expanduser(args.log_dir)
    trial_log_dir = log_dir + "_trial"
    utils.cleanup_log_dir(trial_log_dir)

    if args.overall_aval_actions is not None:
        orig_aval_actions = args.overall_aval_actions[:]
    else:
        orig_aval_actions = None


    data_folder = args.play_data_folder
    model_folder = args.emb_model_folder
    embedder = Embedder(args, trial_log_dir, env_trans_fn, data_folder,
                        option_ids=args.overall_aval_actions)

    # Generate Test Data
    if not args.load_dataset:
        print('No data to load... Generating Dataset')
        embedder.generate_dataset(args, trial_log_dir, env_trans_fn)
    else:
        print('Loading data...')
        embedder.load_data_params()

    assert args.load_emb_model_file is not None
    # Only loading model in testing phase
    args.test_embeddings = True
    embedder.prepare_model(args, model_folder, None, method=args.emb_method)
    embedder.eval_dists_from_ids(dist_mem, None,
                                 dist_mem.n_distributions, load_all=load_all)
    embedder.eval_embs_from_ids(emb_mem, None,
                                load_all=load_all, dist_mem=dist_mem)

    if args.save_embeddings_file is not None:
        emb_mem.save_embeddings(args.save_embeddings_file)
        dist_mem.save_distributions(args.save_embeddings_file)  # Same Name
        print('Verifying!')
        vis_embs(dist_mem, emb_mem, args.num_distributions, args.exp_type,
                 render=args.debug_render, save_prefix='verify', args=args,
                 use_idx=orig_aval_actions)

    args.both_train_test = False


class Embedder():
    def __init__(self, args, trial_log_dir, env_trans_fn,
                 data_folder, option_ids=None):

        self.args = args
        self.data_folder = data_folder

        self.n_trajectories = args.n_trajectories
        if self.args.save_dataset:
            assert self.n_trajectories >= self.args.num_processes
            self.n_trials = self.n_trajectories // self.args.num_processes
            self.n_trajectories = self.n_trials * self.args.num_processes
            self.args.n_trajectories = self.n_trajectories
        args.device = torch.device("cuda:0" if args.cuda else "cpu")

        # TODO: Extract this trial environment parameters based on args.env_name from
        # some other file
        self.trial_env_name = args.env_name

        if option_ids is None:
            assert option_ids is not None
            envs = make_vec_envs(self.trial_env_name,
                             args.seed, 1,
                             None, trial_log_dir, self.args.device, True,
                             env_trans_fn, self.args)
            self.total_options = envs.action_space.n
            self.option_ids = list(range(self.total_options))
        else:
            self.option_ids = option_ids

        self.conv = False

        if args.emb_method == 'tvae' and args.load_all_data and not args.test_embeddings:
            self.batch_size = len(self.option_ids)
            self.sample_size = (args.emb_batch_size * args.n_trajectories) // self.batch_size
        else:
            self.batch_size = args.emb_batch_size
            self.sample_size = args.n_trajectories

        if args.env_name.startswith('StateCreate') or args.env_name.startswith('MiniGrid'):
            self.conv = False
        elif args.env_name.startswith('Create') or args.env_name.startswith('Block'):
            self.conv = True
        else:
            raise NotImplementedError


    def generate_dataset(self, args, trial_log_dir, env_trans_fn):
        '''
            Generating trajectory dataset for given option_ids
        '''
        print('Generating dataset for options')
        envs = make_vec_envs(self.trial_env_name,
                             args.seed, args.num_processes,
                             None, trial_log_dir, args.device, True, env_trans_fn, args)

        use_option_ids = self.option_ids

        labels, _ = args.env_interface.get_env_option_names()

        def get_opt_name(x): return labels[x]

        if args.start_option is not None:
            use_option_ids = use_option_ids[args.start_option:]

        debug_render = args.debug_render
        data_vis_folder = './method/embedder/data_visualization'
        gif_vis_folder = './method/embedder/gif_visualization'

        if debug_render:
            print('Clearing vis folder')
            if osp.exists(data_vis_folder):
                shutil.rmtree(data_vis_folder)
            os.makedirs(data_vis_folder)
            temp_env = gym.make(args.env_name)

            if 'MiniGrid' in args.env_name:
                temp_env = convert_gridworld(temp_env, args)
            elif 'Create' in args.env_name:
                temp_env.update_args(args)
        elif args.render_gifs:
            print('Clearing gif vis folder')
            if osp.exists(gif_vis_folder):
                shutil.rmtree(gif_vis_folder)
            os.makedirs(gif_vis_folder)
            temp_env = gym.make(args.env_name)

            if 'MiniGrid' in args.env_name:
                temp_env = convert_gridworld(temp_env, args)
            elif 'Create' in args.env_name:
                temp_env.update_args(args)

        for option_id in tqdm(use_option_ids):
            if debug_render:
                option_id = random.randint(0, len(use_option_ids))
                #self.n_trials = 1

            opt_data = Option(option_id)

            option_tensor = torch.tensor(option_id).to(args.device)

            if self.args.cuda:
                option_tensor = option_tensor.repeat(self.args.num_processes)
            else:
                option_tensor = option_tensor.repeat(
                    [self.args.num_processes, 1])

            tool_type = get_opt_name(option_id)

            print('running trials for option %i a %s' % (option_id, tool_type))

            for trial in range(self.n_trials):
                _ = envs.reset()
                _, _, _, info = envs.step(option_tensor)
                opt_data.add(info, args.device)

                if trial < 7 and debug_render:
                    if len(info[0]['states'].shape) == 3:
                        save_file = osp.join(
                            data_vis_folder, '%i_%i.png' % (option_id, trial))
                        imageio.imwrite(save_file, (info[0]['states'] * 255).astype('uint8'))
                        print('saved %s' % save_file)
                    else:
                        frames = temp_env.render_obs(
                            info[0]['states'], option_id)
                        opt_data.viz(data_vis_folder, option_id,
                                     trial, tool_type, args, frames)

                if (trial + 1) % 2 == 0 and self.args.render_gifs:
                    frames = temp_env.render_obs(
                        info[0]['states'], option_id)
                    opt_data.viz_gif(gif_vis_folder, option_id,
                                 trial, tool_type, args)
                    break

            opt_data.prepare_option_data(self.args.use_action_trajectory)

            if not args.debug_render:
                opt_data.save_dataset(self.data_folder,
                                      self.trial_env_name, num_separations=1)

            # save the environment normalization factor stuff

        # ob_rms = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        # with open(osp.join(self.data_folder, self.trial_env_name + '_ob_rms.dat'), 'wb') as f:
        #     pickle.dump(ob_rms, f)

        self.datapoint_size = self.args.play_grid_size

    def load_data_params(self):
        opt_data = Option(self.option_ids[0])
        params = opt_data.load_data_params(
            self.data_folder, self.trial_env_name)
        num_separations, self.n_trajectories, \
            self.args.trajectory_len, self.datapoint_size = params
        self.args.n_trajectories = self.n_trajectories

    def visualize(self, model, test_loader=None, test_dataset=None, epoch=-1):
        pass

    def train(self, args, model, optimizer, scheduler, train_loader, train_dataset, model_folder,
              logger, start_epoch=0):
        viz_interval = args.emb_epochs if args.emb_viz_interval == - \
            1 else args.emb_viz_interval
        save_interval = args.emb_epochs if args.emb_save_interval == - \
            1 else args.emb_save_interval

        alpha = 1
        tbar = tqdm(range(start_epoch, args.emb_epochs))

        # main training loop
        for epoch in tbar:
            # train step
            model.train()

            running_vlb = 0
            running_loss = 0
            running_log_likelihood = 0
            running_l2_loss = 0
            running_kl = 0
            for batch in train_loader:
                opt_id, batch = batch

                if args.cuda:
                    batch = batch.cuda()

                vlb, loss, log_likelihood, l2_loss, kl, x_sample, x_mean = \
                    model.step(batch, alpha, optimizer, scheduler,
                               clip_gradients=args.htvae_clip_gradients)
                running_vlb += vlb
                running_loss += loss
                running_log_likelihood += log_likelihood
                running_l2_loss += l2_loss
                running_kl += kl

            running_vlb /= (len(train_dataset) // self.batch_size)
            running_loss /= (len(train_dataset) // self.batch_size)
            running_log_likelihood /= (len(train_dataset) //
                                       self.batch_size)
            running_l2_loss /= (len(train_dataset) // self.batch_size)
            running_kl /= (len(train_dataset) // self.batch_size)

            s = "VLB: {:.3f}".format(running_vlb)
            tbar.set_description(s)

            # reduce weight
            alpha *= 0.5
            # alpha *= 0.99

            if (epoch + 1) % args.emb_log_interval == 0:
                logger.write_scalar('data/running_vlb', running_vlb, epoch)
                logger.write_scalar('data/running_loss', running_loss, epoch)
                logger.write_scalar('data/running_log_likelihood',
                                    running_log_likelihood, epoch)
                logger.write_scalar('data/running_l2_loss',
                                    running_l2_loss, epoch)
                logger.write_scalar('data/running_kl', running_kl, epoch)
                logger.write_scalar('params/learning_rate', float(optimizer.param_groups[0]['lr']), epoch)
            if (epoch+1) % save_interval == 0 and self.conv:
                # visualize
                input_img = (x_sample.data.cpu().numpy().reshape(-1, self.args.image_resolution,
                                                                 self.args.image_resolution, self.args.input_channels) + 0.5) * 255.0
                output = (x_mean.data.cpu().numpy().reshape(-1, self.args.image_resolution,
                                                            self.args.image_resolution, self.args.input_channels) + 0.5) * 255.0
                combs = []
                for i in range(len(input_img)):
                    inp = input_img[i][:, :, :]
                    outp = output[i][:, :, :]
                    comb = np.concatenate([inp, outp], axis=1)
                    if comb.shape[-1] != 3:
                        comb = comb.repeat(repeats=3, axis=-1)
                    combs.append(comb)
                print("created ", epoch + 1)
                viz_folder = osp.join(
                    './method/embedder/training_vids', self.args.env_name, self.args.save_emb_model_file)
                if not osp.exists(viz_folder):
                    os.makedirs(viz_folder)
                save_mp4(np.array(combs), viz_folder,
                         self.args.save_emb_model_file+'_'+str(epoch + 1), fps=5.0, no_frame_drop=True)

            elif (epoch + 1) % save_interval == 0:
                # visualize
                env = gym.make(args.env_name)
                if 'MiniGrid' in args.env_name:
                    env = convert_gridworld(env, args)
                elif 'Create' in args.env_name:
                    env.update_args(args)

                input_img = []
                output_img = []

                for _ in range(args.num_eval):
                    i = random.randint(0, x_mean.shape[0] - 1)
                    j = random.randint(0, x_mean.shape[1] - 1)

                    input_obs = x_sample[i][j].data.cpu().numpy()
                    output = x_mean[i][j].data.cpu().numpy()
                    use_opt_id = int(opt_id[i].item())

                    input_img.extend(env.render_obs(input_obs, use_opt_id))
                    output_img.extend(env.render_obs(output, use_opt_id))

                combs = []
                for i in range(len(input_img)):
                    inp = input_img[i]
                    outp = output_img[i]
                    comb = np.concatenate([inp, outp], axis=1)
                    combs.append(comb)
                print("created ", epoch + 1)
                viz_folder = osp.join(
                    './method/embedder/training_vids', self.args.env_name, self.args.save_emb_model_file)
                if not osp.exists(viz_folder):
                    os.makedirs(viz_folder)
                save_mp4(np.array(combs), viz_folder,
                         self.args.save_emb_model_file+'_'+str(epoch + 1), fps=5.0, no_frame_drop=True)

            if (epoch + 1) % save_interval == 0:
                assert args.save_emb_model_file is not None
                print('Saving model after ' + str(epoch + 1) + ' epochs')
                save_path = osp.join(model_folder,
                                     args.save_emb_model_file +
                                     '-' + args.emb_method + '-' + str(epoch + 1) + '.m')
                if not osp.exists(model_folder):
                    os.makedirs(model_folder)

                model.save(optimizer, scheduler, save_path, epoch)
                print('Saved to ' + save_path)

    def prepare_model(self, args, model_folder, logger, method='htvae'):

        ################# Set up Model #################
        # Dimensionality of state space
        if self.args.env_name.startswith('MiniGrid') and self.args.onehot_state:
            n_features = 2 * self.args.play_grid_size
        else:
            n_features = self.datapoint_size

        # Model dimensions
        model_kwargs = {
            'batch_size': self.batch_size,
            'sample_size': self.sample_size,
            'n_features': n_features,
            'o_dim': args.o_dim,
            'n_hidden_option': args.n_hidden_option,
            'hidden_dim_option': args.hidden_dim_option,
            'n_stochastic': args.n_stochastic,
            'z_dim': args.z_dim,
            'n_hidden': args.n_hidden_traj,
            'hidden_dim': args.hidden_dim_traj,
            'encoder_dim': args.encoder_dim,
            'nonlinearity': F.relu,
            'print_vars': args.print_vars,
            'is_cuda': args.cuda,
            'conv': self.conv,
            'input_channels': args.input_channels,
            'args': args
        }

        if method == 'htvae':
            self.model = HTVAE(**model_kwargs)
        elif method == 'tvae':
            self.model = TVAE(**model_kwargs)

        if args.cuda:
            self.model.cuda()

        if args.emb_use_radam:
            optimizer = RAdam(self.model.parameters(),
                                   lr=args.emb_learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=args.emb_learning_rate)

        if args.emb_schedule_lr:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.7)
        else:
            scheduler = None

        ################# Load/Train Model #################
        if args.train_embeddings or args.resume_emb_training:
            if args.resume_emb_training:
                assert args.load_emb_model_file is not None
                print('Continuing training from loaded model')
                load_path = osp.join(model_folder, args.load_emb_model_file)
                optimizer, scheduler, start_epoch = self.model.load(optimizer, scheduler, load_path)
            else:
                print('No trained model provided... Starting training')
                start_epoch = 0

            # For TVAE set dataset parameters to load varied trajectories
            train_dataset = OptionDataset(self.option_ids,
                                          self.data_folder, self.trial_env_name, self.sample_size, args=self.args)
            train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True, num_workers=10, drop_last=True)

            self.train(args, self.model, optimizer, scheduler, train_loader,
                       train_dataset, model_folder, logger, start_epoch)

        if args.test_embeddings and args.load_emb_model_file is not None:
            print('Found trained model to load!')
            load_path = osp.join(model_folder, args.load_emb_model_file)
            self.model.load(optimizer, scheduler, load_path)

    def get_data_loader(self, args):
        print(self.option_ids)
        train_dataset = OptionDataset(self.option_ids,
                                      self.data_folder, self.trial_env_name, self.sample_size, args=self.args)
        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True, num_workers=4, drop_last=True)
        return train_loader

    def get_option_embedding_from_data(self, test_loader):
        self.model.eval()
        option_embs = []
        option_emb_logvars = []
        with torch.no_grad():
            for batch in test_loader:
                opt_id, batch = batch
                opt_id = int(opt_id[0].item())
                inputs = batch
                if self.args.cuda:
                    inputs = inputs.cuda()
                self.model.set_batch_size(inputs.shape[0], inputs.shape[1])
                emb_means, emb_logvars = self.model.get_option_embedding(
                    inputs)
                option_embs.append(np.squeeze(emb_means.data.cpu().numpy()))
                option_emb_logvars.append(np.squeeze(emb_logvars.data.cpu().numpy()))
        return np.array(option_embs), np.array(option_emb_logvars)

    def eval_embs_from_ids(self, emb_mem, remap_options=None, load_all=False, dist_mem=None):
        # load in all of the options.
        # all_opt_ids = [int(x.split('_')[1].split('.')[0]) for x in
        #         os.listdir(osp.join(self.data_folder, self.trial_env_name))]
        # if load_all:
        #     use_opt_ids = all_opt_ids
        # else:
        #     use_opt_ids = self.option_ids

        use_opt_ids = self.option_ids

        eval_dataset = OptionDataset(
            use_opt_ids, self.data_folder,
            self.trial_env_name, self.sample_size, args=self.args)

        eval_loader = data.DataLoader(dataset=eval_dataset,
                                      batch_size=1, shuffle=False, num_workers=4, drop_last=False)

        # Get embeddings from model
        if dist_mem is not None and self.args.emb_method == 'tvae':
            # Needed for policy input for tvae embeddings
            option_embs = dist_mem.mem_keys.narrow(
                1, 0, 1).squeeze(1).cpu().numpy()
            option_emb_logvars = np.copy(option_embs)
            option_emb_logvars[:] = 0.
        else:
            option_embs, option_emb_logvars = self.get_option_embedding_from_data(
                eval_loader)

        # Store embeddings
        for (i, opt) in enumerate(use_opt_ids):
            # Get the option ID that should be used as input to the environment
            if remap_options is not None:
                if opt not in remap_options:
                    continue
                opt = remap_options[opt]
            emb_mem.add_embedding(option_embs[i], opt)
            emb_mem.add_emb_logvar(option_emb_logvars[i], opt)

    def eval_dists_from_ids(self, dist_mem, remap_options=None,
                            n_distributions=1, load_all=False):
        # if self.args.emb_method == 'htvae':
        #     dist_mem.store_model(self.model)
        #     return

        # # load in all of the options.
        # all_opt_ids = [int(x.split('_')[1].split('.')[0]) for x in
        #         os.listdir(osp.join(self.data_folder, self.trial_env_name))]
        # import pdb; pdb.set_trace()
        # if load_all:
        #     use_opt_ids = all_opt_ids
        # else:
        #     use_opt_ids = self.option_ids

        use_opt_ids = self.option_ids
        eval_dataset = OptionDataset(
            use_opt_ids, self.data_folder,
            self.trial_env_name, self.sample_size, args=self.args)

        eval_loader = data.DataLoader(dataset=eval_dataset,
                                      batch_size=1, shuffle=False, num_workers=1, drop_last=False)

        # Get embeddings from model
        o_means, o_logvars = self.get_option_distributions_from_data(eval_loader,
                                                                     n_distributions)
        # Store embeddings
        for (i, opt) in enumerate(use_opt_ids):
            if remap_options is not None:
                if opt not in remap_options:
                    continue
                opt = remap_options[opt]
            dist_mem.add_distribution(np.array([o_means[i], o_logvars[i]]),
                                      opt)

    def get_option_distributions_from_data(self, test_loader, n_distributions=1):
        self.model.eval()
        z_means = []
        z_logvars = []
        with torch.no_grad():
            for batch in test_loader:
                opt_id, batch = batch
                opt_id = int(opt_id[0].item())
                inputs = batch
                if self.args.cuda:
                    inputs = inputs.cuda()
                self.model.set_batch_size(inputs.shape[0], inputs.shape[1])
                z_mean, z_logvar = self.model.get_option_distributions(
                    inputs, n_distributions)

                z_means.append(np.squeeze(z_mean.data.cpu().numpy()))
                z_logvars.append(np.squeeze(z_logvar.data.cpu().numpy()))

        return z_means, z_logvars

    def get_trajectory_embedding_from_data(self, test_loader, emb_mem):
        self.model.eval()
        traj_embs = None
        option_labels = None
        with torch.no_grad():
            label = 0
            for batch in test_loader:
                opt_id, batch = batch
                opt_id = int(opt_id[0].item())
                option_emb = emb_mem.mem_keys[opt_id]
                # Now batch contains many trajectories from a single option
                for ind in range(batch.shape[1]):
                    inputs = batch[0][ind].unsqueeze(0).unsqueeze(1)
                    if self.args.cuda:
                        inputs = inputs.cuda()
                    self.model.set_batch_size(inputs.shape[0], inputs.shape[1])

                    z = self.model.get_trajectory_embedding(inputs, option_emb)

                    if traj_embs is None:
                        traj_embs = z.data.cpu().numpy()
                        option_labels = np.array([str(label)])
                    else:
                        traj_embs = np.concatenate((traj_embs,
                                                    z.data.cpu().numpy()), axis=0)
                        option_labels = np.concatenate((option_labels,
                                                        np.array([str(label)])), axis=0)

                label += 1

        return traj_embs, option_labels

    def get_traj_embs(self, loader, emb_mem):
        self.model.eval()
        traj_embs = []
        traj_opt_ids = []

        with torch.no_grad():
            label = 0
            for batch in tqdm(loader):
                opt_id, batch = batch
                opt_id = opt_id.numpy()

                real_inds = [emb_mem.get_for_real_ind(oi) for oi in opt_id]

                batch = batch.cuda()

                option_embs = emb_mem.mem_keys[real_inds]
                self.model.set_batch_size(batch.shape[0], batch.shape[1])

                z = self.model.get_trajectory_embedding(batch, option_embs)
                traj_embs.append(z.cpu().numpy())
                traj_opt_ids.append(np.repeat(real_inds, batch.shape[1]))

        traj_embs, traj_opt_ids = np.array(traj_embs), np.array(traj_opt_ids)
        return traj_embs.reshape(-1, self.args.z_dim), traj_opt_ids.reshape(-1)

    def save_embeddings(self, save_file, emb_mem, dist_mem):
        pass

    def load_embeddings(self, load_file, emb_mem, dist_mem):
        pass

    def visualize_reconstructions(self, viz_folder, test_loader, emb_mem):
        self.model.eval()
        inputs = []
        outputs = []
        option_labels = []
        env = gym.make(self.args.env_name)
        if 'MiniGrid' in self.args.env_name:
            env = convert_gridworld(env, self.args)
        elif 'Create' in self.args.env_name:
            env.update_args(self.args)

        ind = 0
        with torch.no_grad():
            for batch in test_loader:
                if self.args.env_name.startswith('StateCreate') or self.args.env_name.startswith('MiniGrid'):
                    opt_id, batch = batch
                    opt_id = int(opt_id[0].item())
                option_emb = emb_mem.mem_keys[ind]
                if self.args.cuda:
                    batch = batch.cuda()
                self.model.set_batch_size(batch.shape[0], batch.shape[1])

                output = self.model.get_reconstruction(batch, option_emb)

                if self.args.env_name.startswith('MiniGrid'):
                    pass
                    # import pdb; pdb.set_trace()
                    # env.agent_view_size = int(np.sqrt(input_img.shape[-1]/3))
                    # input_img = (batch.data.cpu().numpy()[:,:,1:,:].squeeze(0) * 255. + 128.).round().astype(int)
                    # output = (output.data.cpu().numpy().squeeze(0) * 255. + 128.).round().astype(int)
                    # combs = []
                    # for i in range(len(input_img)):
                    #     for j in range(len(input_img[i])):
                    #         inp = env.get_obs_render(input_img[i][j].reshape(
                    #                 int(np.sqrt(input_img.shape[-1]/3)),
                    #                 int(np.sqrt(input_img.shape[-1]/3)),
                    #                 3))
                    #         outp = env.get_obs_render(output[i][j].reshape(
                    #                 int(np.sqrt(output.shape[-1]/3)),
                    #                 int(np.sqrt(output.shape[-1]/3)),
                    #                 3))

                    #         comb = np.concatenate([inp, outp], axis=0)
                    #         combs.append(comb)
                    # print("created ", ind)
                    # save_mp4(np.array(combs), viz_folder, str(ind), True)
                    # ind += 1

                if self.args.env_name.startswith('Create'):
                    input_img = batch.data.cpu().numpy()[:, :, 1:, :].reshape(
                        -1, self.args.image_resolution, self.args.image_resolution, self.args.input_channels) * 255.0
                    output = output.data.cpu().numpy().reshape(-1, self.args.image_resolution,
                                                               self.args.image_resolution, self.args.input_channels) * 255.0
                    combs = []
                    for i in range(len(input_img)):
                        inp = input_img[i][:, :, 0]
                        outp = output[i][:, :, 0]
                        comb = np.concatenate([inp, outp], axis=1)
                        comb = np.expand_dims(
                            comb, -1).repeat(repeats=3, axis=-1)
                        combs.append(comb)
                    print("created ", ind)
                    save_mp4(np.array(combs), viz_folder,
                             self.args.env_name+str(ind), fps=5.0, no_frame_drop=True)
                    ind += 1

    def visualize_trajectory_embeddings(self, method='htvae',
                                        reconstruction=False, emb_mem=None, skip_tsne=False, dist_mem=None):

        # option_ids = {(x, str(x)) for x in self.option_ids}
        option_ids = self.option_ids

        if dist_mem is not None:
            means = dist_mem.mem_keys.select(1, 0)
            logvars = dist_mem.mem_keys.select(1, 1)
            opt_embs = emb_mem.mem_keys
            if self.args.num_distributions is not None:
                # Limit the number of distributions
                random.seed(self.args.seed)
                select_indices = random.sample(list(range(means.shape[0])),
                                               self.args.num_distributions)
                means = means[select_indices]
                logvars = logvars[select_indices]
                option_ids = np.array(option_ids)[select_indices]
                opt_embs = opt_embs[select_indices]

            vis_embs(dist_mem, emb_mem, self.args.num_distributions,
                     self.args.exp_type, True,
                     self.args.load_emb_model_file, self.args)
            return

        import ipdb; ipdb.set_trace()
        # N1 is not supported anymore for CREATE

        n_traj_per_option = 1
        if 'N1' in self.trial_env_name:
            n_traj_per_option = 1
        elif self.trial_env_name.startswith('CreateGamePlay'):
            n_traj_per_option = self.args.n_trajectories
        elif 'create' in str.lower(self.trial_env_name):
            n_traj_per_option = 190

        eval_dataset = OptionDataset(
            self.option_ids, self.data_folder,
            self.trial_env_name, n_trajectories=n_traj_per_option, args=self.args)

        # Now we only load one option and one trajectory
        eval_loader = data.DataLoader(dataset=eval_dataset,
                                      batch_size=1,
                                      shuffle=False, num_workers=1, drop_last=False)

        if reconstruction:
            self.visualize_reconstructions(viz_folder + 'recon/',
                                           eval_loader,
                                           emb_mem)

        # Get trajectory embeddings from the model
        traj_embs, option_labels = self.get_trajectory_embedding_from_data(
            eval_loader,
            emb_mem)
        label_list = [str(x) for x in option_ids]

        if self.trial_env_name.startswith('StateCreate'):
            new_lab = {'0': 'floor',
                       '1': 'floor',
                       '2': 'floor',
                       '3': 'trampoline',
                       '4': 'trampoline',
                       '5': 'bump_ramp',
                       '6': 'bump_ramp',
                       '7': 'bump_ramp',
                       '8': 'ramp',
                       '9': 'ramp',
                       '10': 'ramp',
                       '11': 'wall',
                       '12': 'no_op',
                       '13': 'trampoline',
                       '14': 'bump_ramp',
                       '15': 'bump_ramp',
                       '16': 'bump_ramp',
                       '17': 'ramp',
                       '18': 'ramp',
                       '19': 'ramp'
                       }

        if emb_mem is not None and self.args.o_dim in [2, 3]:
            if self.trial_env_name.startswith('MiniGrid'):
                option_properties, label_list = get_option_properties(
                    self.args, quadrant=True)

                emb_mem.visualize_embeddings(viz_folder, option_properties,
                                             label_list, label_type='n_steps')
            elif self.trial_env_name.startswith('StateCreate'):
                option_labels = np.array(
                    [new_lab[opt] for opt in option_labels])
                label_list = ['floor', 'ramp', 'reverse_ramp',
                              'wall', 'trampoline', 'no_op']
                emb_mem.visualize_embeddings(
                    viz_folder, option_labels, label_list)

        if not skip_tsne:
            perplexity = 10
            tsne = TSNE(n_components=2, perplexity=perplexity, verbose=1)
            tsne2 = TSNE(n_components=3, perplexity=perplexity, verbose=1)

            tsne_results = tsne.fit_transform(traj_embs)

        if self.args.z_dim in [2, 3]:
            tsne2_results = traj_embs
        else:
            tsne2_results = tsne2.fit_transform(traj_embs)

        if 'grid' in str.lower(self.trial_env_name):
            perplexity = 30
            # Quadrants
            option_properties, label_list = get_option_properties(self.args,
                                                                  quadrant=True)
            quad_labels = []
            for i in range(len(option_labels)):
                quad_labels.append(option_properties[int(option_labels[i])])
            quad_labels = np.array(quad_labels)

            if not skip_tsne:
                path = osp.join(viz_folder, self.trial_env_name +
                                method + '_traj_quadrant.pdf')
                scatter_contexts(tsne_results, quad_labels,
                                 label_list, savepath=path)
                path2 = osp.join(viz_folder, self.trial_env_name +
                                 method + '_traj_quadrant3D.pdf')
                scatter_contexts(tsne2_results, quad_labels,
                                 label_list, savepath=path2)

            # Octants
            option_properties, label_list = get_option_properties(self.args,
                                                                  octant=True)
            oct_labels = []
            for i in range(len(option_labels)):
                oct_labels.append(option_properties[int(option_labels[i])])

            if not skip_tsne:
                oct_labels = np.array(oct_labels)
                path = osp.join(viz_folder, self.trial_env_name +
                                method + '_traj_octant.pdf')
                scatter_contexts(tsne_results, oct_labels,
                                 label_list, savepath=path)
                path2 = osp.join(viz_folder, self.trial_env_name +
                                 method + '_traj_octant3D.pdf')
                scatter_contexts(tsne2_results, oct_labels,
                                 label_list, savepath=path2)

            # distance
            option_properties, label_list = get_option_properties(self.args,
                                                                  distance=True)
            dist_labels = []
            for i in range(len(option_labels)):
                dist_labels.append(option_properties[int(option_labels[i])])

            if not skip_tsne:
                dist_labels = np.array(dist_labels)
                path = osp.join(viz_folder, self.trial_env_name +
                                method + '_traj_distance.pdf')
                scatter_contexts(tsne_results, dist_labels,
                                 label_list, savepath=path)
                path2 = osp.join(viz_folder, self.trial_env_name +
                                 method + '_traj_distance3D.pdf')
                scatter_contexts(tsne2_results, dist_labels,
                                 label_list, savepath=path2)

            # man_distance
            option_properties, label_list = get_option_properties(self.args,
                                                                  man_distance=True)
            man_labels = []
            for i in range(len(option_labels)):
                man_labels.append(option_properties[int(option_labels[i])])
            man_labels = np.array(man_labels)

            if not skip_tsne:
                path = osp.join(viz_folder, self.trial_env_name +
                                method + '_traj_man_distance.pdf')
                scatter_contexts(tsne_results, man_labels,
                                 label_list, savepath=path)
                path2 = osp.join(viz_folder, self.trial_env_name +
                                 method + '_traj_man_distance3D.pdf')
                scatter_contexts(tsne2_results, man_labels,
                                 label_list, savepath=path2)

            vec_name = osp.join(
                viz_folder, self.trial_env_name + method + '_vec.tsv')
            attrib_name = osp.join(
                viz_folder, self.trial_env_name + method + '_attrib.tsv')

            with open(vec_name, 'w') as f:
                for i in range(len(traj_embs)):
                    p = [str(x) for x in traj_embs[i]]

                    f.write('\t'.join(p))
                    f.write('\n')

            with open(attrib_name, 'w') as f:
                f.write('Quad\tOct\tDist\tMan\n')
                for i in range(len(traj_embs)):
                    f.write('\t'.join([quad_labels[i], oct_labels[i],
                                       dist_labels[i], man_labels[i]]))
                    f.write('\n')

        else:
            # show coloured by labels
            path = osp.join(viz_folder, self.trial_env_name +
                            method + '_traj.pdf')
            scatter_contexts(tsne_results, option_labels,
                             label_list, savepath=path)
            path2 = osp.join(viz_folder, self.trial_env_name +
                             method + '_traj3D.pdf')
            scatter_contexts(tsne2_results, option_labels,
                             label_list, savepath=path2)

            # Cluster Tools
            if self.trial_env_name.startswith('StateCreate') or self.trial_env_name.startswith('Create'):

                option_labels = np.array(
                    [new_lab[opt] for opt in option_labels])
                label_list = ['floor', 'ramp', 'reverse_ramp',
                              'wall', 'trampoline', 'no_op']

                # show coloured by labels
                path = osp.join(viz_folder, self.trial_env_name +
                                method + '_traj_tool.pdf')
                scatter_contexts(tsne_results, option_labels,
                                 label_list, savepath=path)
                path2 = osp.join(
                    viz_folder, self.trial_env_name + method + '_traj3D_tool.pdf')
                scatter_contexts(tsne2_results, option_labels,
                                 label_list, savepath=path2)

                # for i, batch in enumerate(eval_loader):
                #     # Not Tested
                #     frames = 255.0 * np.concatenate(batch.cpu().numpy(), axis = 0).reshape(-1, self.args.image_resolution, self.args.image_resolution, 2)
                #     frames = np.expand_dims(frames[:,:,:,0], -1).repeat(3, axis=-1)
                #     frames = np.concatenate(frames, axis=0)

                #     save_mp4(frames, viz_folder, 'create_video_' + str(i), fps=5.0)
