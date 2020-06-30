'''
    Hierarchical Trajectory VAE (HTVAE)
'''
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('...')

import os
import sys
import torch

from method.embedder.htvae_utils import (SharedConvolutionalEncoder_84,
    SharedConvolutionalEncoder_64, SharedConvolutionalEncoder_48,
    SharedLSTMEncoder, OptionNetwork, InferenceNetwork,
    LatentDecoder, ObservationMLPDecoder, ObservationLSTMDecoder, ObservationCNNDecoder_84,
    ObservationCNNDecoder_64, ObservationCNNDecoder_48)

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, init
try:
    from method.embedder.utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood, L2_loss)
except ImportError:
    # put parent directory in path for utils
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from method.embedder.utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood)

global_cuda = True

# Model
class HTVAE(nn.Module):
    def __init__(self, batch_size=16, sample_size=200, n_features=1,
                 o_dim=3, n_hidden_option=128, hidden_dim_option=3,
                 n_stochastic=1, z_dim=16, n_hidden=3, hidden_dim=128,
                 encoder_dim=128,
                 nonlinearity=F.relu, print_vars=False,
                 is_cuda=True, conv=False, input_channels = 2,
                 args=None):
        """
        :param sample_size:
        :param n_features:
        :param o_dim:
        :param n_hidden_option:
        :param hidden_dim_option:
        :param n_stochastic:
        :param z_dim:
        :param n_hidden:
        :param hidden_dim:
        :param nonlinearity:
        :param print_vars:
        """
        super().__init__()

        self.batch_size = batch_size
        # This is the number of trajectories for each option
        self.sample_size = sample_size
        self.n_features = n_features
        self.is_cuda = is_cuda
        if not self.is_cuda:
            global_cuda = False

        # context
        self.o_dim = o_dim
        self.n_hidden_option = n_hidden_option
        self.hidden_dim_option = hidden_dim_option

        # latent
        self.n_stochastic = n_stochastic
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        self.args = args

        self.nonlinearity = nonlinearity

        self.input_channels = input_channels

        self.is_img = self.args.env_name.startswith('BlockPlayImg')

        # modules
        # convolutional encoder
        self.conv = conv
        if self.conv:
            if args.image_resolution == 84:
                self.shared_convolutional_encoder = SharedConvolutionalEncoder_84(self.nonlinearity,
                    input_channels=input_channels, args=args)
                if not self.args.deeper_encoder:
                    self.n_features = 32 * 3 * 3
                else:
                    raise NotImplementedError
            elif args.image_resolution == 64:
                self.shared_convolutional_encoder = SharedConvolutionalEncoder_64(self.nonlinearity,
                    input_channels=input_channels, args=args)
                if not self.args.deeper_encoder:
                    self.n_features = 32 * 2 * 2
                else:
                    raise NotImplementedError
            elif args.image_resolution == 48:
                self.shared_convolutional_encoder = SharedConvolutionalEncoder_48(self.nonlinearity,
                    input_channels=input_channels, args=args)
                if not self.args.deeper_encoder:
                    self.n_features = 32 * 3 * 3
                else:
                    self.n_features = 256 * 3 * 3
            else:
                raise NotImplementedError

        if args.onehot_state:
            self.loss_fn = nn.CrossEntropyLoss(reduction='sum') # Mean is taken later

        # Bi-lstm encoder
        self.shared_lstm_encoder = SharedLSTMEncoder(
            self.batch_size, self.sample_size,
            self.n_features, self.encoder_dim, num_layers=2,
            bidirectional=True, batch_first=True)


        # option network
        # The number of features of option network is same as the hidden dimension in LSTM
        option_args = (self.batch_size, self.sample_size, self.encoder_dim,
                          self.n_hidden_option, self.hidden_dim_option,
                          self.o_dim, self.nonlinearity)
        self.option_network = OptionNetwork(*option_args)

        z_args = (self.batch_size, self.sample_size, self.n_features, self.encoder_dim,
                  self.n_hidden, self.hidden_dim, self.o_dim, self.z_dim,
                  self.nonlinearity, args)

        # inference networks (one for each stochastic layer)
        self.inference_networks = nn.ModuleList([InferenceNetwork(*z_args)
                                             for _ in range(self.n_stochastic)])
        # latent decoders (again, one for each stochastic layer)
        self.latent_decoders = nn.ModuleList([LatentDecoder(*z_args)
                                              for _ in range(self.n_stochastic)])

        # observation decoder
        observation_args = (self.batch_size, self.sample_size, self.n_features,
                            self.nonlinearity, args)
        if args.emb_mlp_decoder and self.is_img:
            self.observation_decoder = ObservationMLPDecoder(*observation_args)
        else:
            self.observation_decoder = ObservationLSTMDecoder(*observation_args)

        if self.conv:
            if self.args.image_resolution == 84:
                self.observation_convolutional_decoder = ObservationCNNDecoder_84(
                    self.nonlinearity,
                    input_channels=input_channels,
                    args=args)
            elif self.args.image_resolution == 64:
                self.observation_convolutional_decoder = ObservationCNNDecoder_64(
                    self.nonlinearity,
                    input_channels=input_channels,
                    args=args)
            elif self.args.image_resolution == 48:
                self.observation_convolutional_decoder = ObservationCNNDecoder_48(
                    self.nonlinearity,
                    input_channels=input_channels,
                    args=args)
            else:
                raise NotImplementedError

        # initialize weights
        self.apply(self.weights_init)

        # print variables for sanity check and debugging
        if print_vars:
            for i, pair in enumerate(self.named_parameters()):
                name, param = pair
                print("{} --> {}, {}".format(i + 1, name, param.size()))
            print()


    def set_batch_size(self, batch_size, sample_size):
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.shared_lstm_encoder.batch_size = batch_size
        self.shared_lstm_encoder.sample_size = sample_size

        self.option_network.batch_size = batch_size
        self.option_network.sample_size = sample_size

        for inference_network in self.inference_networks:
            inference_network.batch_size = batch_size
            inference_network.sample_size = sample_size
        for latent_decoder in self.latent_decoders:
            latent_decoder.batch_size = batch_size
            latent_decoder.sample_size = sample_size

        self.observation_decoder.batch_size = batch_size
        self.observation_decoder.sample_size = sample_size


    def forward(self, x):
        if self.args.onehot_state:
            old_x = x.clone()
            x = x.view(*x.shape[:-2], -1)

        if self.conv:
            old_x = x.clone()
            if self.is_img:
                assert len(x.shape) == 5
                # Our input data is a bunch of images per option, not videos.
                # Things work about the same though.
                x = x.unsqueeze(2)
            # x is of dimensions: (batch_size, sample_size, seq_len, im_resolution, im_resolution, input_channels)
            bs, ss, sl = x.shape[0], x.shape[1], x.shape[2]

            x = x.permute(0,1,2,5,3,4)
            x = self.shared_convolutional_encoder(x)
            # x is of shape (-1, 32, 3, 3)
            x = x.view(bs, ss, sl, -1)

        # x is of dimensions: (batch_size, sample_size, seq_len, n_features)
        # lstm encoder
        h = self.shared_lstm_encoder(x)

        # Now h is of dimensions: (batch_size, sample_size, encoder_dim)

        # option network
        o_mean, o_logvar = self.option_network(h)
        o = self.reparameterize_gaussian(o_mean, o_logvar)

        # inference networks
        qz_samples = []
        qz_params = []
        z = None
        for inference_network in self.inference_networks:
            z_mean, z_logvar = inference_network(h, z, o)
            qz_params.append([z_mean, z_logvar])
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            qz_samples.append(z)

        # latent decoders
        pz_params = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, o) # Note: no x[:,:,0,:] here
            pz_params.append([z_mean, z_logvar])
            z = qz_samples[i]

        # observation decoder
        zs = torch.cat(qz_samples, dim=1)
        if self.args.emb_mlp_decoder and self.is_img:
            x_mean, x_logvar = self.observation_decoder(zs, o)
        else:
            x_mean, x_logvar = self.observation_decoder(zs, o, x[:,:,0,:], x.shape[-2] - 1)

        if self.conv:
            x_mean, x_mask, x_logvar = self.observation_convolutional_decoder(x_mean)
            # x is of shape (-1, input_channels, im_resolution, im_resolution)
            x_mean = x_mean.permute(0, 2, 3, 1).contiguous()
            x_logvar = x_logvar.permute(0, 2, 3, 1).contiguous()
            if not self.is_img and self.args.image_mask:
                x_mask = x_mask.permute(0, 2, 3, 1).contiguous()

                x_mean = x_mask * x_mean
                x_init_image = ((1 - x_mask).reshape(bs, ss, -1, self.args.image_resolution, self.args.image_resolution, 1) * old_x[:,:,:1,:]).reshape(x_mean.shape)
                x_mean = x_mean + x_init_image
            x = old_x

        if self.args.onehot_state:
            x = old_x

        if self.args.no_initial_state or self.is_img:
            outputs = (
                (o_mean, o_logvar),
                (qz_params, pz_params),
                (x, x_mean, x_logvar)
            )
        else:
            outputs = (
                (o_mean, o_logvar),
                (qz_params, pz_params),
                (x[:,:,1:], x_mean, x_logvar)
            )


        return outputs


    def loss(self, outputs, weight):
        o_outputs, z_outputs, x_outputs = outputs

        # 1. Reconstruction loss
        x, x_mean, x_logvar = x_outputs

        if self.args.onehot_state:
            x_mean = x_mean.view(-1, self.args.play_grid_size)
            labels = torch.argmax(x, axis=-1).view(-1)
            recon_loss = -self.loss_fn(x_mean, labels)
            # recon_loss /= (2. * self.args.trajectory_len)
        else:
            x = x.view(self.batch_size * self.sample_size, -1)
            x_mean = x_mean.view(self.batch_size * self.sample_size, -1)
            x_logvar = x_logvar.contiguous().view(self.batch_size * self.sample_size, -1)
            recon_loss = gaussian_log_likelihood(x, x_mean, x_logvar)

        recon_loss /= (self.batch_size * self.sample_size)

        # 2. KL Divergence terms
        kl = 0

        # a) Option/Context divergence
        o_mean, o_logvar = o_outputs
        kl_o = kl_diagnormal_stdnormal(o_mean, o_logvar)
        kl += kl_o

        # b) Latent divergences
        qz_params, pz_params = z_outputs
        shapes = (
            (self.batch_size, self.sample_size, self.z_dim),
            (self.batch_size, 1, self.z_dim)
        )
        for i in range(self.n_stochastic):
            args = (qz_params[i][0].view(shapes[0]),    # mean
                    qz_params[i][1].view(shapes[0]),    # logvar
                    # pz_params[i][0].view(shapes[0]),
                    # pz_params[i][1].view(shapes[0]))
                    pz_params[i][0].view(shapes[1] if i == 0 else shapes[0]),
                    pz_params[i][1].view(shapes[1] if i == 0 else shapes[0]))
            kl_z = kl_diagnormal_diagnormal(*args)
            kl += kl_z

        kl /= (self.batch_size * self.sample_size)

        # Variational lower bound and weighted loss
        vlb = recon_loss - kl
        loss = - ((weight * recon_loss) - (kl / weight))

        # l2 loss
        if x.shape == x_mean.shape and not self.args.onehot_state:
            l2_loss = L2_loss(x, x_mean)
        elif not self.args.onehot_state:
            l2_loss = L2_loss(x.reshape(-1), x_mean.reshape(-1))
        else:
            l2_loss = torch.tensor(0.)

        return loss, vlb, recon_loss, l2_loss, kl

    def step(self, batch, alpha, optimizer, scheduler, clip_gradients=True):
        assert self.training is True

        if self.is_cuda:
            inputs = Variable(batch.cuda())
        else:
            inputs = Variable(batch)

        batch_size = batch.shape[0]
        sample_size = batch.shape[1]
        self.set_batch_size(batch_size, sample_size)

        outputs = self.forward(inputs)
        loss, vlb, recon_loss, l2_loss, kl = self.loss(outputs, weight=(alpha + 1))

        # perform gradient update
        optimizer.zero_grad()
        loss.backward()
        if clip_gradients:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        _, _, x_outputs = outputs
        x, x_mean, x_logvar = x_outputs
        if self.conv:
            x = x.view(self.batch_size, self.sample_size, -1, self.args.image_resolution, self.args.image_resolution, self.input_channels)
            x_mean = x_mean.view(self.batch_size, self.sample_size, -1, self.args.image_resolution, self.args.image_resolution, self.input_channels)

            x_sample = x[0]
            x_mean = x_mean[0]

            # output variational lower bound
            return vlb.item(), loss.item(), recon_loss.item(), l2_loss.item(), kl.item(), x_sample, x_mean
        else:
            x = x.view(self.batch_size, self.sample_size, -1, self.n_features)
            x_mean = x_mean.view(self.batch_size, self.sample_size, -1, self.n_features)
            x_sample = x
            x_mean = x_mean
            if self.args.onehot_state:
                x_sample = x_sample.view(*x_sample.shape[:-1], 2, -1).argmax(axis = -1)
                x_mean = x_mean.view(*x_mean.shape[:-1], 2, -1).argmax(axis = -1)
            return vlb.item(), loss.item(), recon_loss.item(), l2_loss.item(), kl.item(), x_sample, x_mean


    def save(self, optimizer, scheduler, path, epoch):
        if scheduler is None:
            torch.save({
                'epoch': epoch,
                'model_state': self.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state': self.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }, path)


    def load(self, optimizer, scheduler, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch_num = checkpoint['epoch']
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        return optimizer, scheduler, epoch_num


    def get_trajectory_embedding(self, inputs, option_embedding):
        x = inputs
        if self.conv:
            bs, ss, sl = x.shape[0], x.shape[1], x.shape[2]
            x = x.permute(0,1,2,5,3,4)
            x = self.shared_convolutional_encoder(x)
            # x is of shape (-1, 32, 5, 5)
            x = x.view(bs, ss, sl, -1)

        # lstm encoder
        h = self.shared_lstm_encoder(x)

        # option network
        if option_embedding.shape[0] != inputs.shape[0]:
            o = option_embedding.repeat(inputs.shape[0])
        else:
            o = option_embedding
        # o, _ = self.option_network(h)

        # inference networks
        qz_samples = []
        z = None
        for inference_network in self.inference_networks:
            z, _ = inference_network(h, z, o)
            qz_samples.append(z)

        return z


    def get_reconstruction(self, inputs, option_embedding):
        x = inputs
        if self.conv:
            bs, ss, sl = x.shape[0], x.shape[1], x.shape[2]
            x = x.permute(0,1,2,5,3,4)
            x = self.shared_convolutional_encoder(x)
            # x is of shape (-1, 32, 5, 5)
            x = x.view(bs, ss, sl, -1)

        # lstm encoder
        h = self.shared_lstm_encoder(x)

        # option network
        o = option_embedding.repeat(inputs.shape[0])
        # o, _ = self.option_network(h)

        # inference networks
        qz_samples = []
        z = None
        for inference_network in self.inference_networks:
            z, _ = inference_network(h, z, o)
            qz_samples.append(z)


        # latent decoders
        pz_params = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, o) # other version: no x[:,:,0,:] here
            pz_params.append([z_mean, z_logvar])
            z = qz_samples[i]

        # observation decoder
        zs = torch.cat(qz_samples, dim=1)
        if self.args.emb_mlp_decoder:
            x_mean, _ = self.observation_decoder(zs, o)
        else:
            x_mean, _ = self.observation_decoder(zs, o, x[:,:,0,:], x.shape[-2] - 1)

        if self.conv:
            x_mean, x_mask, _ = self.observation_convolutional_decoder(x_mean)
            # x is of shape (-1, input_channels, 84, 84)
            x_mean = x_mean.permute(0, 2, 3, 1).contiguous()
            if self.args.image_mask:
                x_mask = x_mask.permute(0, 2, 3, 1).contiguous()

                x_mean = x_mask * x_mean
                x_init_image = ((1 - x_mask).reshape(bs, ss, -1, self.args.image_resolution, self.args.image_resolution, 1) * x[:,:,:1,:]).reshape(x_mean.shape)
                x_mean = x_mean + x_init_image

        return x_mean


    def get_option_distributions(self, inputs, n_distributions=1):
        x = inputs
        if self.conv:
            if len(x.shape) == 5:
                # Our input data is a bunch of images per option, not videos.
                # Things work about the same though.
                x = x.unsqueeze(2)
            bs, ss, sl = x.shape[0], x.shape[1], x.shape[2]
            x = x.permute(0,1,2,5,3,4)
            x = self.shared_convolutional_encoder(x)
            # x is of shape (-1, 32, 5, 5)
            x = x.view(bs, ss, sl, -1)

        # lstm encoder
        h = self.shared_lstm_encoder(x)

        # option network
        o_mean, o_logvar = self.option_network(h)

        z = None

        if len(self.latent_decoders) == 1:
            n_distributions = 1

        for i, latent_decoder in enumerate(self.latent_decoders):
            latent_decoder.sample_size = n_distributions
            z_mean, z_logvar = latent_decoder(z, o_mean)

            # At the first step sample multiple distributions
            if n_distributions > 1 and i == 0:
                z = self.reparameterize_gaussian(z_mean, z_logvar)
            else:
                z = z_mean

        return z_mean, z_logvar


    def get_option_embedding(self, inputs):
        x = inputs
        if self.conv:
            if len(x.shape) == 5:
                # Our input data is a bunch of images per option, not videos.
                # Things work about the same though.
                x = x.unsqueeze(2)
            bs, ss, sl = x.shape[0], x.shape[1], x.shape[2]
            x = x.permute(0,1,2,5,3,4)
            x = self.shared_convolutional_encoder(x)
            # x is of shape (-1, 32, 5, 5)
            x = x.view(bs, ss, sl, -1)

        # lstm encoder
        h = self.shared_lstm_encoder(x)

        # option network
        o_mean, o_logvar = self.option_network(h)

        return o_mean, o_logvar


    def sample_conditioned(self, inputs):
        h = self.shared_lstm_encoder(inputs)
        s, _ = self.option_network(h)

        # latent decoders
        pz_samples = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, s)
            if i == 0:
                z_mean = z_mean.repeat(self.sample_size, 1)
                z_logvar = z_logvar.repeat(self.sample_size, 1)
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            pz_samples.append(z)

        # observation decoder
        zs = torch.cat(pz_samples, dim=1)
        x = self.observation_decoder(zs, s)

        return x

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        if global_cuda:
            eps = Variable(torch.randn(std.size()).cuda())
        else:
            eps = Variable(torch.randn(std.size()))
        return mean + std * eps

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def transform_dataset(self, data):
        return data



