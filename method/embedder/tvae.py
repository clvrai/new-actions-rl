import torch.nn as nn
from  torch.nn import functional as F
from rlf.rl.model import Flatten
import torch

from method.embedder.htvae_utils import (SharedConvolutionalEncoder_84, FCResBlock,
    SharedConvolutionalEncoder_64, SharedConvolutionalEncoder_48,
    SharedLSTMEncoder, ObservationLSTMDecoder, ObservationMLPDecoder, ObservationCNNDecoder_84,
    ObservationCNNDecoder_64, ObservationCNNDecoder_48)
from method.embedder.htvae import HTVAE
from method.embedder.utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal, gaussian_log_likelihood, L2_loss)

# q(z | h)
class InferenceNetwork(nn.Module):
    def __init__(self, args, nonlinearity):
        super().__init__()

        self.args = args
        self.encoder_dim = self.args.encoder_dim
        self.z_dim = self.args.z_dim
        self.hidden_dim = self.args.hidden_dim_traj
        self.n_hidden = self.args.n_hidden_traj
        self.nonlinearity = nonlinearity

        self.fc_h = nn.Linear(self.encoder_dim, self.hidden_dim)

        self.fc_block1 = FCResBlock(width=self.hidden_dim,
                                    n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)
        self.fc_block2 = FCResBlock(width=self.hidden_dim,
                                    n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)

        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

        if self.args.shared_var:
            self.fc_params = nn.Linear(self.hidden_dim, self.z_dim)
            if args.no_cuda:
                self.logvar = nn.Parameter(torch.randn(1, self.z_dim))
            else:
                self.logvar = nn.Parameter(torch.randn(1, self.z_dim).cuda())
        else:
            self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)

    def forward(self, h):
        e = self.fc_h(h)
        e = self.nonlinearity(e)

        # residual blocks
        e = self.fc_block1(e)
        e = self.fc_block2(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        if self.args.shared_var:
            # 'global' batch norm
            e = e.view(-1, 1, self.z_dim)
            e = self.bn_params(e)
            e = e.view(-1, self.z_dim)
            return e, self.logvar.expand_as(e)
        else:
            # 'global' batch norm
            e = e.view(-1, 1, 2 * self.z_dim)
            e = self.bn_params(e)
            e = e.view(-1, 2 * self.z_dim)
            return e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

global_cuda = True

'''
	Trajectory VAE
'''
class TVAE(nn.Module):
    def __init__(self, batch_size=16, sample_size=200, n_features=1,
                 o_dim=3, n_hidden_option=128, hidden_dim_option=3,
                 n_stochastic=1, z_dim=16, n_hidden=3, hidden_dim=128,
                 encoder_dim=128, nonlinearity=F.relu, print_vars=False,
                 is_cuda=True, conv=False, input_channels=2,
                 args=None):

        super().__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.is_cuda = is_cuda
        if not self.is_cuda:
            global_cuda = False

        self.hidden_dim = hidden_dim

        self.encoder_dim = encoder_dim # This comes out of LSTM Encoder
        self.z_dim = z_dim  # This goes into LSTM decoder

        self.conv = conv
        self.nonlinearity = nonlinearity

        self.args = args

        # n_hidden_img = 32 * 3 * 3
        n_layers = 2
        self.input_channels = input_channels

        self.is_img = self.args.env_name.startswith('BlockPlayImg')
        if args.onehot_state:
            self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        # modules
        # convolutional encoder
        self.conv = conv
        if self.conv:
            if args.image_resolution == 84:
                self.image_encoder = SharedConvolutionalEncoder_84(self.nonlinearity,
                    input_channels=input_channels, args=args)
                if not self.args.deeper_encoder:
                    self.n_features = 32 * 3 * 3
                else:
                    raise NotImplementedError
            elif args.image_resolution == 64:
                self.image_encoder = SharedConvolutionalEncoder_64(self.nonlinearity,
                    input_channels=input_channels, args=args)
                if not self.args.deeper_encoder:
                    self.n_features = 32 * 2 * 2
                else:
                    raise NotImplementedError
            elif args.image_resolution == 48:
                self.image_encoder = SharedConvolutionalEncoder_48(self.nonlinearity,
                    input_channels=input_channels, args=args)
                if not self.args.deeper_encoder:
                    self.n_features = 32 * 3 * 3
                else:
                    self.n_features = 256 * 3 * 3
            else:
                raise NotImplementedError


        # Bi-lstm encoder
        self.seq_encoder = SharedLSTMEncoder(
            self.batch_size, self.sample_size,
            self.n_features, self.encoder_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True)

        # Sampling network after lstm output
        self.inference_net = InferenceNetwork(self.args, self.nonlinearity)

        self.args.effect_only_decoder = True
        if self.args.emb_mlp_decoder and self.is_img:
            self.seq_decoder = ObservationMLPDecoder(
                self.batch_size, self.sample_size, self.n_features, self.nonlinearity, args)
        else:
            self.seq_decoder = ObservationLSTMDecoder(
                self.batch_size, self.sample_size, self.n_features, self.nonlinearity, args)

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


    def set_batch_size(self, batch_size, sample_size):
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.seq_encoder.batch_size = batch_size
        self.seq_encoder.sample_size = sample_size

        self.seq_decoder.batch_size = batch_size
        self.seq_decoder.sample_size = sample_size


    def forward(self, x):
        # Expects batch * seq_len * n_features
        if self.args.onehot_state:
            old_x = x.clone()
            x = x.view(*x.shape[:-2], -1)

        if self.conv:
            old_x = x.clone()
            if self.is_img:
                assert len(x.shape) == 4
                # Our input data is a bunch of images per option, not videos.
                # Things work about the same though.
                x = x.unsqueeze(1)
            # Expects batch * seq_len * 84 * 84 * input_channels
            bs, sl = x.shape[0], x.shape[1]
            x = x.permute(0,1,-1,2,3)   # batch * seq_len * input_channels * 84 * 84
            x = self.image_encoder(x)
            # x is of shape (-1, 32, 3, 3)
            x = x.view(bs, sl, -1)

        hidden = self.seq_encoder(x)
        # # Average out the LSTM layer dimension
        # hidden = torch.mean(hidden, dim=0)

        z_mean, z_logvar = self.inference_net(hidden)
        z = HTVAE.reparameterize_gaussian(z_mean, z_logvar)

        if self.args.emb_mlp_decoder and self.is_img:
            x_mean, x_logvar = self.seq_decoder(z, None)
        else:
            x_mean, x_logvar = self.seq_decoder(z, None, x[:, 0], x.shape[1] - 1)

        if self.conv:
            x_mean, x_mask, x_logvar = self.observation_convolutional_decoder(x_mean)
            # x is of shape (-1, input_channels, im_resolution, im_resolution)
            x_mean = x_mean.permute(0, 2, 3, 1).contiguous()
            x_logvar = x_logvar.permute(0, 2, 3, 1).contiguous()
            if not self.is_img and self.args.image_mask:
                x_mask = x_mask.permute(0, 2, 3, 1).contiguous()
                x_mean = x_mask * x_mean
                x_init_image = ((1 - x_mask).reshape(bs, -1, self.args.image_resolution, self.args.image_resolution, 1) * old_x[:,:1]).reshape(x_mean.shape)
                x_mean = x_mean + x_init_image
            x = old_x

        if self.args.onehot_state:
            x = old_x

        if self.args.no_initial_state or self.is_img:
            outputs = (
                (x, x_mean, x_logvar),
                (z_mean, z_logvar)
                )
        else:
            outputs = (
                (x[:, 1:], x_mean, x_logvar),
                (z_mean, z_logvar)
                )

        return outputs


    def loss(self, outputs, weight):
        out_x, out_z = outputs

        # 1. Reconstruction loss
        x, x_mean, x_logvar = out_x

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

        # 2. KL divergence
        z_mean, z_logvar = out_z
        kl_loss = kl_diagnormal_stdnormal(z_mean, z_logvar)
        kl_loss /= (self.batch_size * self.sample_size)

        # Variational lower bound and weighted loss
        vlb = recon_loss - kl_loss
        loss = - ((weight * recon_loss) - (kl_loss / weight))

        # l2 loss
        if x.shape == x_mean.shape and not self.args.onehot_state:
            l2_loss = L2_loss(x, x_mean)
        elif not self.args.onehot_state:
            l2_loss = L2_loss(x.reshape(-1), x_mean.reshape(-1))
        else:
            l2_loss = torch.tensor(0.)


        return loss, vlb, recon_loss, l2_loss, kl_loss


    def step(self, batch, alpha, optimizer, scheduler, clip_gradients):
        batch_size = batch.shape[0]
        sample_size = batch.shape[1]

        self.set_batch_size(batch_size, sample_size)

        # This step merges samples and batches
        batch = batch.view([-1, *batch.shape[2:]])

        outputs = self.forward(batch)

        x_outputs, _ = outputs
        x, x_mean, x_logvar = x_outputs

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


    def transform_dataset(self, data):
        return data.reshape([-1, *data.shape[2:]])

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

    def get_trajectory_embedding(self, inputs):
        x = inputs
        x = x.view([-1, *x.shape[2:]])

        if self.conv:
            if self.is_img:
                assert len(x.shape) == 4
                # Our input data is a bunch of images per option, not videos.
                # Things work about the same though.
                x = x.unsqueeze(1)
            # Expects batch * seq_len * 84 * 84 * input_channels
            bs, sl = x.shape[0], x.shape[1]
            x = x.permute(0,1,-1,2,3)   # batch * seq_len * input_channels * 84 * 84
            x = self.image_encoder(x)
            # x is of shape (-1, 32, 3, 3)
            x = x.view(bs, sl, -1)

        # lstm encoder
        hidden = self.seq_encoder(x)
        z_mean, _ = self.inference_net(hidden)

        return z_mean


    def get_option_distributions(self, inputs, n_distributions=None):
        x = inputs
        x = x.view([-1, *x.shape[2:]])
        if self.conv:
            if self.is_img:
                assert len(x.shape) == 4
                # Our input data is a bunch of images per option, not videos.
                # Things work about the same though.
                x = x.unsqueeze(1)
            # Expects batch * seq_len * 84 * 84 * input_channels
            bs, sl = x.shape[0], x.shape[1]
            x = x.permute(0,1,-1,2,3)   # batch * seq_len * input_channels * 84 * 84
            x = self.image_encoder(x)
            # x is of shape (-1, 32, 3, 3)
            x = x.view(bs, sl, -1)

        # lstm encoder
        hidden = self.seq_encoder(x)
        z_mean, _ = self.inference_net(hidden)

        all_z_mean = z_mean.mean(dim=0)
        all_z_logvar = z_mean.var(dim=0).log()

        return all_z_mean, all_z_logvar


    def get_option_embedding(self, inputs):
        x = inputs
        x = x.view([-1, *x.shape[2:]])
        if self.conv:
            if self.is_img:
                assert len(x.shape) == 4
                # Our input data is a bunch of images per option, not videos.
                # Things work about the same though.
                x = x.unsqueeze(1)
            # Expects batch * seq_len * 84 * 84 * input_channels
            bs, sl = x.shape[0], x.shape[1]
            x = x.permute(0,1,-1,2,3)   # batch * seq_len * input_channels * 84 * 84
            x = self.image_encoder(x)
            # x is of shape (-1, 32, 3, 3)
            x = x.view(bs, sl, -1)

        # lstm encoder
        hidden = self.seq_encoder(x)
        z_mean, _ = self.inference_net(hidden)
        z_mean = z_mean.view([x.shape[0], x.shape[1], -1])
        all_z_mean = z_mean.mean(dim=1)

        return all_z_mean
