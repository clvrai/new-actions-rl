import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F



# Building block for convolutional encoder with same padding
class Conv2d3x3(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Conv2d3x3, self).__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, stride=stride)

    def forward(self, x):
        return self.conv(x)


# SHARED CONVOLUTIONAL ENCODER
class SharedConvolutionalEncoder_84(nn.Module):
    def __init__(self, nonlinearity, input_channels=2, args=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.input_channels = input_channels
        self.args = args
        # Input shape is N * 2 * 84 * 84
        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=input_channels, out_channels=8, downsample=True),
            # shape is now (-1, 8, 42, 42)
            Conv2d3x3(in_channels=8, out_channels=8),
            Conv2d3x3(in_channels=8, out_channels=16, downsample=True),
            # shape is now (-1, 16, 21, 21)
            Conv2d3x3(in_channels=16, out_channels=16),
            Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
            # shape is now (-1, 16, 11, 11)
            Conv2d3x3(in_channels=16, out_channels=16),
            Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
            # shape is now (-1, 32, 6, 6)
            Conv2d3x3(in_channels=32, out_channels=32),
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 3, 3)
        ])
        if args.emb_use_batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm2d(num_features=8),
                nn.BatchNorm2d(num_features=8),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
            ])

    def forward(self, x):
        h = x.view(-1, self.input_channels, 84, 84)   # hard-coded input image size
        if not self.args.emb_use_batch_norm:
            for conv in self.conv_layers:
                h = conv(h)
                h = self.nonlinearity(h)
        else:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                h = conv(h)
                h = bn(h)
                h = self.nonlinearity(h)
        return h



# SHARED CONVOLUTIONAL ENCODER
class SharedConvolutionalEncoder_64(nn.Module):
    def __init__(self, nonlinearity, input_channels=1, args=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.input_channels = input_channels
        self.args = args
        # Input shape is N * 1 * 64 * 64
        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=input_channels, out_channels=8, downsample=True),
            # shape is now (-1, 8, 32, 32)
            Conv2d3x3(in_channels=8, out_channels=8),
            Conv2d3x3(in_channels=8, out_channels=16, downsample=True),
            # shape is now (-1, 16, 16, 16)
            Conv2d3x3(in_channels=16, out_channels=16),
            Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
            # shape is now (-1, 16, 8, 8)
            Conv2d3x3(in_channels=16, out_channels=16),
            Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
            # shape is now (-1, 32, 4, 4)
            Conv2d3x3(in_channels=32, out_channels=32),
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 2, 2)
        ])
        if args.emb_use_batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm2d(num_features=8),
                nn.BatchNorm2d(num_features=8),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
            ])

    def forward(self, x):
        h = x.view(-1, self.input_channels, 64, 64)   # hard-coded input image size
        if not self.args.emb_use_batch_norm:
            for conv in self.conv_layers:
                h = conv(h)
                h = self.nonlinearity(h)
        else:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                h = conv(h)
                h = bn(h)
                h = self.nonlinearity(h)
        return h

# SHARED CONVOLUTIONAL ENCODER
class SharedConvolutionalEncoder_48(nn.Module):
    def __init__(self, nonlinearity, input_channels=1, args=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.input_channels = input_channels

        self.args = args
        
        if not args.deeper_encoder:
            # Input shape is N * 1 * 48 * 48
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=input_channels, out_channels=8, downsample=True),
                # shape is now (-1, 8, 24, 24)
                Conv2d3x3(in_channels=8, out_channels=8),
                Conv2d3x3(in_channels=8, out_channels=16, downsample=True),
                # shape is now (-1, 16, 12, 12)
                Conv2d3x3(in_channels=16, out_channels=16),
                Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
                # shape is now (-1, 16, 6, 6)
                Conv2d3x3(in_channels=16, out_channels=16),
                Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
                # shape is now (-1, 32, 3, 3)
            ])
            if args.emb_use_batch_norm:
                self.bn_layers = nn.ModuleList([
                    nn.BatchNorm2d(num_features=8),
                    nn.BatchNorm2d(num_features=8),
                    nn.BatchNorm2d(num_features=16),
                    nn.BatchNorm2d(num_features=16),
                    nn.BatchNorm2d(num_features=16),
                    nn.BatchNorm2d(num_features=16),
                    nn.BatchNorm2d(num_features=32),
                ])
        else:
            # Input shape is N * 1 * 48 * 48
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=input_channels, out_channels=32),
                Conv2d3x3(in_channels=32, out_channels=32),
                Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
                # shape is now (-1, 32, 24, 24)
                Conv2d3x3(in_channels=32, out_channels=64),
                Conv2d3x3(in_channels=64, out_channels=64),
                Conv2d3x3(in_channels=64, out_channels=64, downsample=True),
                # shape is now (-1, 64, 12, 12)
                Conv2d3x3(in_channels=64, out_channels=128),
                Conv2d3x3(in_channels=128, out_channels=128),
                Conv2d3x3(in_channels=128, out_channels=128, downsample=True),
                # shape is now (-1, 128, 6, 6)
                Conv2d3x3(in_channels=128, out_channels=256),
                Conv2d3x3(in_channels=256, out_channels=256),
                Conv2d3x3(in_channels=256, out_channels=256, downsample=True),
                # shape is now (-1, 256, 3, 3)
            ])

            if args.emb_use_batch_norm:
                self.bn_layers = nn.ModuleList([
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=64),
                    nn.BatchNorm2d(num_features=64),
                    nn.BatchNorm2d(num_features=64),
                    nn.BatchNorm2d(num_features=128),
                    nn.BatchNorm2d(num_features=128),
                    nn.BatchNorm2d(num_features=128),
                    nn.BatchNorm2d(num_features=256),
                    nn.BatchNorm2d(num_features=256),
                    nn.BatchNorm2d(num_features=256),
                ])


    def forward(self, x):
        h = x.view(-1, self.input_channels, 48, 48)   # hard-coded input image size
        if not self.args.emb_use_batch_norm:
            for conv in self.conv_layers:
                h = conv(h)
                h = self.nonlinearity(h)
        else:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                h = conv(h)
                h = bn(h)
                h = self.nonlinearity(h)
        return h


# Module for residual/skip connections
class FCResBlock(nn.Module):
    def __init__(self, width, n, nonlinearity):
        """

        :param width:
        :param n:
        :param nonlinearity:
        """
        super(FCResBlock, self).__init__()
        self.n = n
        self.nonlinearity = nonlinearity
        self.block = nn.ModuleList([nn.Linear(width, width) for _ in range(self.n)])

    def forward(self, x):
        e = x + 0
        for i, layer in enumerate(self.block):
            e = layer(e)
            if i < (self.n - 1):
                e = self.nonlinearity(e)
        return self.nonlinearity(e + x)



# The input to this might have [batch_size, sample_size,...] format
# In this case, maybe it is better to flatten it before passing to lstm.
class SharedLSTMEncoder(nn.Module):
    def __init__(self, batch_size, sample_size, n_features,
        hidden_dim, num_layers=2,
        bidirectional=True, batch_first=True):

        super(SharedLSTMEncoder, self).__init__()
        self.sample_size = sample_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        # LSTM Module
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, num_layers,
            batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x):
        # output of lstm is of shape (batch, seq_len, num_directions*hidden_size)
        # We average this to get (batch, hidden_size)
        x = x.view(self.batch_size * self.sample_size, -1, self.n_features)
        x, _ = self.lstm(x)
        x = x.view(self.batch_size * self.sample_size, -1, 2, self.hidden_dim)
        x = torch.mean(x, [1,2])
        x = x.view(self.batch_size, self.sample_size, self.hidden_dim)
        return x



# PRE-POOLING FOR OPTION NETWORK
class PrePool(nn.Module):
    """

    """

    def __init__(self, n_features, n_hidden, hidden_dim, nonlinearity):
        super(PrePool, self).__init__()
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_initial = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                   nonlinearity=self.nonlinearity)
        self.fc_final = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        # reshape and initial affine
        e = x.view(-1, self.n_features)
        e = self.fc_initial(e)
        e = self.nonlinearity(e)

        # residual block
        e = self.fc_block(e)

        # final affine
        e = self.fc_final(e)

        return e


# POST POOLING FOR OPTION NETWORK
class PostPool(nn.Module):
    """

    """

    def __init__(self, n_hidden, hidden_dim, o_dim, nonlinearity):
        super(PostPool, self).__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.o_dim = o_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden,
                                   nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.o_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        e = self.fc_block(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.o_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.o_dim)

        mean, logvar = e[:, :self.o_dim], e[:, self.o_dim:]

        return mean, logvar


# OPTION (STATISTIC) NETWORK q(s|D)
class OptionNetwork(nn.Module):
    """
        Encode each trajectory into a vector
        Pool these trajectory vectors in to one dataset vector for each option
    """

    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, o_dim, nonlinearity):
        super(OptionNetwork, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.o_dim = o_dim

        self.nonlinearity = nonlinearity

        # modules
        self.prepool = PrePool(self.n_features, self.n_hidden,
                               self.hidden_dim, self.nonlinearity)
        self.postpool = PostPool(self.n_hidden, self.hidden_dim,
                                 self.o_dim, self.nonlinearity)

    def forward(self, x):
        e = self.prepool(x)
        e = self.pool(e)
        e = self.postpool(e)
        return e

    def pool(self, e):
        e = e.view(self.batch_size, self.sample_size, self.hidden_dim)
        e = e.mean(1).view(self.batch_size, self.hidden_dim)
        return e


# INFERENCE NETWORK q(z|x, z, s)
class InferenceNetwork(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features, encoder_dim,
                 n_hidden, hidden_dim, o_dim, z_dim, nonlinearity, args):
        super(InferenceNetwork, self).__init__()

        self.args = args
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.encoder_dim = encoder_dim

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.o_dim = o_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_h = nn.Linear(self.encoder_dim, self.hidden_dim)
        self.fc_s = nn.Linear(self.o_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_block1 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)
        self.fc_block2 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)

        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

        if args.shared_var:
            self.fc_params = nn.Linear(self.hidden_dim, self.z_dim)
            if args.no_cuda:
                self.logvar = nn.Parameter(torch.randn(1, self.z_dim))
            else:
                self.logvar = nn.Parameter(torch.randn(1, self.z_dim).cuda())
        else:
            self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)


    def forward(self, h, z, s):
        # combine h, z, and s
        # embed h
        eh = h.view(-1, self.encoder_dim)
        eh = self.fc_h(eh)
        eh = eh.view(self.batch_size, self.sample_size, self.hidden_dim)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.sample_size, self.hidden_dim)
        else:
            if next(self.parameters()).is_cuda:
                ez = Variable(torch.zeros(eh.size()).cuda())
            else:
                ez = Variable(torch.zeros(eh.size()))

        # embed s and expand for broadcast addition
        es = self.fc_s(s)
        es = es.view(self.batch_size, 1, self.hidden_dim).expand_as(eh)

        # sum and reshape
        e = eh + ez + es
        e = e.view(self.batch_size * self.sample_size, self.hidden_dim)
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

# LATENT DECODER p(z|z, s)
class LatentDecoder(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features, encoder_dim,
                 n_hidden, hidden_dim, o_dim, z_dim, nonlinearity, args):
        super().__init__()

        self.args = args
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.o_dim = o_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_s = nn.Linear(self.o_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_block1 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)
        self.fc_block2 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)

        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

        if args.shared_var:
            self.fc_params = nn.Linear(self.hidden_dim, self.z_dim)
            if self.args.no_cuda:
                self.logvar = nn.Parameter(torch.randn(1, self.z_dim))
            else:
                self.logvar = nn.Parameter(torch.randn(1, self.z_dim).cuda())
        else:
            self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)


    def forward(self, z, s):
        # combine z and s
        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.sample_size, self.hidden_dim)
        else:
            if next(self.parameters()).is_cuda:
                ez = Variable(torch.zeros(
                    self.batch_size, 1, self.hidden_dim).cuda())
            else:
                ez = Variable(torch.zeros(
                    self.batch_size, 1, self.hidden_dim))

        # embed s
        es = self.fc_s(s)
        es = es.view(self.batch_size, 1, self.hidden_dim).expand_as(ez)

        # sum and reshape
        e = ez + es
        e = e.view(-1, self.hidden_dim)
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



# p(s_{t+1} | z, s_t, s)
# Note: Ours is a deterministic mapping to s_t+1, because we need to differentiate over it

class ObservationLSTMDecoder(nn.Module):

    def __init__(self, batch_size, sample_size, n_features, nonlinearity, args=None):

        super(ObservationLSTMDecoder, self).__init__()

        self.batch_size = batch_size
        self.sample_size = sample_size

        self.n_features = n_features
        self.hidden_dim = args.hidden_dim_traj
        self.n_stochastic = args.n_stochastic
        self.o_dim = args.o_dim
        self.z_dim = args.z_dim
        self.n_hidden = args.n_hidden_traj
        self.nonlinearity = nonlinearity

        self.args = args
        self.no_initial_state = args.no_initial_state

        if not args.effect_only_decoder:
            self.fc_s = nn.Linear(self.o_dim, self.hidden_dim)
            if args.concat_oz:
                self.fc_e_traj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        # modules
        self.fc_zs = nn.Linear(self.n_stochastic * self.z_dim, self.hidden_dim)

        if self.no_initial_state:
            self.lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        else:
            self.fc_st1 = nn.Linear(self.n_features, self.hidden_dim)
            self.fc_st2 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                nonlinearity=self.nonlinearity)

            self.lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        if not self.args.no_cuda:
            self.logvar = nn.Parameter(torch.randn(1, self.n_features).cuda())
        else:
            self.logvar = nn.Parameter(torch.randn(1, self.n_features))

        if self.args.emb_non_linear_lstm:
            self.fc_non_linear = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                nonlinearity=self.nonlinearity)

        self.fc_out = nn.Linear(self.hidden_dim, self.n_features)


    def forward(self, zs, s, s0, n_steps):

        if self.no_initial_state:
            n_steps += 1

        # Note: we ignore s0 in output
        out_mean = torch.zeros(self.batch_size, self.sample_size, n_steps, self.n_features)

        if next(self.parameters()).is_cuda:
            out_mean = out_mean.cuda()

        ezs = self.fc_zs(zs)
        ezs = ezs.view(self.batch_size, self.sample_size, self.hidden_dim)

        if not self.args.effect_only_decoder:
            es = self.fc_s(s)
            es = es.view(self.batch_size, 1, self.hidden_dim).expand_as(ezs)
            if self.args.concat_oz:
                e_traj_option = torch.cat([ezs, es], dim=-1)
                e_traj_option = self.fc_e_traj(e_traj_option)
            else:
                e_traj_option = ezs + es
        else:
            e_traj_option = ezs

        e_traj_option = self.nonlinearity(e_traj_option)

        # Note: we can concatenate traj+option embedding with state embedding
        # because they are independent
        if not self.no_initial_state:
            st = s0

        for step in range(n_steps):
            if self.no_initial_state:
                e = e_traj_option.view(-1, self.hidden_dim)
            else:
                st = st.view(-1, self.n_features)
                e_st = self.fc_st1(st)
                e_st = self.fc_st2(e_st)
                e_st = e_st.view(self.batch_size, self.sample_size, self.hidden_dim)
                e = e_traj_option * e_st
                e = e.view(-1, self.hidden_dim)

            if step == 0:
                h_n, c_n = self.lstm_cell(e)
            else:
                h_n, c_n = self.lstm_cell(e, (h_n, c_n))

            if self.args.emb_non_linear_lstm:
                s_n = self.fc_non_linear(h_n)
                s_n = self.fc_out(s_n)
            else:
                s_n = self.fc_out(h_n)
            out_mean[:,:,step,:] = s_n.view(self.batch_size, self.sample_size, self.n_features)

            if not self.no_initial_state:
                st = s_n

        return out_mean, self.logvar.expand_as(out_mean)

class ObservationMLPDecoder(nn.Module):

    def __init__(self, batch_size, sample_size, n_features, nonlinearity, args=None):
        super().__init__()

        self.batch_size = batch_size
        self.sample_size = sample_size

        self.n_features = n_features
        self.hidden_dim = args.hidden_dim_traj
        self.n_stochastic = args.n_stochastic
        self.o_dim = args.o_dim
        self.z_dim = args.z_dim
        self.n_hidden = args.n_hidden_traj
        self.nonlinearity = nonlinearity

        self.args = args

        if not args.effect_only_decoder:
            self.fc_s = nn.Linear(self.o_dim, self.hidden_dim)

        # modules
        self.fc_zs = nn.Linear(self.n_stochastic * self.z_dim, self.hidden_dim)

        if not self.args.no_cuda:
            self.logvar = nn.Parameter(torch.randn(1, self.n_features).cuda())
        else:
            self.logvar = nn.Parameter(torch.randn(1, self.n_features))

        if self.args.emb_non_linear_lstm:
            self.fc_non_linear = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                nonlinearity=self.nonlinearity)

        self.fc_out = nn.Linear(self.hidden_dim, self.n_features)


    def forward(self, zs, s):
        # Note: we ignore s0 in output
        ezs = self.fc_zs(zs)
        ezs = ezs.view(self.batch_size, self.sample_size, self.hidden_dim)

        if not self.args.effect_only_decoder:
            es = self.fc_s(s)
            es = es.view(self.batch_size, 1, self.hidden_dim).expand_as(ezs)
            e_traj_option = ezs + es
        else:
            e_traj_option = ezs

        e_traj_option = self.nonlinearity(e_traj_option)

        if self.args.emb_non_linear_lstm:
            e = self.fc_non_linear(e_traj_option)
            e = self.fc_out(e)
        else:
            e = self.fc_out(e_traj_option)
        out_mean = e.view(self.batch_size, self.sample_size, 1, self.n_features)

        return out_mean, self.logvar.expand_as(out_mean)

# Observation Decoder p(x|z_x)
class ObservationCNNDecoder_84(nn.Module):
    """
        Decode LSTM output to image using transposed convolutions
    """
    def __init__(self, nonlinearity, input_channels=2, args=None):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.input_channels = input_channels
        self.args = args
        # Input shape is N * 32 * 3 * 3
        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=32, out_channels=32),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            # shape is now (-1, 32, 6, 6)
            Conv2d3x3(in_channels=32, out_channels=32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1),
            Conv2d3x3(in_channels=32, out_channels=32),
            # shape is now (-1, 32, 11, 11)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            Conv2d3x3(in_channels=16, out_channels=16),
            # shape is now (-1, 16, 21, 21)
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            # shape is now (-1, 16, 42, 42)
            Conv2d3x3(in_channels=16, out_channels=16),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
            # shape is now (-1, 16, 84, 84)
        ])
        if self.args.emb_use_batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
            ])

        self.conv_final = nn.Conv2d(16, input_channels + 1, kernel_size=1)

        if self.args.no_cuda:
            self.logvar = nn.Parameter(torch.randn(1, input_channels, 84, 84))
        else:
            self.logvar = nn.Parameter(torch.randn(1, input_channels, 84, 84).cuda())

    def forward(self, h_mean):
        e = h_mean.view(-1, 32, 3, 3)

        if not self.args.emb_use_batch_norm:
            for conv in self.conv_layers:
                e = conv(e)
                e = self.nonlinearity(e)
        else:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                e = conv(e)
                e = bn(e)
                e = self.nonlinearity(e)

        e = self.conv_final(e)
        e = torch.sigmoid(e)
        im = e[:,:self.input_channels,:,:] - 0.5    # This means we are dealing with -0.5 to +0.5 values
        mask = e[:, self.input_channels:,:,:]

        return im, mask, self.logvar.expand_as(im)



# Observation Decoder p(x|z_x)
class ObservationCNNDecoder_64(nn.Module):
    """
        Decode LSTM output to image using transposed convolutions
    """
    def __init__(self, nonlinearity, input_channels=1, args=None):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.input_channels = input_channels
        self.args = args
        # Input shape is N * 32 * 2 * 2
        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=32, out_channels=32),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            # shape is now (-1, 32, 4, 4)
            Conv2d3x3(in_channels=32, out_channels=32),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            Conv2d3x3(in_channels=32, out_channels=32),
            # shape is now (-1, 32, 8, 8)
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            Conv2d3x3(in_channels=16, out_channels=16),
            # shape is now (-1, 16, 16, 16)
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            # shape is now (-1, 16, 32, 32)
            Conv2d3x3(in_channels=16, out_channels=16),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
            # shape is now (-1, 16, 64, 64)
        ])
        if self.args.emb_use_batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=32),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
                nn.BatchNorm2d(num_features=16),
            ])

        self.conv_final = nn.Conv2d(16, input_channels + 1, kernel_size=1)

        if self.args.no_cuda:
            self.logvar = nn.Parameter(torch.randn(1, input_channels, 64, 64))
        else:
            self.logvar = nn.Parameter(torch.randn(1, input_channels, 64, 64).cuda())

    def forward(self, h_mean):
        e = h_mean.view(-1, 32, 2, 2)

        if not self.args.emb_use_batch_norm:
            for conv in self.conv_layers:
                e = conv(e)
                e = self.nonlinearity(e)
        else:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                e = conv(e)
                e = bn(e)
                e = self.nonlinearity(e)

        e = self.conv_final(e)
        e = torch.sigmoid(e)
        im = e[:,:self.input_channels,:,:] - 0.5    # This means we are dealing with -0.5 to +0.5 values
        mask = e[:, self.input_channels:,:,:]

        return im, mask, self.logvar.expand_as(im)





# Observation Decoder p(x|z_x)
class ObservationCNNDecoder_48(nn.Module):
    """
        Decode LSTM output to image using transposed convolutions
    """
    def __init__(self, nonlinearity, input_channels=1, args=None):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.input_channels = input_channels

        self.args = args

        if not args.deeper_encoder:
            # Input shape is N * 32 * 3 * 3
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=32, out_channels=32),
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                # shape is now (-1, 32, 6, 6)
                Conv2d3x3(in_channels=32, out_channels=32),
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                Conv2d3x3(in_channels=32, out_channels=32),
                # shape is now (-1, 32, 12, 12)
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                Conv2d3x3(in_channels=16, out_channels=16),
                # shape is now (-1, 16, 24, 24)
                nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
                # shape is now (-1, 16, 48, 48)
            ])
            if self.args.emb_use_batch_norm:
                self.bn_layers = nn.ModuleList([
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=16),
                    nn.BatchNorm2d(num_features=16),
                    nn.BatchNorm2d(num_features=16),
                ])
            self.conv_final = nn.Conv2d(16, input_channels + int(self.args.image_mask), kernel_size=1)
        else:
            # Input shape is N * 256 * 3 * 3
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=256, out_channels=256),
                Conv2d3x3(in_channels=256, out_channels=256),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                # shape is now (-1, 256, 6, 6)
                Conv2d3x3(in_channels=256, out_channels=128),
                Conv2d3x3(in_channels=128, out_channels=128),
                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                # shape is now (-1, 128, 12, 12)
                Conv2d3x3(in_channels=128, out_channels=64),
                Conv2d3x3(in_channels=64, out_channels=64),
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                # shape is now (-1, 64, 24, 24)
                Conv2d3x3(in_channels=64, out_channels=32),
                Conv2d3x3(in_channels=32, out_channels=32),
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                # shape is now (-1, 32, 48, 48)
            ])
            if self.args.emb_use_batch_norm:
                self.bn_layers = nn.ModuleList([
                    nn.BatchNorm2d(num_features=256),
                    nn.BatchNorm2d(num_features=256),
                    nn.BatchNorm2d(num_features=256),
                    nn.BatchNorm2d(num_features=128),
                    nn.BatchNorm2d(num_features=128),
                    nn.BatchNorm2d(num_features=128),
                    nn.BatchNorm2d(num_features=64),
                    nn.BatchNorm2d(num_features=64),
                    nn.BatchNorm2d(num_features=64),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                    nn.BatchNorm2d(num_features=32),
                ])
            self.conv_final = nn.Conv2d(32, input_channels + int(self.args.image_mask), kernel_size=1)

        if self.args.no_cuda:
            self.logvar = nn.Parameter(torch.randn(1, input_channels, 48, 48))
        else:
            self.logvar = nn.Parameter(torch.randn(1, input_channels, 48, 48).cuda())

    def forward(self, h_mean):
        if not self.args.deeper_encoder:
            e = h_mean.view(-1, 32, 3, 3)
        else:
            e = h_mean.view(-1, 256, 3, 3)

        if not self.args.emb_use_batch_norm:
            for conv in self.conv_layers:
                e = conv(e)
                e = self.nonlinearity(e)
        else:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                e = conv(e)
                e = bn(e)
                e = self.nonlinearity(e)

        e = self.conv_final(e)
        e = torch.sigmoid(e)
        if self.args.image_mask:
            im = e[:,:self.input_channels,:,:] - 0.5    # This means we are dealing with -0.5 to +0.5 values
            mask = e[:, self.input_channels:,:,:]
        else:
            im = e - 0.5    # This means we are dealing with -0.5 to +0.5 values
            mask = None
        return im, mask, self.logvar.expand_as(im)
