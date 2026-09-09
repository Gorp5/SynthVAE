import torch
from torch import nn
from torch.functional import F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, DROPOUT_PROB):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Dropout(DROPOUT_PROB),
            nn.SELU(),
            nn.Linear(middle_channels, out_channels),
            nn.Dropout(DROPOUT_PROB),
        )
        self.skip_connection = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.selu(self.skip_connection(x) + self.block(x))

def res_blocks(n, LATENT_DIM=16, INPUT_DIM=225, for_encoder=False, DROPOUT_PROB=0.033):
    a = LATENT_DIM
    b = INPUT_DIM
    diff = b - a
    # channel_count = 3 * (n - (n // 3)) - (n % 3)
    channel_count = 2 * n
    d_size = diff / channel_count

    def calc_size(i, a, s):
        size = round(a + i * s)
        print(size)  # DEBUG
        return size

    blocks = []
    if for_encoder:
        for i in range(n - 1, -1, -1):
            j = i + 1
            i_channels = calc_size(2 * j, a, d_size)
            h_channels = calc_size(2 * j - 1, a, d_size)
            o_channels = calc_size(2 * j - 2, a, d_size)
            block = ResidualBlock(i_channels, h_channels, o_channels, DROPOUT_PROB=DROPOUT_PROB)
            blocks.append(block)
    else:
        for i in range(0, n, 1):
            j = i + 1
            i_channels = calc_size(2 * j - 2, a, d_size)
            h_channels = calc_size(2 * j - 1, a, d_size)
            o_channels = calc_size(2 * j, a, d_size)
            block = ResidualBlock(i_channels, h_channels, o_channels, DROPOUT_PROB=DROPOUT_PROB)
            blocks.append(block)
    return blocks


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=225, output_dim=225, latent_dim=16, RESIDUAL_BLOCKS=5, DROPOUT_PROB=0.033):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module("linear_first", nn.Linear(input_dim, input_dim))
        self.encoder.add_module("dropout_first", nn.Dropout(DROPOUT_PROB))
        self.encoder.add_module("selu_first", nn.SELU())
        encoder_blocks = res_blocks(RESIDUAL_BLOCKS, for_encoder=True)
        for i, block in enumerate(encoder_blocks):
            self.encoder.add_module("ResidualBlock" + str(i), block)
        self.encoder.add_module("linear_last", nn.Linear(latent_dim, latent_dim))
        self.encoder.add_module("dropout_last", nn.Dropout(DROPOUT_PROB))
        self.encoder.add_module("selu_last", nn.SELU())

        # Latent mean & logvar
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module("linear_first", nn.Linear(latent_dim, latent_dim))
        self.decoder.add_module("dropout_first", nn.Dropout(DROPOUT_PROB))
        self.decoder.add_module("selu_first", nn.SELU())
        decoder_blocks = res_blocks(RESIDUAL_BLOCKS, for_encoder=False)
        for i, block in enumerate(decoder_blocks):
            self.decoder.add_module("ResidualBlock" + str(i), block)
        self.decoder.add_module("linear_last", nn.Linear(output_dim, output_dim))
        self.decoder.add_module("dropout_last", nn.Dropout(DROPOUT_PROB))
        self.decoder.add_module("sigmoid_last", nn.Sigmoid())

        # print(self.encoder) # DEBUG
        # print(self.decoder) # DEBUG
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        self.use_sampling = True

    def toggle_sampling(self):
        self.use_sampling = not self.use_sampling

    def forward(self, x):
        if self.use_sampling:
            x1 = self.encoder(x)
            mean = self.fc_mu(x1)
            logvar = self.fc_logvar(x1)
            # convert to standard deviation
            std_dev = torch.exp(logvar * 0.5)
            std_dev = torch.clamp(std_dev, 0.000000001, 10000000)  # don't allow negative standard deviations
            # sample from normal distribution
            z_dist = torch.distributions.normal.Normal(loc=mean, scale=std_dev)
            z = z_dist.rsample()

            # reconstruct using decoder
            recon_x = self.decoder(z)

            return recon_x, mean, logvar
        else:
            x1 = self.encoder(x)
            mean = self.fc_mu(x1)
            recon_x = self.decoder(mean)
            return recon_x, mean, None
