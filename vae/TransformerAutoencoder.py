from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)


    def forward(self, x, alibi_bias=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if alibi_bias is not None:
            attn_mask = alibi_bias.unsqueeze(1).expand(-1, self.heads, -1, -1)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.1 if self.training else 0.0,
            is_causal=False
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads),
                FeedForward(dim, mlp_dim),
            ]))

    def forward(self, x, algorithm_distance_matricies=None):
        for attn, ff in self.layers:

            attn_out = attn(x, algorithm_distance_matricies)

            x = attn_out + x
            x = ff(x) + x

        x = self.norm(x)

        return x


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_size, latent_space, d_model, depth, heads, mlp_dim, reparameterization=False, adjacency_matrix=None, device="cuda"):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.input_projection = nn.Linear(1, d_model)
        self.encoder = Transformer(d_model, depth, heads, mlp_dim)

        self.algorithm_index = 176

        self.reparameterization = reparameterization

        self.to_latent = nn.Linear(d_model * input_size, latent_space)

        if reparameterization:
            self.extra_logvar = nn.Linear(d_model * input_size, latent_space)

        self.from_latent = nn.Linear(latent_space, d_model * input_size)
        self.decoder = Transformer(d_model, depth, heads, mlp_dim)
        self.output_projection = nn.Linear(d_model * input_size, input_size)

        self.learned_embeddings_encoder = nn.Parameter(torch.randn(input_size, d_model))
        self.learned_embeddings_decoder = nn.Parameter(torch.randn(input_size, d_model))

        self.num_nodes = 6

    def generate(self, latent):
        B, T, F = latent.shape
        x_p = self.from_latent(latent)

        x_p = x_p.view(B, T, F)
        x_p += self.learned_embeddings_decoder.expand(B, x_p.shape[1], x_p.shape[2])
        x_p = self.decoder(x_p)

        x_p = x_p.view(B, T * F)
        x_p = self.output_projection(x_p)

        return x_p

    def reparameterize(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)
        return mean + torch.exp(logvar / 2) * epsilon

    def forward(self, x):
        x_p = self.input_projection(x.unsqueeze(-1))
        B, T, F = x_p.shape

        x_p += self.learned_embeddings_encoder.expand(B, x_p.shape[1], x_p.shape[2])
        x_p = self.encoder(x_p)

        # Latent Stuff
        x_p = x_p.view(B, T * F)
        latent = self.to_latent(x_p)
        logvar = torch.ones((1), device=self.device)

        if self.reparameterization:
            logvar = self.extra_logvar(x_p)
            latent = self.reparameterize(latent, logvar)

        x_p = self.from_latent(latent)

        x_p = x_p.view(B, T, F)
        x_p += self.learned_embeddings_decoder.expand(B, x_p.shape[1], x_p.shape[2])
        x_p = self.decoder(x_p)

        x_p = x_p.view(B, T * F)
        x_p = self.output_projection(x_p)

        return x_p, latent, logvar