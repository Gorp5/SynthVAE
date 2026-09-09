from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricBiasUpdater(nn.Module):

    def __init__(
        self,
        d_model: int,
        geom_dim: int = 64,
        alpha: float = 1.0,
        beta: float = 0.1,
        clamp_value: float = 10.0,
        learnable_scales: bool = True,
    ):
        super().__init__()

        self.geom_proj = nn.Linear(d_model, geom_dim, bias=False)

        if learnable_scales:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta", torch.tensor(beta))


        self.clamp_value = clamp_value

        self.geom_norm = nn.LayerNorm(geom_dim)

        # Initialize near zero so early training = no geometry modulation
        nn.init.normal_(self.geom_proj.weight, mean=0.0, std=1e-3)

    def forward(self, H, B_prev):
        G = self.geom_proj(H)

        # G = self.geom_norm(G)

        # ||x - y||^2 = x^2 + y^2 - 2xy
        G_sq = (G ** 2).sum(dim=-1, keepdim=True)

        dist = G_sq + G_sq.transpose(1, 2) - 2 * torch.matmul(G, G.transpose(1, 2))
        dist = torch.clamp(dist, min=0.0)

        sigma = dist.mean().detach()
        delta_B = -torch.exp(-dist / (2 * sigma**2))

        B_next = B_prev + torch.tanh(self.beta) * delta_B
        B_next = 10 * torch.tanh(B_next / 10)

        return B_next

class SparseAutoencoderBlock(nn.Module):
    def __init__(self, in_dim, sparse_dim, rho=0.05):
        super().__init__()

        self.encoder = nn.Linear(in_dim, sparse_dim)
        self.decoder = nn.Linear(sparse_dim, in_dim)

        self.rho = rho  # target sparsity

    def forward(self, l):
        # Encode
        sparse_code = F.relu(self.encoder(l))  # enforce non-negativity

        # Decode
        reconstructed = self.decoder(sparse_code)

        # -----------------------
        # L1 sparsity loss
        # -----------------------
        l1_loss = sparse_code.abs().mean()

        # -----------------------
        # KL sparsity loss
        # -----------------------
        # Compute average activation per unit over batch
        rho_hat = sparse_code.mean(dim=0)  # [sparse_dim]

        # Clamp to avoid log(0)
        eps = 1e-8
        rho_hat = torch.clamp(rho_hat, eps, 1 - eps)

        rho = torch.full_like(rho_hat, self.rho)

        kl_loss = torch.sum(
            rho * torch.log(rho / rho_hat) +
            (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )

        return reconstructed, sparse_code, l1_loss, kl_loss

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
    def __init__(self, dim, heads=4, sparsity_weight=0.0):
        super().__init__()

        dim_head = dim / heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # sparsity
        self.sparsity_weight = sparsity_weight

    def forward(self, x, alibi_bias=None, alg_film=None):
        x = self.norm(x)

        if alg_film is not None:
            gamma, beta = alg_film
            x = gamma[:, None, :] * x + beta[:, None, :]

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if alibi_bias is not None:
            alibi_bias = alibi_bias.unsqueeze(1).expand(-1, 4, -1, -1)
            dots = dots + alibi_bias

        attn = self.attend(dots)

        # sparsity_loss = None
        # if return_attn_loss and self.sparsity_weight > 0.0:
        #     eps = 1e-8
        #     entropy = -(attn * (attn + eps).log()).sum(dim=-1)
        #     sparsity_loss = entropy.mean() * self.sparsity_weight

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, sparsity_weight=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, sparsity_weight=sparsity_weight),
                FeedForward(dim, mlp_dim),
                MetricBiasUpdater(d_model=dim,geom_dim=7, learnable_scales=True)
            ]))

    def forward(self, x, alg_film=None, global_film=None, algorithm_distance_matricies=None):
        alibi_bias = algorithm_distance_matricies
        total_sparsity_loss = 0.0

        for attn, ff, alibi_bias_function in self.layers:

            attn_out = attn(
                x,
                alibi_bias,
                alg_film=alg_film,
            )

            x = attn_out + x

            if global_film is not None:
                gamma, beta = global_film
                x = gamma[:, None, :] * x + beta[:, None, :]

            x = ff(x) + x

            alibi_bias = alibi_bias_function(x, alibi_bias)

        x = self.norm(x)

        return x, alibi_bias


class GraphTransformerAutoencoderAlibi(nn.Module):
    def __init__(
            self,
            input_size,
            latent_space,
            d_model,
            depth,
            heads,
            mlp_dim,
            num_algorithms=32,
            num_global_params=24,
            sparsity_weight=0.0,
            reparameterization=False,
            add_noise=False,
            algorithm_distance_matricies=False,
            device="cuda",
            sparse=False,
            sparse_latent_space=16,
            learned_alibi_dists=False,
            masking=None,
            mask_ratio=0.05,
            hard_prediction=True,
            geom_dim=16,
            algorithm_index=176
    ):
        super().__init__()

        self.masking = masking
        self.mask_ratio = mask_ratio
        self.device = device
        self.input_size = input_size
        self.input_projection = nn.Linear(input_size, d_model)
        self.encoder = Transformer(d_model, depth, heads, mlp_dim, sparsity_weight=sparsity_weight)

        self.reparameterization = reparameterization

        self.to_latent = nn.Linear(d_model, latent_space)

        if reparameterization:
            self.extra_logvar = nn.Linear(d_model, latent_space)

        self.from_latent = nn.Linear(latent_space, d_model)
        self.decoder = Transformer(d_model, depth, heads, mlp_dim, sparsity_weight=sparsity_weight)
        self.output_projection = nn.Linear(d_model, input_size)

        self.enc_global_film_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.enc_global_film_beta = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())

        self.dec_global_film_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.dec_global_film_beta = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())

        self.hard_prediction = hard_prediction

        if self.hard_prediction:
            self.alg_pred = nn.Linear(latent_space, num_algorithms)
        else:
            self.num_tokens = 7  # 6 nodes + 1 global
            self.geom_dim = geom_dim  # small geometry space (tune 8–32)

            self.latent_to_geometry = nn.Linear(
                latent_space,
                self.num_tokens * self.geom_dim
            )

            self.biases_to_token = nn.Linear(49, d_model)

            # small initialization for stability
            nn.init.normal_(self.latent_to_geometry.weight, mean=0.0, std=1e-3)

            self.alg_pred = nn.Sequential(
                nn.Linear(21, 32)
                # nn.Linear(21, 128),
                # nn.GELU(),
                # nn.Linear(128, 32)
            )

        self.global_embedding = nn.Sequential(
            nn.Linear(num_global_params, d_model),
            # nn.SiLU(),
            # nn.Linear(d_model, d_model)
        )

        self.global_pred = nn.Sequential(
            nn.Linear(d_model, num_global_params)
            # nn.Linear(d_model, d_model),
            # nn.SiLU(),
            # nn.Linear(d_model, num_global_params)
        )

        self.learned_embeddings = nn.Parameter(torch.randn(7, d_model))

        if algorithm_distance_matricies is not None:
            self.algorithm_distance_matricies = algorithm_distance_matricies.to("cuda")
        else:
            self.algorithm_distance_matricies = None

        if learned_alibi_dists:
            algo_dists = torch.randn((num_algorithms, 7, 7))
            self.algorithm_distance_matricies = nn.Parameter(algo_dists).to("cuda")

        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.num_nodes = 6

        self.algorithm_index = algorithm_index
        self.sparse = sparse
        # self.sparse_block = SparseAutoencoderBlock(latent_space, sparse_latent_space, rho=0.1)
        self.use_add = add_noise

    def reparameterize(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)
        # return mean + var * epsilon
        return mean + torch.exp(logvar / 2) * epsilon

    def add(self, latent, beta):
        epsilon = torch.randn(latent.shape).to(self.device) * beta
        return latent + epsilon * beta

    def noise_mask(self, x, mask_ratio=0.05, noise_std=0.05):
        B, F = x.shape
        mask = torch.rand(B, F, device=x.device) < mask_ratio
        mask[:, self.algorithm_index] = False
        noise = torch.randn_like(x) * noise_std
        x = x.clone()
        x[mask] += noise[mask]
        return x

    def zero_mask(self, x, mask_ratio=0.05):
        B, F = x.shape
        mask = torch.rand(B, F, device=x.device) < mask_ratio
        mask[:, self.algorithm_index] = False
        return x.masked_fill(mask, 0.0)

    def generate(self, latent):
        node_param_dim = self.input_size
        num_nodes = 6

        if self.hard_prediction:
            alg_probs = self.alg_pred(latent)
            algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(alg_probs, dim=1)]
        else:
            # latent: [B, latent_dim]
            B = latent.size(0)

            G = self.latent_to_geometry(latent)  # [B, 7 * geom_dim]
            G = G.view(B, self.num_tokens, self.geom_dim)  # [B, 7, geom_dim]

            # ||x - y||^2 = x^2 + y^2 - 2xy
            G_sq = (G ** 2).sum(dim=-1, keepdim=True)  # [B, 7, 1]

            algorithm_distance_matricies = (
                    G_sq +
                    G_sq.transpose(1, 2) -
                    2 * torch.matmul(G, G.transpose(1, 2))
            )

            # numerical safety
            algorithm_distance_matricies = torch.clamp(
                algorithm_distance_matricies,
                min=0.0
            )

            # zero diagonal
            algorithm_distance_matricies = (
                    algorithm_distance_matricies -
                    torch.diag_embed(torch.diagonal(
                        algorithm_distance_matricies, dim1=1, dim2=2
                    ))
            )

        x_p = self.from_latent(latent)
        x_p = x_p.unsqueeze(1).repeat(1, self.num_nodes + 1, 1)

        x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
        x_p, final_alibi = self.decoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        global_hat = self.global_pred(x_p[:, 0])
        x_p = self.output_projection(x_p[:, 1:])
        node_end = num_nodes * node_param_dim
        x_p = x_p.view(B, node_end)

        new_algo_index = 8
        if not self.hard_prediction:
            indices = torch.triu_indices(7, 7, offset=1)
            struct_features = final_alibi[:, indices[0], indices[1]]
            alg_probs = self.alg_pred(struct_features)

        x_recon = torch.cat([
            x_p,
            global_hat[:, :new_algo_index],
            alg_probs,
            global_hat[:, new_algo_index:],
        ], dim=1)

        return x_recon

    def encode(self, x):
        node_param_dim = self.input_size
        num_nodes = 6

        node_end = num_nodes * node_param_dim

        if self.masking is not None:
            if self.masking == "noise":
                x = self.noise_mask(x, mask_ratio=self.mask_ratio)
            elif self.masking == "zero":
                x = self.zero_mask(x, mask_ratio=self.mask_ratio)

        node_flat = x[:, :node_end]
        algorithm_id = x[:, self.algorithm_index:self.algorithm_index + 32].to(torch.int)
        global_params_half_one = x[:, node_end:self.algorithm_index]
        global_params_half_two = x[:, self.algorithm_index + 32:]
        global_params = torch.cat((global_params_half_one, global_params_half_two), dim=1)

        if self.algorithm_distance_matricies is not None:
            algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(algorithm_id, dim=1)]
        else:
            algorithm_distance_matricies = None

        B, F = node_flat.shape

        nodes = node_flat.view(
            B,
            num_nodes,
            node_param_dim
        )

        x_p = self.input_projection(nodes)

        global_emb_enc = self.global_embedding(global_params).unsqueeze(1)

        x_p = torch.cat([global_emb_enc, x_p], dim=1)

        x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
        x_p, biases = self.encoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        # Pool over nodes to get graph-level latent
        x_p = x_p.mean(dim=1)
        latent = self.to_latent(x_p)
        return latent

    def forward(self, x, return_attn_loss=True, beta=0.1):
        node_param_dim = self.input_size
        num_nodes = 6

        node_end = num_nodes * node_param_dim

        if self.masking is not None:
            if self.masking == "noise":
                x = self.noise_mask(x, mask_ratio=self.mask_ratio)
            elif self.masking == "zero":
                x = self.zero_mask(x, mask_ratio=self.mask_ratio)

        node_flat = x[:, :node_end]
        algorithm_id = x[:, self.algorithm_index:self.algorithm_index + 32].to(torch.int)
        global_params_half_one = x[:, node_end:self.algorithm_index]
        global_params_half_two = x[:, self.algorithm_index + 32:]
        global_params = torch.cat((global_params_half_one, global_params_half_two), dim=1)

        if self.algorithm_distance_matricies is not None:
            algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(algorithm_id, dim=1)]
        else:
            algorithm_distance_matricies = None

        B, F = node_flat.shape

        nodes = node_flat.view(
            B,
            num_nodes,
            node_param_dim
        )

        x_p = self.input_projection(nodes)

        global_emb_enc = self.global_embedding(global_params).unsqueeze(1)

        x_p = torch.cat([global_emb_enc, x_p], dim=1)

        x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
        x_p, biases = self.encoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        biases = self.biases_to_token(biases.view(B, -1))

        # Pool over nodes to get graph-level latent
        x_p = x_p.mean(dim=1)
        x_p += biases

        latent = self.to_latent(x_p)

        sparse_loss = 0
        if self.sparse:
            latent, sparse_latent, l1_loss, kl_loss = self.sparse_block(latent)
            sparse_loss = l1_loss + kl_loss

        logvar = torch.ones((1), device=self.device)
        if self.reparameterization:
            logvar = self.extra_logvar(x_p)
            latent = self.reparameterize(latent, logvar)

        if hasattr(self, "use_add") and self.use_add:
            latent = self.add(latent, beta)

        if self.hard_prediction:
            alg_probs = self.alg_pred(latent)
            algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(alg_probs, dim=1)]
        else:
            # latent: [B, latent_dim]
            B = latent.size(0)

            G = self.latent_to_geometry(latent)  # [B, 7 * geom_dim]
            G = G.view(B, self.num_tokens, self.geom_dim)  # [B, 7, geom_dim]

            # ||x - y||^2 = x^2 + y^2 - 2xy
            G_sq = (G ** 2).sum(dim=-1, keepdim=True)  # [B, 7, 1]

            algorithm_distance_matricies = (
                    G_sq +
                    G_sq.transpose(1, 2) -
                    2 * torch.matmul(G, G.transpose(1, 2))
            )

            # numerical safety
            algorithm_distance_matricies = torch.clamp(
                algorithm_distance_matricies,
                min=0.0
            )

            # optional: zero diagonal (cleaner)
            algorithm_distance_matricies = (
                    algorithm_distance_matricies -
                    torch.diag_embed(torch.diagonal(
                        algorithm_distance_matricies, dim1=1, dim2=2
                    ))
            )


        x_p = self.from_latent(latent)
        x_p = x_p.unsqueeze(1).repeat(1, num_nodes + 1, 1)

        x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
        x_p, final_alibi = self.decoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        global_hat = self.global_pred(x_p[:, 0])
        x_p = self.output_projection(x_p[:, 1:])
        x_p = x_p.view(B, F)

        new_algo_index = 8
        if not self.hard_prediction:
            indices = torch.triu_indices(7, 7, offset=1)
            struct_features = final_alibi[:, indices[0], indices[1]]
            alg_probs = self.alg_pred(struct_features)

        x_recon = torch.cat([
            x_p,
            global_hat[:, :new_algo_index],
            alg_probs,
            global_hat[:, new_algo_index:],
        ], dim=1)

        return x_recon, latent, logvar