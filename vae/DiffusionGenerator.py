import torch
import torch.nn as nn
from diffusers import AudioLDM2Pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchConditioningAdapter(nn.Module):
    def __init__(
        self,
        number_of_params,
        stream_1_dim,
        stream_2_dim,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
    ):
        super().__init__()

        self.num_params = number_of_params

        # Each scalar parameter becomes a token embedding
        self.input_projection = nn.Linear(
            1,
            hidden_dim,
        )

        # One learned embedding for each parameter position.
        # This tells the Transformer that token 0 is, e.g., operator 1
        # frequency, token 1 is operator 1 coarse, etc.
        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                number_of_params,
                hidden_dim,
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Transformer output → AudioLDM2 stream 1
        self.stream_1_projection = nn.Linear(
            hidden_dim,
            stream_1_dim,
        )

        # Replaces the second AudioLDM2 text-conditioning stream
        self.learned_token = nn.Parameter(
            torch.randn(
                1,
                1,
                stream_2_dim,
            ) * 0.02
        )

    def forward(self, params):
        """
        Parameters
        ----------
        params:
            [B, num_params]

        Returns
        -------
        patch_condition:
            [B, num_params, stream_1_dim]

        learned_condition:
            [B, 1, stream_2_dim]
        """

        # ---------------------------------------------------------
        # Make sure params is [B, num_params]
        # ---------------------------------------------------------

        if params.ndim == 3 and params.shape[-1] == 1:
            params = params.squeeze(-1)

        if params.ndim != 2:
            raise ValueError(
                f"Expected params to have shape "
                f"[B, {self.num_params}], "
                f"but got {tuple(params.shape)}"
            )

        B, T = params.shape

        if T != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters, "
                f"but got {T}"
            )

        # ---------------------------------------------------------
        # Each parameter becomes its own token
        # ---------------------------------------------------------

        # [B, 155]
        x = params.unsqueeze(-1)

        # [B, 155, 1] → [B, 155, hidden_dim]
        x = self.input_projection(x)

        # Add parameter-specific positional identity
        x = x + self.position_embedding.to(
            device=x.device,
            dtype=x.dtype,
        )

        # ---------------------------------------------------------
        # Parameter Transformer
        # ---------------------------------------------------------

        x = self.transformer(x)

        x = self.norm(x)

        # ---------------------------------------------------------
        # AudioLDM2 conditioning stream 1
        # ---------------------------------------------------------

        patch_condition = self.stream_1_projection(x)

        # ---------------------------------------------------------
        # AudioLDM2 conditioning stream 2
        # ---------------------------------------------------------

        learned_condition = self.learned_token.expand(
            B,
            -1,
            -1,
        )

        learned_condition = learned_condition.to(
            device=x.device,
            dtype=x.dtype,
        )

        return patch_condition, learned_condition


class DX7AudioLDM2(nn.Module):
    """
    AudioLDM2 adapted for DX7 patch-conditioned audio generation.

    The original text-conditioning stack is discarded. Instead:

        DX7 patch
            ↓
        PatchConditioningAdapter
            ↓
        ┌───────────────────────┐
        │ patch condition       │
        │ learned condition     │
        └───────────────────────┘
            ↓
        AudioLDM2 U-Net
            ↓
        diffusion latent
            ↓
        AudioLDM2 VAE / vocoder
            ↓
        audio

    During training:

        patch + target mel
            ↓
        target latent
            ↓
        random noise + timestep
            ↓
        conditioned U-Net
            ↓
        predicted noise
            ↓
        MSE(predicted_noise, noise)
    """

    def __init__(
        self,
        audioldm,
        number_of_params,
        stream_1_dim,
        stream_2_dim,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        train_unet=True,
        train_vae=False,
    ):
        super().__init__()

        self.unet = audioldm.unet
        self.vae = audioldm.vae
        self.vocoder = audioldm.vocoder
        self.scheduler = audioldm.scheduler

        self.conditioner = PatchConditioningAdapter(
            number_of_params=number_of_params,
            stream_1_dim=stream_1_dim,
            stream_2_dim=stream_2_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Freeze / train configuration

        for parameter in self.vae.parameters():
            parameter.requires_grad = train_vae

        for parameter in self.unet.parameters():
            parameter.requires_grad = train_unet

        # Vocoder isn't trained
        for parameter in self.vocoder.parameters():
            parameter.requires_grad = False

        # AudioLDM2 VAE normally has a scaling factor associated
        # with its latent representation.
        self.vae_scaling_factor = getattr(
            self.vae.config,
            "scaling_factor",
            1.0,
        )

    # CONDITIONING

    def condition(
        self,
        params
    ):
        """
        Convert a DX7 patch into AudioLDM2's two conditioning streams.
        """

        return self.conditioner(
            params
        )

    # =============================================================
    # AUDIO ENCODING
    # =============================================================

    def encode_mel(self, mel):
        """
        Encode a preprocessed AudioLDM2 mel/spectrogram into its latent.

        Parameters
        ----------
        mel:
            Tensor containing the spectrogram representation expected by
            the AudioLDM2 VAE.

        Returns
        -------
        latents:
            Scaled latent representation used by the diffusion model.
        """

        posterior = self.vae.encode(mel).latent_dist

        latents = posterior.sample()

        latents = latents * self.vae_scaling_factor

        return latents

    # =============================================================
    # DIFFUSION
    # =============================================================

    def predict_noise(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        encoder_hidden_states_1,
    ):
        """
        Run the AudioLDM2 U-Net.
        """

        output = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_1=encoder_hidden_states_1,
            return_dict=True,
        )

        return output.sample

    def diffusion_loss(
        self,
        params,
        target_mel,
    ):
        """
        Compute the standard diffusion training objective.

        Conceptually:

            patch
              ↓
            conditioner
              ↓
            AudioLDM2 U-Net
              ↑
            noisy target latent
              ↑
            target mel

        Returns
        -------
        loss:
            Scalar diffusion noise-prediction loss.
        """

        device = target_mel.device
        batch_size = target_mel.shape[0]

        target_latents = self.encode_mel(target_mel)

        # Sample random diffusion timestep
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        # Sample Gaussian noise
        noise = torch.randn_like(target_latents)

        # Add noise according to the diffusion schedule
        noisy_latents = self.scheduler.add_noise(
            target_latents,
            noise,
            timesteps,
        )

        # Generate DX7 conditioning
        (
            patch_condition,
            learned_condition,
        ) = self.condition(
            params=params,
        )

        # ---------------------------------------------------------
        # Predict the noise
        # ---------------------------------------------------------

        noise_pred = self.predict_noise(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=patch_condition,
            encoder_hidden_states_1=learned_condition,
        )

        # ---------------------------------------------------------
        # Standard diffusion objective
        # ---------------------------------------------------------

        loss = F.mse_loss(
            noise_pred,
            noise,
        )

        return loss

    # =============================================================
    # FORWARD
    # =============================================================

    def forward(
        self,
        params,
        target_mel
    ):
        """
        Training interface.

        Example:

            loss = model(
                operators,
                global_params,
                target_mel,
            )
        """

        return self.diffusion_loss(
            params=params,
            target_mel=target_mel,
        )

    # =============================================================
    # GENERATION
    # =============================================================

    @torch.no_grad()
    def generate_latents(
        self,
        operators,
        global_params,
        num_inference_steps=50,
        generator=None,
    ):
        """
        Generate AudioLDM2 latent audio from a DX7 patch.
        """

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        batch_size = global_params.shape[0]

        # ---------------------------------------------------------
        # Conditioning
        # ---------------------------------------------------------

        (
            patch_condition,
            learned_condition,
        ) = self.condition(
            operators=operators,
            global_params=global_params,
        )

        # ---------------------------------------------------------
        # Set inference schedule
        # ---------------------------------------------------------

        self.scheduler.set_timesteps(
            num_inference_steps,
            device=device,
        )

        # ---------------------------------------------------------
        # Determine latent shape
        #
        # AudioLDM2's exact latent dimensions should ultimately be
        # obtained from the pipeline/config rather than hardcoded.
        # ---------------------------------------------------------

        latent_channels = self.unet.config.in_channels

        # This is intentionally based on the U-Net configuration.
        # The spatial dimensions need to match the AudioLDM2 pipeline's
        # spectrogram/latent configuration.
        raise NotImplementedError(
            "The exact AudioLDM2 latent spatial dimensions should be "
            "taken from the loaded AudioLDM2 pipeline before enabling "
            "generation."
        )

    # =============================================================
    # DECODING
    # =============================================================

    @torch.no_grad()
    def decode_latents(self, latents):
        """
        Decode AudioLDM2 latents back into the VAE output space.

        Vocoder conversion to waveform should be added after matching
        the exact AudioLDM2 pipeline preprocessing/postprocessing.
        """

        latents = latents / self.vae_scaling_factor

        mel = self.vae.decode(
            latents,
            return_dict=True,
        ).sample

        return mel