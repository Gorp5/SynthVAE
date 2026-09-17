import os
import numpy as np
import torch
import tqdm

from diffusers import AutoencoderKL
from vae.dataset import SpectrogramStreamingDataset


SPEC_PATH = r"E:\SynthesizerDataset\single_note_dataset\compressed_specs"
PATCH_PATH = r"E:\SynthesizerDataset\patches"
LATENT_PATH = r"D:\SynthesizerDataset\ldm2_latents"

BATCH_SIZE = 32       # Increase until VRAM becomes limiting
NUM_WORKERS = 4       # Increase/decrease based on CPU/storage performance


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reasonable performance.")

    device = torch.device("cuda")

    print("Using GPU:", torch.cuda.get_device_name())

    # Useful when all spectrograms have identical dimensions
    torch.backends.cudnn.benchmark = True

    print("Loading AudioLDM2 VAE...")

    vae = AutoencoderKL.from_pretrained(
        "cvssp/audioldm2-music",
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    vae = vae.to(device)
    vae.eval()

    scaling_factor = vae.config.scaling_factor

    print("VAE scaling factor:", scaling_factor)

    dataset = SpectrogramStreamingDataset(
        SPEC_PATH,
        PATCH_PATH,
    )

    print("Total samples:", len(dataset))

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
    }

    # persistent_workers requires workers > 0
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dataloader = torch.utils.data.DataLoader(
        dataset,
        **loader_kwargs,
    )

    os.makedirs(LATENT_PATH, exist_ok=True)

    with torch.inference_mode():

        for specs, patches, idxs in tqdm.tqdm(
            dataloader,
            desc="Encoding",
        ):


            specs = specs.unsqueeze(1).to(
                device=device,
                dtype=torch.float16,
                non_blocking=True,
            )


            latents = vae.encode(specs).latent_dist.mode()

            latents.mul_(scaling_factor)
            latents = latents.cpu().numpy()

            # idxs should be something like [B, ...]
            idxs = idxs.cpu().numpy().reshape(-1)


            for idx, latent in zip(idxs, latents):

                idx = int(idx)
                path = os.path.join(
                        LATENT_PATH,
                        f"spec_{idx:07d}.npy",
                    )

                if os.path.exists(path):
                    continue

                np.save(
                    path,
                    latent,
                )


if __name__ == "__main__":
    main()