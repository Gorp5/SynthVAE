import torch
import numpy as np

from diffusers import AudioLDM2Pipeline
from vae.dataset import SpectrogramStreamingDataset

def encode_mel(pipe, mel):
    with torch.no_grad():
        posterior = pipe.vae.encode(mel).latent_dist
        latent = posterior.mode()

    return latent * pipe.vae.config.scaling_factor

if __name__ == "__main__":
    model = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float32,
    )

    dataset = SpectrogramStreamingDataset("E:\\SynthesizerDataset\\single_note_dataset\\compressed_specs\\",
                                          "E:\\SynthesizerDataset\\patches\\")

    latent_dataset_path = "E:\\SynthesizerDataset\\single_note_dataset\\ldm2_latents\\"

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    if has_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    print("Total samples:", len(dataset))

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in train_dataloader:
        spec, patch, idx = batch
        latents = encode_mel(model, spec)

        for latent in latents.to("cpu").detach().numpy():
            np.save(f"{latent_dataset_path} + idx + .npy, latent)
