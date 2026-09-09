import os

import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F

def dataset_to_numpy(dataset):
    xs = []
    ys = []

    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x.view(-1).numpy())
        ys.append(y)

    return np.stack(xs), np.array(ys)

def plot_loss(train_losses, train_mse_recon_losses, train_ce_recon_losses, train_kl_losses, train_be_recon_losses,
              train_algo_recon_losses, val_losses, val_mse_recon_losses, val_ce_recon_losses, val_be_recon_losses,
              val_algo_recon_loss, val_kl_losses, name):

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 6, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["training", "validation"])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 6, 2)
    plt.plot(train_mse_recon_losses)
    plt.plot(val_mse_recon_losses)
    plt.legend(["training", "validation"])
    plt.title('MSE Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 6, 3)
    plt.plot(train_ce_recon_losses)
    plt.plot(val_ce_recon_losses)
    plt.legend(["training", "validation"])
    plt.title('CE Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 6, 4)
    plt.plot(train_be_recon_losses)
    plt.plot(val_be_recon_losses)
    plt.legend(["training", "validation"])
    plt.title('BE Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 6, 6)
    plt.plot(train_kl_losses)
    plt.plot(val_kl_losses)
    plt.legend(["training", "validation"])
    plt.title('KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 6, 5)
    plt.plot(train_algo_recon_losses)
    plt.plot(val_algo_recon_loss)
    plt.title('Algorithm Reconstruction loss')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')

    plt.tight_layout()
    os.makedirs(f"./models/{name}", exist_ok=True)
    plt.savefig(f"./models/{name}/losses.png")
    plt.show()


def visualize_latent_space(model, dataloader, device, NAME, sample_count=1000):
    model.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for sample in dataloader:
            if len(latents) > sample_count:
                break

            x = sample[0].to(device)
            _, y, _ = model(x)
            latents.append(y.cpu())
            # print(algorithm_label.shape) # DEBUG
            labels.append(torch.argmax(x[:, 176:176 + 32], dim=1).cpu())

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latents)
    plt.figure()
    cmap = plt.cm.get_cmap('nipy_spectral', 32)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap)
    plt.savefig(f"./models/{NAME}/latent_umap.png")
    plt.show()

    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(latents)
    # plt.figure()
    # cmap = plt.cm.get_cmap('nipy_spectral', 32)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap)
    # plt.savefig(f"./models/{NAME}/latent_pca.png")
    # plt.show()

def grouped_cross_entropy(logits, targets, c_lengths):
    losses = []

    start = 0
    for length in c_lengths:
        end = start + length

        group_logits = logits[:, start:end]
        group_targets = targets[:, start:end]

        target_idx = group_targets.argmax(dim=1)

        losses.append(
            F.cross_entropy(
                group_logits,
                target_idx.long(),
                reduction="mean"
            )
        )

        start = end

    return torch.stack(losses).mean()

def vae_total_loss(pred, target, be_mask, ce_mask, mse_mask, alg_mask, mu, logvar, beta, mse_coeff=1, use_kl=True, alg_coeff=1):
    assert not torch.isnan(pred).any()
    c_lengths = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6]

    device = pred.device
    be_mask = be_mask.to(device)
    ce_mask = ce_mask.to(device)
    mse_mask = mse_mask.to(device)
    alg_mask = alg_mask.to(device)

    if be_mask.any():
        bce_loss = F.binary_cross_entropy_with_logits(
            pred[:, be_mask],
            target[:, be_mask],
            reduction="mean"
        )
    else:
        bce_loss = torch.tensor(0.0, device=device)

    if mse_mask.any():
        metric_loss = F.l1_loss(
            pred[:, mse_mask],
            target[:, mse_mask],
            reduction="mean"
        ) * mse_coeff

    else:
        metric_loss = torch.tensor(0.0, device=device)

    if ce_mask.any():
        ce_logits = pred[:, ce_mask]
        ce_targets = target[:, ce_mask]

        ce_loss = grouped_cross_entropy(
            ce_logits,
            ce_targets,
            c_lengths
        )
    else:
        ce_loss = torch.tensor(0.0, device=device)

    if alg_mask.any():
        ce_logits = pred[:, alg_mask]
        alg_targets = target[:, alg_mask]

        alg_target_idx = alg_targets.argmax(dim=1)

        alg_loss = F.cross_entropy(
            ce_logits,
            alg_target_idx.long(),
            reduction="mean"
        )
    else:
        alg_loss = torch.tensor(0.0, device=device)

    total_mse_vals = []

    if mse_mask.any():
        total_mse_vals.append(pred[:, mse_mask] - target[:, mse_mask])

    if be_mask.any():
        total_mse_vals.append(torch.sigmoid(pred[:, be_mask]) - target[:, be_mask])

    if ce_mask.any():
        ce_logits = pred[:, ce_mask]
        ce_targets = target[:, ce_mask]

        start = 0
        for length in c_lengths:
            end = start + length

            probs = F.softmax(ce_logits[:, start:end], dim=-1)

            total_mse_vals.append(probs - ce_targets[:, start:end])

            start = end

    def mask_slices(mask):
        mask = mask.cpu().numpy().astype(int)
        slices = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                slices.append((start, i))
                start = None
        if start is not None:
            slices.append((start, len(mask)))
        return slices

    if alg_mask.any():
        for start, end in mask_slices(alg_mask):
            probs = F.softmax(pred[:, start:end], dim=-1)
            total_mse_vals.append(probs - target[:, start:end])

    if total_mse_vals:
        total_mse_tensor = torch.cat(total_mse_vals, dim=1)
        total_mse = torch.abs(total_mse_tensor).mean()
        # total_mse = torch.mean(total_mse_tensor**2)
    else:
        total_mse = torch.tensor(0.0, device=pred.device)

    # # KL divergence loss
    if use_kl:
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        # free bits (forces KL to stay above some minimum value)
        kl_loss = torch.mean(torch.clamp(kl_loss, min=0.01))
    else:
        kl_loss = torch.tensor(0.0)

    total_loss = metric_loss + bce_loss + ce_loss
    total_loss *= alg_coeff
    total_loss += alg_loss
    total_loss += beta * kl_loss

    return total_loss, metric_loss, ce_loss, bce_loss, beta * kl_loss, total_mse, alg_loss

