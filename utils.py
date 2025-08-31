import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path


def l2_normalize(x, dim=1):
    return x / np.sqrt((x ** 2).sum(dim, keepdims=True))


BASE_CENTROID_PATH = Path("/mnt/A3000/ML/Personalized/shlomi.fenster/PixelsBioID/centroids")


def centroids_init(num_centroids, embedding_dim, save_dir=BASE_CENTROID_PATH):
    save_path = save_dir / f"{num_centroids}_{embedding_dim}.pt"
    if save_path.exists():
        return torch.load(save_path)

    num_combinations = (num_centroids - 1) * num_centroids / 2
    triu_mask = torch.triu(torch.ones(num_centroids, num_centroids), 1)
    triu_mask = triu_mask.to("cuda") if torch.cuda.is_available() else triu_mask

    def centroids_repulsion_loss(centroids):
        centroid_dists = F.linear(centroids, centroids)
        loss = torch.sum(centroid_dists * triu_mask) / num_combinations
        return loss

    centroids = torch.tensor(l2_normalize(np.random.rand(num_centroids, embedding_dim)))
    centroids = centroids.to("cuda") if torch.cuda.is_available() else centroids

    centroids = torch.nn.Parameter(centroids)

    opt = torch.optim.SGD([centroids], 1e-1)
    for step in range(1000):
        loss = centroids_repulsion_loss(F.normalize(centroids))
        opt.zero_grad()
        loss.backward()
        opt.step()

    opt = torch.optim.SGD([centroids], 1e-2)
    for step in range(1000):
        loss = centroids_repulsion_loss(F.normalize(centroids))
        opt.zero_grad()
        loss.backward()
        opt.step()

    opt = torch.optim.SGD([centroids], 1e-3)
    for step in range(10000):
        if not step % 1000:
            print(step, loss.item())
        loss = centroids_repulsion_loss(F.normalize(centroids))
        opt.zero_grad()
        loss.backward()
        opt.step()

    opt = torch.optim.SGD([centroids], 1e-4)
    for step in range(10000):
        if not step % 1000:
            print(step, loss.item())
        loss = centroids_repulsion_loss(F.normalize(centroids))
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.save(centroids.data.cpu().detach(), save_path)

    return centroids.data
