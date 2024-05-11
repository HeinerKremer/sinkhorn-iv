import torch
import numpy as np
from scipy.spatial.distance import cdist


def get_all_data(dataset):
    print('Collecting all data! Dont use with large datasets!')
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)))


def get_rbf_kernel(x_1, x_2=None, sigma=None, numpy=False):
    if x_2 is None:
        x_2 = x_1

    sq_dist = calc_sq_dist(x_1, x_2, numpy=False)

    if sigma is None and numpy:
        median = np.median(sq_dist.flatten()) ** 0.5
        sigma = median
    elif sigma is None and not numpy:
        sigma = torch.median(sq_dist) ** 0.5

    kernel_zz = torch.exp((-1 / (2 * sigma ** 2)) * sq_dist)
    if numpy:
        kernel_zz = kernel_zz.detach().numpy()
    return kernel_zz, sigma


def calc_sq_dist(x_1, x_2, numpy=True):
    n_1, n_2 = x_1.shape[0], x_2.shape[0]
    if numpy:
        return cdist(x_1.reshape(n_1, -1), x_2.reshape(n_2, -1),
                        metric="sqeuclidean")
    else:
        if not torch.is_tensor(x_1):
            x_1 = torch.from_numpy(x_1).float()
            x_2 = torch.from_numpy(x_2).float()
        return torch.cdist(torch.reshape(x_1, (n_1, -1)), torch.reshape(x_2, (n_2, -1))) ** 2
