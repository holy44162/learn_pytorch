import numpy as np
import torch
from sklearn.covariance import LedoitWolf

class GaussianDensityTorch(object):
    """Gaussian Density estimation similar to the implementation used by Ripple et al.
    The code of Ripple et al. can be found here: https://github.com/ORippler/gaussian-ad-mvtec.
    """
    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        self.inv_cov = torch.Tensor(LedoitWolf().fit(embeddings.cpu()).precision_,device="cpu")

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()

if __name__ == "__main__":
    # write inv_cov
    s = ''
    data_inv_cov = np.load("d:/backup/project/learn_pytorch/test_cutpaste/inv_cov.npy")
    r, c = data_inv_cov.shape
    data_flat = data_inv_cov.reshape(r*c)
    for v in data_flat:
        s += str(v) + '\t'
    path = "d:/backup/project/learn_pytorch/test_cutpaste/data_inv_cov.txt"
    f = open(path, "w+")
    f.write(s)
    f.close()

    # write mean
    s = ''
    data_mean = np.load("d:/backup/project/learn_pytorch/test_cutpaste/mean.npy")
    for v in data_mean:
        s += str(v) + '\t'
    path = "d:/backup/project/learn_pytorch/test_cutpaste/data_mean.txt"
    f = open(path, "w+")
    f.write(s)
    f.close()

    # write best threshold
    s = ''
    data_threshold = np.load("d:/backup/project/learn_pytorch/test_cutpaste/best_threshold.npy")
    path = "d:/backup/project/learn_pytorch/test_cutpaste/data_threshold.txt"
    f = open(path, "w+")
    f.write(str(data_threshold))
    f.close()

    sample = 10.*np.ones((1,1024))
    # sample = sample.astype(np.double)
    sample[0][6] = 7.
    sample[0][7] = 6.
    print('sample[0][6]: ', sample[0][6])
    print('sample[0][16]: ', sample[0][16])

    sample_tensor = torch.from_numpy(sample)
    data_mean_tensor = torch.from_numpy(data_mean)
    data_inv_cov_tensor = torch.from_numpy(data_inv_cov)

    density = GaussianDensityTorch()
    density.mean = data_mean_tensor.double()
    density.inv_cov = data_inv_cov_tensor.double()
    distance = density.predict(sample_tensor.double())
    print('distance: ', distance)