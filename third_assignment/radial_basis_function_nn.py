from numpy import vstack, true_divide, mean, loadtxt
from random import sample
from h5py import File

class Rbf:
    def __init__(self, prefix = 'rbf', workers = 4, extra_neurons = 0, from_files = None):
        self.prefix = prefix
        self.workers = workers
        self.extra_neurons = extra_neurons

        # Import partial model
        if from_files is not None:            
            w_handle = self.w_handle = File(from_files['w'], 'r')
            mu_handle = self.mu_handle = File(from_files['mu'], 'r')
            sigma_handle = self.sigma_handle = File(from_files['sigma'], 'r')
            
            self.w = w_handle['w']
            self.mu = mu_handle['mu']
            self.sigmas = sigma_handle['sigmas']
            
            self.neurons = self.sigmas.shape[0]

    def _calculate_error(self, y):
        self.error = mean(abs(self.os - y))
        self.relative_error = true_divide(self.error, mean(y))

    def _generate_mu(self, x):
        n = self.n
        extra_neurons = self.extra_neurons

        # TODO: Make reusable
        mu_clusters = loadtxt('clusters100.txt', delimiter='\t')

        mu_indices = sample(range(n), extra_neurons)
        mu_new = x[mu_indices, :]
        mu = vstack((mu_clusters, mu_new))

        return mu

