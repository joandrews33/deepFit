import torch
import torch.distributions as dist
import numpy as np


def get_camera_image(n_px, camera_gain = torch.tensor([380.0]), device='cpu'):

    # Gamma distributions do a reasonable job of approximating the noise of a EMCCD camera.
    # The precise values of the scale and shape parameters were fit for a particular video acquired of Dendra2 photoactivations.
    # The scale and shape parameters can be modeled as the mean number of background photons per pixel (shape),
    # and the camera gain (scale).

    gamma_scale = camera_gain
    gamma_shape = torch.tensor([11.0])

    noise_model = dist.gamma.Gamma(gamma_shape, 1/gamma_scale)
    X = noise_model.sample(sample_shape=torch.Size([n_px, n_px])).view([1, 1, n_px, n_px])

    return X

# def add_molecules(n, X):
#
#     # Hardcoded parameters are for imaging on a 16 um / pixel EMCCD with 100x magnification,
#     # with a photon wavelength approx.  consistent with mCherry emission.
#     sigma = torch.tensor([80.]) #nm The positional uncertainty of a photon due to diffraction. 0.42 lambda / 2 / NA with lambda about 600nm and NA 1.4
#     sigma_px = sigma/160. #Assume pixel size of 160nm
#     mean_num_photons = 64 #Depends on illumination intensity and camera integration time. This value gives a similar SNR to typical experimental data.
#
#     r, c = X.shape()
#
#     for i in range(n):
#         x_c = np.random.rand()*r
#         y_c = np.random.rand()*c
#
#         num_emitted_photons = dist.poisson.Poisson(mean_num_photons).sample()
#         position_dist = dist.normal.Normal(torch.tensor([x_c, y_c]), covariance_matrix=torch.eye(2)*sigma**2)
#         positions = position_dist.sample(torch.size(num_emitted_photons))


def add_molecule(im, molecule_list=torch.tensor([]), sigma=80, mean_num_photons=64, camera_gain=torch.tensor(380.0), trim_px = 0):
    # sigma = torch.tensor([80.]) #nm The positional uncertainty of a photon due to diffraction. 0.42 lambda / 2 / NA with lambda about 600nm and NA 1.4
    sigma_px = sigma / 160.  # Assume pixel size of 160nm
    # mean_num_photons = 64 #Depends on illumination intensity and camera integration time. This value gives a similar SNR to typical experimental data.

    r, c = im.squeeze().size()

    x_c = torch.rand(1) * (r-2*trim_px)+trim_px
    y_c = torch.rand(1) * (c-2*trim_px)+trim_px

    if not molecule_list.numel():  # is empty
        molecule_list = torch.cat([molecule_list, torch.tensor([x_c, y_c])], dim=0)
    else:  # is NOT empty
        molecule_list = torch.cat([molecule_list.view(-1, 2), torch.tensor([[x_c, y_c]])], dim=0)

    num_emitted_photons = dist.poisson.Poisson(mean_num_photons).sample()

    position_dist = dist.multivariate_normal.MultivariateNormal(torch.tensor([x_c, y_c]),
                                                                covariance_matrix=torch.eye(2) * sigma_px ** 2)

    positions = position_dist.sample((num_emitted_photons.int(),))

    out_of_bounds = torch.logical_or( \
        torch.logical_or(positions[:, 0] < 0, positions[:, 0] >= r), \
        torch.logical_or(positions[:, 1] < 0, positions[:,
                                              1] >= c))  # Simulated data is not restricted to appearing on the camera. Out of bounds data must be removed to prevent indexing errors.


    positions = positions[torch.logical_not(out_of_bounds), :]
    indices = torch.cat((torch.zeros(num_emitted_photons.int()-sum(out_of_bounds), 2, dtype=torch.long), torch.floor(positions).long()), 1)
    #indices = torch.cat((torch.zeros(positions.numel()/2, 2, dtype=torch.long), torch.floor(positions).long()), 1)

    for idx in indices:
        im[idx[0], idx[1], idx[2], idx[3]] += camera_gain

    return im, molecule_list
