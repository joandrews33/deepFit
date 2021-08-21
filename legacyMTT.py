import numpy as np
import torch


# I think that the fft is not centered right, because I am getting hits in weird places.
#  Also, I have not tested if alpha and wmap are coded correctly

def wilks_map(Im, radius, n_pix):
    R, C = np.shape(Im)
    N = n_pix**2
    mask = np.ones((n_pix, n_pix))

    def pad(Mask,R,C):
        r,  c = np.shape(Mask)
        Out = np.zeros((R, C))
        r_indices = slice(np.floor_divide((R-r), 2), np.floor_divide((R-r), 2)+r)
        c_indices = slice(np.floor_divide((C-c), 2), np.floor_divide((C-c), 2)+c)
        Out[r_indices, c_indices] += Mask
        return Out

    def gauss_mask(r,n_pix):
        xx = np.arange(n_pix).reshape(-1,1)@np.ones((1,n_pix))-(n_pix-1)/2
        yy = np.ones((n_pix,1))@np.arange(n_pix).reshape(1,-1)-(n_pix-1)/2
        zz = -(xx**2+yy**2)/2/r**2
        mask = 1/np.sqrt(np.pi*r**2)*np.exp(zz)
        return mask

    mask = pad(mask, R, C)


# Null Hypothesis

    trans_mask = np.fft.fft2(mask)
    trans_image = np.fft.fft2(Im)
    m0 = np.real(np.fft.fftshift(np.fft.ifft2(trans_mask*trans_image)))/N  #Mean intensity in window

    trans_image2 = np.fft.fft2(Im*Im)
    im_variance = np.real(np.fft.fftshift(np.fft.ifft2(trans_mask*trans_image2)))/N #Mean second moment of initensity in window

# H1

    g_mask = gauss_mask(radius, n_pix)
    g_mask_mean_shift = g_mask - np.sum(g_mask)/N
    g_mask_mean_shift = pad(g_mask_mean_shift, R, C)
    g_variance = np.sum(g_mask_mean_shift**2)

    trans_g_mask = np.fft.fft2(g_mask_mean_shift)

    alpha = np.real(np.fft.fftshift(np.fft.ifft2(trans_g_mask*trans_image))) / g_variance  #fit value of gaussian term prefactor.



    test = 1 - (g_variance * alpha**2)/ (im_variance - N*m0**2 )
    test[test <= 0] = 1
    w_map = - N * np.log(test)

    return m0, alpha, w_map


def wilks_map_tensor(Im, radius, n_pix):
    R, C = Im.size()

    N = n_pix ** 2

    def pad(Mask, R, C):
        r, c = Mask.size()
        Out = torch.zeros((R, C))
        r_indices = slice(torch.floor_divide((R - r), 2), torch.floor_divide((R - r), 2) + r)
        c_indices = slice(torch.floor_divide((C - c), 2), torch.floor_divide((C - c), 2) + c)
        Out[r_indices, c_indices] += Mask
        return Out

    def gauss_mask(r, n_pix):
        xx = torch.arange(n_pix, dtype=torch.float).reshape(-1, 1) @ torch.ones((1, n_pix)) - (n_pix - 1) / 2
        yy = torch.ones((n_pix, 1)) @ torch.arange(n_pix, dtype=torch.float).reshape(1, -1) - (n_pix - 1) / 2
        zz = -(xx ** 2 + yy ** 2) / 2 / r ** 2
        mask = 1 / np.sqrt(np.pi * r ** 2) * torch.exp(zz)
        return mask

    # Null Hypothesis
    # Use fourier transforms to convolve image with specific kernel masks quickly. A mask of ones will sum elements in the window.

    addition_mask = torch.ones(
        (n_pix, n_pix))  # This mask of ones will sum all pixels within the window when convolved with image
    addition_mask = pad(addition_mask, R, C)

    trans_add_mask = torch.fft.fft2(addition_mask)  # Fourier Transformed mask
    trans_image = torch.fft.fft2(Im)
    m0 = torch.real(torch.fft.fftshift(torch.fft.ifft2(trans_add_mask * trans_image))) / N  # Mean intensity in window

    trans_image2 = torch.fft.fft2(Im * Im)
    im_variance = torch.real(torch.fft.fftshift(
        torch.fft.ifft2(trans_add_mask * trans_image2))) / N  # Mean second moment of intensity in window

    # H1
    # Intensity Profile ~ Alpha* Gaussian Profile + Background

    g_mask = gauss_mask(radius, n_pix)
    g_mask_mean_shift = g_mask - torch.sum(g_mask) / N
    g_mask_mean_shift = pad(g_mask_mean_shift, R, C)
    g_variance = torch.sum(g_mask_mean_shift ** 2)

    trans_g_mask = torch.fft.fft2(g_mask_mean_shift)

    alpha = torch.real(torch.fft.fftshift(
        torch.fft.ifft2(trans_g_mask * trans_image))) / g_variance  # fit value of gaussian term prefactor.

    # Evaluating the test statistic -log of the likelihood ratios under the two models
    test = 1 - (g_variance * alpha ** 2) / (im_variance - N * m0 ** 2)
    test[test <= 0] = 1
    w_map = - N * torch.log(test)

    return m0, alpha, w_map