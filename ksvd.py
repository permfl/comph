from __future__ import print_function

import multiprocessing
import numpy as np
import timeit

from utils import center

from sklearn.feature_extraction.image import (
    extract_patches_2d, reconstruct_from_patches_2d
)

from sklearn import linear_model


def ksvd(patches, dictionary, iters, n_nonzero, tol=None, verbose=True):
    """
        Train a dictionary from image patches

        This algorithm alternates between finding a
        sparse approximation with omp_batch and
        updating the dictionary


        This is algorithm 5 in
            Rubinstein, Ron, Michael Zibulevsky, and Michael Elad.
            "Efficient implementation of the K-SVD algorithm using batch orthogonal
            matching pursuit." Cs Technion 40.8 (2008): 1-15.

        Comments with (number) show the corresponding line in
        algorithm number 5 in the paper


        Args
        ----
            patches: Image patches, shape (patch_size, n_patches)
            dictionary: Initial dictionary, shape (patch_size, n_atoms)
            iters: Number of training iterations
            n_nonzero: Max number of nonzero coefficients to use in
                       sparse approximation
            tol: This overwrites n_nonzero. Finds a sparse approx
                 such that the residual is less then this tolerance

        Returns
        -------
            Dictionary with shape (patch_size, n_atoms)
    """
    m, p = dictionary.shape

    for t in range(iters):
        if verbose:
            print('K-SVD iteration %d' % (t + 1))
            print('  Sparse coding...')
            
        decomp = omp_batch(patches, dictionary, n_nonzero, tol)  # (5)

        if verbose:
            print('  Updating dictionary...')

        for k in range(p):
            row_k = decomp[k]  # (8)
            # Index of coeffs using atom number k
            w = np.nonzero(row_k)[0]  # (8)

            if len(w) == 0:
                # This atom is not used
                continue

            dictionary[:, k] = 0  # (7)
            g = decomp[k, w]  # (9)

            decomp_w = decomp[:, w]
            signals_w = patches[:, w]
            dict_d_w = dictionary.dot(decomp_w)
            d = signals_w.dot(g) - dict_d_w.dot(g)  # (10)
            d /= np.linalg.norm(d)  # (11)

            g = signals_w.T.dot(d) - dict_d_w.T.dot(d)  # (12)
            dictionary[:, k] = d  # (13)
            decomp[k, w] = g  # (14)

    return dictionary


def omp_batch(signals, dictionary, n_nonzero=0, tol=0):
    """
        Sparse Decomposition

        Find a sparse approximation using the OMP algorithm. This is an
        implementation of algorithm 3 in the paper reference in the ksvd
        function above

        If tolerance in zero this finds the minimum of:
            min || x - Da||_2^2 such that ||a||_0 <= n_nonzero

        If tolerance is not zero:
            min ||a||_0 such that ||x - Da||_2^2 <= tol

        Args
        ----
            signals: Signals to approximate
            dictionary:
            n_nonzero: Max number of coefficients to use
            tol: Tolerance of approximation, overwrites n_nonzero

        Returns
        -------
            Sparse approximation with shape (n_atoms, n_signals)
    """
    n = signals.shape[1]
    norms_squared = np.zeros(n)
    for k in range(n):
        norms_squared[k] = np.linalg.norm(signals[:, k]) ** 2

    gram = dictionary.T.dot(dictionary)
    Xy = dictionary.T.dot(signals)
    tol = None if tol == 0 else tol

    decomp = linear_model.orthogonal_mp_gram(
        gram, Xy, n_nonzero, tol, norms_squared
    )

    return decomp


def create_patches(image, patch_size, max_patches=None, random=None):
    """
        Create a 2D array of shape
         (patch_size*patch_size, n_patches), with flattened 2D image
         patches along the columns

        Args
        ----
            image: Image as numpy array
            patch_size: Tuple, image patch size.
                        Need patch_size[0] == patch_size[1]
            random: Extract patches form random locations

        Returns
        -------
            Image patches, shape (patch_size**2, n_patches)
    """

    if patch_size[0] != patch_size[1]:
        raise ValueError('Both patch dims has to be equal '
                         '%d != %d' % (patch_size[0], patch_size[1]))

    # extract_patches_2d from sklearn return an array of 2D patches with the
    # shape (n_patches, patch_size[0], patch_size[1])
    patches = extract_patches_2d(image, patch_size, 
                                 max_patches, random_state=random)

    # Reshape such that we get each patch as a column
    return patches.reshape(patches.shape[0], -1).T


def reconstruct_patches(image_patches, image_size):
    """
        Reconstruct an image from image patches.
        Average the values from overlapping pixels

        Args
        ----
            image_patches: 2d ndarray shape (patch_size**2, n_patches)
            image_size: Original image size as a tuple. (height, width)

        Returns
        -------
            Reconstructed image

    """
    size = int(np.sqrt(image_patches.shape[0]))
    p = image_patches.T.reshape((image_patches.shape[1], size, size))
    return reconstruct_from_patches_2d(p, image_size)


def denoise(patches, dictionary, sigma, noise_gain=1.15, verbose=True):
    """
        Denoise the image patches. This is based on
            Elad, Michael, and Michal Aharon. "Image denoising via sparse and 
            redundant representations over learned dictionaries." 
            IEEE Transactions on Image processing 15.12 (2006): 3736-3745.

        Args
        ----
            patches: Noisy image patches, shape (patch_size, n_patches)
            dictioanry: shape (patch_size, n_atoms)
            sigma: variance of noise
        
        Returns
        -------
            Denoised image patches, shape (patch_size, n_patches)

    """
    if verbose:
        print('Denoising...\n')

    tol = patches.shape[0]*(noise_gain*sigma)**2
    new_patches = omp_batch(patches, dictionary, tol=tol)
    return np.dot(dictionary, new_patches)


def denoise_3d(volume, dictionary, patch_size, sigma, 
               noise_gain=1.15, verbose=True):
    """
        Denoise a 3D image volume

        Args
        ----
            volume: 3D numpy array, noisy image
            dictionary: Dictionary already trained on the noisy volume
            patch_size: Size of image patches
            sigma: Variance noise

        Returns
        -------
            Denoised image volume
    """

    return 'Write your code here'
