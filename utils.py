"""
    This file contains various utility functions
"""

from __future__ import print_function
import numpy as np


def psnr(original, noisy, max_pixel_value):
    """
        Peak Signal to Noise ratio in decibel

        Args
        ----
            original: Original clean image
            noisy: Image to compare against the original
            max_pixel_value: Maximum possible pixel value in image

        Returns
        -------
            number, peak signal to noise ratio
    """
    mse = ((original - noisy)**2).mean()
    return 20*np.log10(max_pixel_value) - 10*np.log10(mse)


def center(patches):
    """
        Remove the mean value from every patch

        Args
        ----
            patches: Image patches to center

        Returns
        -------
            A tuple, (patches, mean)
    """
    mean = patches.mean(axis=0)
    return patches - mean, mean


def normalize(image):
    """
        Normalize an image

        Maps first to [0, image.max() - image.min()]
        then to [0, 1]
    """
    image = image.astype(float)
    image -= image.min()
    return image / max(image.max(), 1)


def initial_dictionary(rows, n_atoms=None):
    """
       Create an random initial dictionary

        Args
        ----
            rows: Number of rows in dictionary. Has to be equal to the
            number of rows in the image patches matrix
            n_atoms: Number of atoms in dictionary, default 2*rows

        Returns
        -------
            Random dictionary of shape (rows, n_atoms
    """

    if n_atoms is None:
        n_atoms = 2*rows

    dictionary = np.random.rand(rows, n_atoms)

    # Normalize the columns to have unit l2 norm
    for col in range(n_atoms):
        atom = dictionary[:, col]
        dictionary[:, col] = atom / np.linalg.norm(atom)

    return dictionary



def visualize_dictionary(dictionary, rows, cols):
    """
        Visualize a dictionary. This is somewhat slow

        Args
        ----
            dictionary: Dictionary to plot
            rows: Number of rows in plot
            cols: Number of columns in plot. Rows*cols has to be equal
            to the number of atoms in dictionary

    """
    import matplotlib.pyplot as plt

    size = int(np.sqrt(dictionary.shape[0])) + 2

    for row in range(rows):
        for col in range(cols):
            idx = row*cols + col
            atom = dictionary[:, idx].reshape(size - 2, size - 2)
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(atom)
            plt.axis('off')

    plt.show()

def numpy_from_vti(path):
    """
        Convert a vti image volume into a 3d numpy array

        Args
        ----
            path: Path to .vti file

        Returns
        -------
            3d numpy.ndarray
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    image_data = reader.GetOutput()
    dimensions = image_data.GetDimensions()
    return vtk_to_numpy(image_data.GetPointData().GetArray(0)).reshape(dimensions)


def numpy_to_vti(image, path):
    """
        Convert a numpy array into a .vti file.

        Args
        ----
            image: 3D ndarray with image data
            path: Where to save the image

        Returns
        -------
            True if writing successful
    """
    import vtk
    assert image.ndim == 3
    dim_x, dim_y, dim_z = image.shape
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dim_x, dim_y, dim_z)
    image_data.AllocateScalars(vtk.VTK_DOUBLE, 1)

    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                data = image[i, j, k]
                image_data.SetScalarComponentFromDouble(i, j, k, 0, data)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(image_data)
    return bool(writer.Write())
