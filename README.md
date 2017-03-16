# Accurate biophysical simulations with machine learning techniques

Here you'll find all code needed to get started denoising the medical image.
After you have installed the required software see the file `denoise_example.py` for the next steps.


### Included images

* `lena.png` and `lena_noisy.png`, small 256x256 images for testing
* `lena512.png` and `lena_noisy512.png` 512x512, same image as above. You can compare your results from denoising this image with the state-of-the-art algorithm BM3D. See the results [here](http://www.cs.tut.fi/~foi/GCF-BM3D/index.html#ref_results), row for sigma=35
* ...


## Installation

Using the [anaconda](https://docs.continuum.io/anaconda/install) python distribution with python 2.7 is by far the easiest. First click on the link to get to the install instructions. When you get to the download page, download **Python 2.7**. It's the big blue button. 

Once that is done, open a terminal (cmd if you're on windows) and type `python` and then press enter. You should see something that looks like

```
Python 2.7.12 |Anaconda custom (64-bit)| (default, Jun 29 2016, 11:07:13) [MSC v.1500 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://anaconda.org
>>>
```

This is the python console, type `help()` for help or `exit()` to exit.

Now we can start installing the required python packages. 

To run this code the following packages are needed:

* [**numpy**](http://www.numpy.org/): For faster and simpler math
* [**scikit-learn**](http://scikit-learn.org/stable/): General machine learning library
* [**VTK**](http://www.vtk.org/): Visualization Toolkit


All requirements except **VTK** are included with anaconda. To install
**VTK** run the following

```
$ conda install --yes vtk
```


## Python Tutorials

*   If you're new to python you can check out this short introduction, it's short but covers the most important parts. http://www.scipy-lectures.org/language/python_language.html

* A quickstart guide for numpy: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

* Python documentation: https://docs.python.org/2.7/
* Numpy docs: https://docs.scipy.org/doc/numpy/reference/
* Scikit-learn docs: http://scikit-learn.org/stable/documentation.html


### Another useful tip

Save and load your dictionaries as 
```python
np.save('dictionary.npy', dictionary)
dictionary = np.load('dictionary.npy')
```

## Papers

This code and algorithms are based on the work in:

> Rubinstein, Ron, Michael Zibulevsky, and Michael Elad. "Efficient implementation of the K-SVD algorithm using batch orthogonal matching pursuit." Cs Technion 40.8 (2008): 1-15.

> Elad, Michael, and Michal Aharon. "Image denoising via sparse and redundant representations over learned dictionaries." IEEE Transactions on Image processing 15.12 (2006): 3736-3745.
