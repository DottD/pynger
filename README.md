# Pynger

This package was developed during my Ph.D. as a mean to share frequently-used code snippets among larger projects and easily make tests.

It contains:

* the algorithms developed in _Pierluigi Maponi, Riccardo Piergallini, and Filippo Santarelli. “Fingerprint Orientation Refinement Through Iterative Smoothing”. In: Signal & Image Processing : An International Journal 8.5 (Oct. 2017), pp. 29-43_.
* some algorithms of the NBIS library ported to Python
* utilities for parameters selection through CMA-ES
* a partial implementation of the SFinGe generator
* fingerprint visualization utilities
* estimators to make part of the package Scikit-Learn compliant
* utilities for complex field manipulation, calculus and visualization
* functions for image and mask manipulation
* some procedures for signal processing, among which an implementation of the Fast anisotropic Gauss filter introduced by _Geusebroek, J-M., Arnold WM Smeulders, and Joost Van De Weijer. "Fast anisotropic gauss filtering." IEEE Transactions on Image Processing 12.8 (2003): 938-943._

> __Note__: Regretfully, I am not maintaining this package anymore, since I no longer work with fingerprints.

## Installation

Installing it the usual way, through `pip`, is not possible because there are functions written in C that need to be compiled and linked. If you are not interested in these C functions, you can skip most of the following steps and slightly modify the setup script to disable their compilation.

The package should be installed using `make`, but please ensure to meet all the requirements. You can read [`Makefile`](Makefile) and [`setup.py`](setup.py) for a deeper understanding of what is under the hood, however the installation steps should be as follows.

* Create a folder where all the required libraries will be installed as static libraries, let's call it `LIBDIR`. Even though this is not convenient for reusing such libraries in other projects, this reflects the folder structure in my system at that time and I not going to change it soon.
* Install the [NBIS library](https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis) as separate static libraries and not as a single `.a` file in a folder called `nbis-install` under `LIBDIR`. You can follow the guidelines in the official repository.
* Install [OpenBLAS](https://www.openblas.net) as static libraries in a folder called `openblas-install` under `LIBDIR`.
* Install [Lapack](http://www.netlib.org/lapack/) as static libraries in a folder called `lp-install` under `LIBDIR`.
* Install [Armadillo](http://arma.sourceforge.net) as static libraries in a folder called `armadillo-install` under `LIBDIR`.
* Install [OpenCV](https://opencv.org) as static libraries in a folder called `cv-install` under `LIBDIR`.
* Run `LIBDIR=/path/to/LIBDIR && make install`

Other targets are available to `make`:

* `dev` to install in development mode
* `clean` to clean the project directory from compilation and installation leftover
* `uninstall`
* `rebuild`, that is a short for `clean uninstall install`
* `doc` to produce Sphinx documentation from the scripts. Although far from being comprehensive, it can be a good help to start with the package.
* `undoc` to remove the documentation folder.

## Tests

Due to the fast development required during the Ph.D. I was not able to generate a suite of tests. Actually, most of the functions need to be thoroughly checked and any good hearted contributor will be welcomed.

## Contributing

Any contribution will be very much appreciated. Feel free to contact me in case of any doubts about contributing.

## Authors

* [__Filippo Santarelli__](https://github.com/DottD) - Initial work

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
