# razorback

An open source python library for magnetotelluric robust processing.

It provides robust (M-estimator, Bounded Influence) and multi-remote methods for MT.

## How to cite

If you use this software in a scientific publication, we'd very much appreciate if you could cite the following paper:

- Smaï, F., and Wawrzyniak, P., 2020.
Razorback, an open source Python library for robust processing of magnetotelluric data.
Frontiers in Earth Science.
https://doi.org/10.3389/feart.2020.00296

## Installing

Just use `pip` to get razorback and all its dependencies:
```
pip install razorback
```

## Tutorials

Get the tutorials and their attached data by cloning the repository with its submodules:
```
git clone --recursive https://github.com/brgm/razorback.git
```

You will find the tutorials in [./docs/source/tutorials/](docs/source/tutorials/)


## Dependencies

- Python >= 3.6
- Numpy
- Scipy
- Dask

## Supported Platforms
Linux, Windows, MacOS

## Running tests

To run the tests, clone the repository and run pytest in the main directory:
```
pytest
```

## Documentation
<https://razorback.readthedocs.io>
