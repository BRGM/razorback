# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import pathlib


PKG = pathlib.Path(__file__).parent


setup(
    name='razorback',
    version='0.3.0.1',
    description='Robust estimation of linear response functions',
    author='Farid Smai',
    author_email='f.smai@brgm.fr',
    url='https://github.com/BRGM/razorback',
    license='GNU GPLv3',
    long_description=(PKG/"README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'scipy', 'dask'],
)
