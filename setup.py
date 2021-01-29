# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import pathlib
import codecs
import os.path


PKG = pathlib.Path(__file__).parent


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name='razorback',
    version=get_version('razorback/__init__.py'),
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
