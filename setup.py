# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='razorback',
    version='0.2.2b',
    description='Robust estimation of linear response functions',
    author='Farid Smai',
    author_email='f.smai@brgm.fr',
    url='https://github.com/BRGM/razorback',
    license='GNU GPLv3',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'scipy'],
    entry_points = {
        'console_scripts':
            ['razorback-procats121=razorback.scripts.procats_121:main'],
    }
)
