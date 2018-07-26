# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


# with open('LICENSE') as f:
#     license = f.read()
license = None

setup(
    name='razorback',
    version='0.2.1',
    description='Robust estimation of linear response functions',
    author='Farid Smai',
    author_email='f.smai@brgm.fr',
    url='https://forge.brgm.fr/projects/emprocessing/repository',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points = {
        'console_scripts':
            ['razorback-procats121=razorback.scripts.procats_121:main'],
    }
)
