from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name = 'temp_map',
    version = '0.1.0',
    author = 'Zachary Stone',
    author_email = 'stone28@illinois.edu',
    packages = ['temp_map'],
    license = 'MIT',
    description = 'A package to calculate the temerature perturbation map of an AGN, following Stone et al. 2022.',
    long_description = open('README.md').read(),
    install_requires = install_requires
)