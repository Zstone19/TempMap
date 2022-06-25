from setuptools import setup    

exec(open('temp_prof/version.py').read())

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='TempProf',
    version=__version__,
    author='Zachary Stone',
    author_email='stone28@illinois.edu',
    description='A package for analyzing temperature fluctuation maps for AGN accretion disks',
    package_dir={'temp_prof': 'temp_prof'},
    packages=['temp_prof'],
    license='MIT',
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
