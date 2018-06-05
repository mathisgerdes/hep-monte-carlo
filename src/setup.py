from setuptools import setup, find_packages
setup(
    name='hepmc',
    version='0.1',
    packages=find_packages('.', exclude=['tests']),
    scripts=['make_sample.py'],
    install_requires=[
        'Numpy>=1.14.0',
        'matplotlib>=2.1.2',
        'scipy>=1.0',
    ],
    include_package_data=True,
    package_data={'hepmc': "data/*.dat"},  # Sherpa run card
)
