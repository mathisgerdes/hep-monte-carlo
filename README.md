# Monte Carlo Methods for High Energy Physics

This repository contains the `hepmc` package (high energy physics - Monte Carlo)
providing several Monte Carlo methods of interest to applications in 
high energy physics. 

The implementations of the sampling and integration algorithms are
located in `src/hepmc/`. For an introduction to the Monte Carlo
methods and details on the implementations provided here see the
various Jupyter notebooks located under `notebooks/`.

Besides the main package and notebooks containing introductions and examples,
the package also contains a `make_sample.py` script that can be used to
generate and analyze samples using a given sampler and parameters.
To this end, the package should be installed as detailed below such that
new interfaces can be added to `src/hepmc/interfaces/` and used by the script.

The `examples/` folder contains examples to the major samplers.

# Installation and usage

Phase space sampling uses Sherpa and requires a working installation.
The path to the Sherpa python interface has to be included in `PYTHONPATH`.
If Sherpa is not installed, dependent modules will not be imported.

## Run via virtualenv (venv)
To install the package from source, including required dependencies, 
run the following:
- Create a virtual environment: `python3 -m venv hepmc_env`
- Switch to it: `. hepmc_env/bin/activate`
- Install in *development* (editable) mode: `pip install -e src`

## Run the Jupyter notebooks
The Jupyter notebooks in `notebooks` depend on the package. The following
installs an ipykernel corresponding to the virtualenv (containing hepmc).
Run the notebook server from outside the virtual environment.
- Within the virtualenv active, run `pip install ipykernel`
- Install a kernel: `ipython kernel install --user --name=hepmc`

## Automated sampling script
The `make_sample.py` script is included in the package and can
be used to run samples for a range of parameters specified in a
json configuration file (see `samples/` for examples).
With the virtual environment active, simply run `make_sample.py <config.json>`
in the console.
