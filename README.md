# Introduction to Monte Carlo Integration

The implementations of the sampling and integration algorithms are
located in `monte_carlo/`. For an introduction to the Monte Carlo
methods and details on the implementations provided here see the
`monte_carlo.ipynb` jupyter notebook.

The notebook `plots.ipynb` contains a few dense plots highlighting
specific behaviors of Monte Carlo and Markov Chain techniques.
`hamilton_mc.ipynb` introduces the Hamilton Monte Carlo sampling method,
`mc3.ipynb` focuses on the MC<sup>3</sup> implementations. 

## Covered Algorithms

### Monte Carlo Integration
Implemented in `monte_carlo/integration.py`.
- Plain (ordinary) Monte Carlo
- Stratified Monte Carlo
- Importance Sampling Monte Carlo
- VEGAS Monte Carlo
- Importance Multi-Channel Monte Carlo
- Markov Chain Monte Carlo (only in `monte_carlo.ipynb`)

### Sampling Methods
- Acceptance Rejection (`monte_carlo/sampling.py`)
- Metropolis / Metropolis Hasting (`monte_carlo/markov.py`)
- Hamilton Monte Carlo (`monte_carlo/hmc.py`)

### Combined Method
-  Multi-Channel Markov Chain Monte Carlo (`monte_carlo/mc3.py`)

## Implementation
The implementation is done in pure python with `numpy`.
To speed up the methods `numpy` was used as much as possible, however
the implementation should still be considered educational rather than
efficient.
