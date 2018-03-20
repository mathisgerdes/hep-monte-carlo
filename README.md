# Introduction to Monte Carlo Integration

The implementations of the sampling and integration algorithms are
located in `monte_carlo/`. For an introduction to the Monte Carlo
methods and details on the implementations provided here see the
`monte_carlo.ipynb` jupyter notebook.

Some examples on Markov Chains and related statistics can be found in
`markov_chains.ipynb`.

## Covered Algorithms

### Monte Carlo Integration
- Plain (ordinary) Monte Carlo
- Stratified Monte Carlo
- Importance Sampling Monte Carlo
- VEGAS Monte Carlo
- Importance Multi-Channel Monte Carlo
- Markov Chain Monte Carlo (in `markov_chains.ipynb`)

### Sampling Methods
- Acceptance Rejection
- Metropolis / Metropolis Hasting

### Combined Method
-  Multi-Channel Markov Chain Monte Carlo

## Implementation
The implementation is done in pure python with `numpy`.
To speed up the methods `numpy` was used as much as possible, however
the implementation should still be considered educational rather than
efficient.
