# Solving a system of linear equations

This is a collection of utilities to solve systems of linear equations. They are written in Python and use numpy and matplotlib.
Numerous examples are provided.

More details about the algorithm can be found in the Youtube playlist on my channel [Solving System of Linear Equations using Python](https://www.youtube.com/playlist?list=PLWv23ocV_hYYORMAgPh5Ys4dhzZXbfyrw). If you find it useful, please consider subscribing to my channel. 

The following algorithms are available in `utils.py`:
- `cholesky_decomposition`
- `crout_decomposition`
- `doolittle_decomposition`
- `gauss_elimination`
- `gauss_jordan_elimination`- 
- `gauss_seidel`
- `conjugate_gradient`
- `qr_decomposition`

Along with the following helper functions:
- `is_symmetric`
- `is_positive_definite`
- `forward_substitution_unitdiag`
- `forward_substitution_nonunitdiag`
- `backward_substitution_unitdiag`
- `backward_substitution_nonunitdiag`
- `plot_slope_intercept_line`
- `check_pivots`
- `forward_elimination`
- `backward_elimination`

## License

MIT (see the LICENSE file for more details)