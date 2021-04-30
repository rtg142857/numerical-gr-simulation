# numerical-gr-simulation
This repository contains files used in A Model for Spherically Symmetric Matter Collapse, Black Holes and Wormholes, by Alexander Menegas (2021). The following is a brief overview of the programs:

bhfunctions.pyx is a list of functions written in Cython that are used by the other programs. It is converted into bhfunctions.c, and compiled/assembled into bhfunctions.cpython-37m-darwin.so, via setup.py.

blackholesimulator.py and bhcritical.py are both models of spherically symmetric general relativity for a massless scalar field in R x R^3. The former is for visualisation and testing purposes, whereas the latter finds the value of a parameter in the initial configuration of matter that leads to critical collapse.

bhevolutionconvergence.py is a means of testing the output of blackholesimulator.py in order to find the rate of convergence of the simulation (as well as whether it outputs physical results).

fitting.py is a program that fits a multi-parameter function to data and finds the parameters of best fit.
