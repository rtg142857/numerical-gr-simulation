############################################################
# Alternating stepping forward Pi and Phi using RK2 in time
# and calculating a and alpha using RK2 in space
# Initial data is an arbitrary function of phi with Neumann conditions at r=0,
# from which Pi and Phi are calculated
############################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os, sys
import matplotlib
from scipy import sparse
from bhfunctions import *

#-----------------------------------------------------------
# Definition of parameters and associated default values. 
#-----------------------------------------------------------

# r is the spatial coordinate
rmin = 0
rmax = 10 #arbitrary, might change later — system is invariant
# under changes in scale+time

N = 20001 #number of grid points #usually 501 or 1001
rarray = np.linspace(rmin, rmax, num=N) # array of r-coordinates
dr = (rmax-rmin)/(N-1) #grid spacing

c = 1.0 #CFL factor
dt = c/(N-1) #timestep
nsteps = 200000 #number of timesteps #1200 for normal stuff takes ~1min #120 for checking
nplot = 20 #number of timesteps before plotting
#20 means plotting takes about half the time running
checktime = 100*round((N-1)/500) #factor up front can be anything less than nsteps/4

# For storing the Ricci scalar at r=0
RicciScalar = np.empty(nsteps)

#Initial array of data: Gaussian centred at r=0
phiwidth = 1
phiheight = 0.3 #0.5 is stable, 0.6 is unstable
phi = phiheight*np.asarray([np.exp((-rr**2)/(phiwidth**2)) for rr in rarray])
cutoff = round((N-1)/2)
#phi[cutoff:]=np.zeros(N-cutoff)
#needs Neumann conditions at r=0

# Phi initially = d/dr phi (First order accurate here, but doesn't matter)
Phi = np.asarray([(phi[n+1]-phi[n])/(rarray[n+1]-rarray[n]) for n in range(N-1)])
Phi = np.append([0], Phi)

# Pi initially = a/alpha d/dt phi
#Here just set as a function of Phi; can be changed almost arbitrarily
#Pi = Phi*0.1
Pi = np.zeros(N)

# Initialising a and alpha as empty arrays
a = np.empty(N)
a[0] = 1
alpha = np.empty(N)
alpha[0] = 1


#-----------------------------------------------------------
# Main loop
#-----------------------------------------------------------
def f(phiheight):
    return collapsesimulator(N, rarray, dt, dr, a, alpha, Pi, Phi,
                       nsteps, phiwidth, phiheight, RicciScalar)

critlow, crithigh = bisection(f, 0.336, 0.337, 40)
print("Critical value is between "+str(critlow)+" and "+str(crithigh))
