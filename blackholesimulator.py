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
#----------------------------------------------------------

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
nplot = 20000 #number of timesteps before plotting
#20 means plotting takes about half the time running
checktime = 100*round((N-1)/500) #factor up front can be anything less than nsteps/4

# For storing the Ricci scalar at r=0
RicciScalar = np.empty(nsteps)

# For use immediately below
phiheightarray = np.geomspace(start=5*10**(-9), stop=0.000110132329, num=51)
phiheightarray *= -1
phiheightarray += 0.336035306

#Initial array of data: Gaussian centred at r=0
phiwidth = 1
phiheight = phiheightarray[0] #0.3 is subcritical, 0.4 is critical
phi = phiheight*np.asarray([np.exp((-rr**2)/(phiwidth**2)) for rr in rarray])
cutoff = round((N-1)/2)
#phi[cutoff:]=np.zeros(N-cutoff)
#needs Neumann conditions at r=0

# Phi initially = d/dr phi (Second order accurate)
Phi = np.asarray([(phi[n+1]-phi[n-1])/(2*dr) for n in range(1,N-1)])
Phi = np.append([0], Phi)
Phi = np.append(Phi, [0])

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
# Sorting out the initial figure(s)
#-----------------------------------------------------------

#variables for plotting
yuplim = 2
ylowlim = -1.5

fig = plt.figure()
plt.plot(rarray,Phi, label='Phi')
plt.plot(rarray, Pi,label='Pi')
plt.ylim([ylowlim,yuplim])
filename = 'foo000.png';
fig.set_tight_layout(True);
plt.xlabel("r")
#plt.ylabel("u")
plt.title("t = 0a")
plt.legend()
plt.savefig(filename)
plt.clf()

#-----------------------------------------------------------
# Main loop
#-----------------------------------------------------------

c = 0
Pidot1 = np.empty(N) #1st-order time derivative
Phidot1 = np.empty(N)
Pidot2 = np.empty(N) #2nd-order time derivative
Phidot2 = np.empty(N)
tempPi = np.empty(N)
tempPhi = np.empty(N)

#Main loop
for m in range(nsteps):
    nextpiphi(N, rarray, dt, dr, a, alpha, Pi, Phi,
              Pidot1, Phidot1, Pidot2, Phidot2, tempPi, tempPhi)

    #filling in Ricci Scalar array
    RicciScalar[m] = Ricci(a[0], Phi[0], Pi[0])

    if(m % nplot == 0): #plot results every nplot timesteps
        plt.plot(rarray, Phi, label=r"$\Phi$")
        plt.plot(rarray, Pi,label=r"$\Pi$")
        plt.plot(rarray, alpha, label=r"$\alpha$")
        plt.plot(rarray, a, label='a')
        filename = 'foo' + str(c+1).zfill(3) + '.png'; #0 is at start
        plt.ylim([ylowlim,yuplim])
        plt.xlabel("r")
        plt.legend()
        plt.title("Matter and spacetime configuration at physical time $t$ = %2.2f"%(dt*(m+1)))
        plt.savefig(filename)
        plt.clf()
        c += 1

##    #Error analysis
##    if(m== checktime-1): #obtaining a before checktime, for 2nd-order accuracy in da/dt
##        am = np.copy(a)
##    if(m== checktime):
##        os.system("rm -f *{}.txt".format(str(N)))
##        printres('checkalpha', alpha, N)
##        printres('checkPi', Pi, N)
##        printres('checkPhi', Phi, N)
##        printres('checka', a, N)
##    if(m== checktime+1): #obtaining a after checktime, for 2nd-order accuracy in da/dt
##        ap = np.copy(a)
##        dta = (ap-am)/(2*dt) #Centralised first derivative
##        printres('checkdta', dta, N)
    
#os.system("ffmpeg -y -i 'foo%03d.png' blackholesimulator.m4v")
os.system("rm -f *.png")

RicciPeak = np.amax(RicciScalar)
print("The critical value of the Ricci Scalar is "+str(RicciPeak)+
              " for initial height "+str(phiheight))

plt.plot(np.arange(0, nsteps)*dt, RicciScalar)
plt.xlabel("time")
plt.ylabel("R")
plt.title("Ricci scalar for subcritical collapse")
plt.show()
