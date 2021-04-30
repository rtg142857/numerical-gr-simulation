import numpy as np
cimport numpy as np
import os
cimport cython
from libc.math cimport M_PI
from libc.math cimport isnan
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.


# Given f(0), f(1), f(2), calculates f(0.5) to second order
cdef double f05(double f0, double f1, double f2):
    return (1/8)*(-f2 + 6*f1 + 3*f0)

# Given f(0), f(1), f(2), calculates f(1.5) to second order
cdef double f15(double f0, double f1, double f2):
    return (1/8)*(-f0 + 6*f1 + 3*f2)

# Used when stepping forward a and alpha in space
cdef double phifac(double Pi, double Phi):
    return Pi*Pi+Phi*Phi

@cython.cdivision(True)
# Calculates the slope of alpha in space times dr
cdef double alphaprime(double dr, double alpha, double a, double r, double phifactor):
    return dr*alpha*(2*M_PI*r*phifactor + (a*a-1)/(2*r))

@cython.cdivision(True)
# Calculates the slope of a in space times dr
cdef double aprime(double dr, double a, double r, double phifactor):
    return dr*a*(2*M_PI*r*phifactor + (1-a*a)/(2*r))

@cython.cdivision(True)
# calculates the slope of Pi in space times dt
cdef double Pidot(double dt, double dr, double r, double alphap, double alpham,
	  double ap, double am, double Phip, double Phim):
    cdef double k
    k = ((r+dr)*(r+dr)*alphap/ap*Phip - (r-dr)*(r-dr)*alpham/am*Phim)/(2*dr)
    return dt /(r*r) * k

@cython.cdivision(True)
# calculates the slope of Phi in space times dt
cdef double Phidot(double dt, double dr, double alphap, double alpham,
		   double ap, double am, double Pip, double Pim):
    return dt * (alphap/ap*Pip - alpham/am*Pim)/(2*dr)

@cython.cdivision(True)
# calculates the Ricci scalar at a point; may be negative of the actual scalar?
cpdef double Ricci(double a, double Phi, double Pi):
    return (-8*M_PI/(a*a))*(Phi*Phi-Pi*Pi)

# prints array to a file including name and the number of data points
def printres(name, array, N):
    with open(name + str(N) + '.txt', 'w') as f:
        for xx in array:
            print(str(xx), file=f)
    return

# Returns true if BH is collapsing
def collapse(ric, m):
    return (ric[m] >= ric[m-1])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
# Function for calculating a and alpha arrays given Pi and Phi
#N≥2 is the number of data points
#rarray is the grid of N space steps
#a and alpha are arrays to be overwritten
#Pi and Phi are arrays representing the configuration of the matter
#Return nothing, overwrite a and alpha
cdef void nextaalpha(size_t N, double[:] rarray, double dr,
		     double[:] a, double[:] alpha, double[:] Pi, double[:] Phi):
    #-----------------------------------------------------------
    # Calculating a and alpha for the first space step r=dr
    # (avoiding a divergence at r=0)
    # (Calculating for r=0 not necessary)
    #-----------------------------------------------------------

    # Initialising variables because Cython
    cdef size_t n
    cdef double Pi_dr2, Phi_dr2, phifactor_dr2, r, k1_alpha, k1_a, alphatemp, atemp
    cdef double k2_alpha, k2_a, phifactor, alpha_infinity

    #At r=dr/2; not necessary to find at r=0
    Pi_dr2 = f05(Pi[0], Pi[1], Pi[2]) #(1/8)*(-Pi[2]+6*Pi[1]+3*Pi[0])
    Phi_dr2 = f05(Phi[0], Phi[1], Phi[2]) #(1/8)*(-Phi[2]+6*Phi[1]+3*Phi[0])
    phifactor_dr2 = phifac(Pi_dr2, Phi_dr2)

    r = rarray[0] #current radius
    # Finding initial slope of (alpha, a)
    k1_alpha = 0 #dr*alpha[0]*(2*np.pi*r*phifactor + (a[0]**2-1)/(2*r))
    k1_a = 0 #dr*a[0]*(2*np.pi*r*phifactor + (1-a[0]**2)/(2*r))
    # Finding slope after half-step
    r += dr/2 #halfway between current and next radius
    alphatemp = alpha[0] + k1_alpha/2
    atemp = a[0] + k1_a/2
    
    k2_alpha = alphaprime(dr, alphatemp, atemp, r, phifactor_dr2)
    k2_a = aprime(dr, atemp, r, phifactor_dr2)
    #Calculating next point
    alpha[1] = alpha[0] + k2_alpha
    a[1] = a[0] + k2_a

    #-----------------------------------------------------------
    # Calculating a and alpha via RK2 in space
    #-----------------------------------------------------------
    #print(str(m))
    for n in range(1, N-1): #calculating a and alpha for n+1
        phifactor = phifac(Pi[n], Phi[n])
        #calculating Pi and Phi at Pi[n+0.5]
        Pi_dr2 = f15(Pi[n-1], Pi[n], Pi[n+1])
        Phi_dr2 = f15(Phi[n-1], Phi[n], Phi[n+1])
        phifactor_dr2 = phifac(Pi_dr2, Phi_dr2)

        r = rarray[n] #current radius
        
        # Finding initial slope of (alpha, a)
        k1_alpha = alphaprime(dr, alpha[n], a[n], r, phifactor)
        k1_a = aprime(dr, a[n], r, phifactor)
        
        # Finding slope after half-step
        r += dr/2 #halfway between current and next radius
        alphatemp = alpha[n] + k1_alpha/2
        atemp = a[n] + k1_a/2
        k2_alpha = alphaprime(dr, alphatemp, atemp, r, phifactor_dr2)
        k2_a = aprime(dr, atemp, r, phifactor_dr2)
        
        #Calculating next point
        alpha[n+1] = alpha[n] + k2_alpha
        a[n+1] = a[n] + k2_a
        
    #Normalising alpha to be 1 at 'infinity'
    alpha_infinity = alpha[N-1]
    if alpha_infinity != 0:
        for n in range(N):
            alpha[n] /= alpha_infinity

    return

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void nextpiphi(size_t N, double[:] rarray, double dt, double dr, double[:] a,
		double[:] alpha, double[:] Pi, double[:] Phi,
              double[:] Pidot1, double[:] Phidot1, double[:] Pidot2,
		double[:] Phidot2, double[:] tempPi, double[:] tempPhi):

    # initialising variables because cython
    cdef size_t n
    cdef double r

    # Updating a and alpha for Pidot1 and Phidot1
    nextaalpha(N, rarray, dr, a, alpha, Pi, Phi)

    #-----------------------------------------------------------
    # Calculating the next step of Pi and Phi via RK2 in time
    #-----------------------------------------------------------
    
    #Finding 1st-order time derivative
    #Spatial derivatives are done centralised
    for n in range(1,N-1):
        r = rarray[n]
        Pidot1[n] = Pidot(dt, dr, r, alpha[n+1], alpha[n-1],
                          a[n+1], a[n-1], Phi[n+1], Phi[n-1])
        Phidot1[n] = Phidot(dt, dr, alpha[n+1], alpha[n-1],
                            a[n+1], a[n-1], Pi[n+1], Pi[n-1])

    #Neumann and Dirichlet BCs
    Pidot1[0] = (4*Pidot1[1]-Pidot1[2])/3 #Neumann
    Pidot1[N-1] = (4*Pidot1[N-2]-Pidot1[N-3])/3 #Neumann
    Phidot1[0] = 0 #Dirichlet
    Phidot1[N-1] = 0 #Dirichlet

    for n in range(N):
        tempPhi[n] = Phi[n] + Phidot1[n]/2
        tempPi[n] = Pi[n] + Pidot1[n]/2

    # Updataing a and alpha for Pidot2 and Phidot2
    nextaalpha(N, rarray, dr, a, alpha, tempPi, tempPhi)

    #Finding 2nd-order time derivative
    for n in range(1,N-1):
        r = rarray[n]
        Pidot2[n] = Pidot(dt, dr, r, alpha[n+1], alpha[n-1],
                          a[n+1], a[n-1], tempPhi[n+1], tempPhi[n-1])
        Phidot2[n] = Phidot(dt, dr, alpha[n+1], alpha[n-1],
                            a[n+1], a[n-1], tempPi[n+1], tempPi[n-1])

    #More BCs
    Pidot2[0] = (4*Pidot2[1]-Pidot2[2])/3
    Pidot2[N-1] = (4*Pidot2[N-2]-Pidot2[N-3])/3
    Phidot2[0] = 0
    Phidot2[N-1] = 0

    #Updating Pi and Phi
    for n in range(N):
        Pi[n] += Pidot2[n]
        Phi[n] += Phidot2[n]

    #Implementing Kreiss-Oliger Dissipation at r=dr
    Pi[1] = (1/7)*(4*Pi[0]+4*Pi[2]-Pi[3])
    Phi[1] = (1/5)*(4*Phi[0]+4*Phi[2]-Phi[3])

    #Doing the same at r=0
    Pi[0]=(1/3)*(4*Pi[1]-Pi[2])
    return

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef collapsesimulator(size_t N, double[:] rarray, double dt, double dr, double[:] a,
			double[:] alpha, double[:] Pi, double[:] Phi,
                size_t nsteps, double phiwidth, double phiheight, double[:] RicciScalar):
    # initialising variables
    cdef size_t m
    cdef double RicciPeak
    cdef double[:] phi, Pidot1, Phidot1, Pidot2, Phidot2, tempPi, tempPhi

    # resetting a and alpha
    a[0] = 1
    alpha[0] = 1
    
    phi = phiheight*np.asarray([np.exp((-rr**2)/(phiwidth**2)) for rr in rarray])
    # Phi initially = d/dr phi (Second order accurate)
    Phi = np.asarray([(phi[n+1]-phi[n-1])/(2*dr) for n in range(1,N-1)])
    Phi = np.append([0], Phi)
    Phi = np.append(Phi, [0])

    # Pi initially = a/alpha d/dt phi
    #Here just set as a function of Phi; can be changed almost arbitrarily
    #Pi = Phi*0.1
    Pi = np.zeros(N)

    Pidot1 = np.empty(N) #1st-order time derivative
    Phidot1 = np.empty(N)
    Pidot2 = np.empty(N) #2nd-order time derivative
    Phidot2 = np.empty(N)
    tempPi = np.empty(N)
    tempPhi = np.empty(N)
    #Main loop for calculating the black hole
    for m in range(nsteps):
        nextpiphi(N, rarray, dt, dr, a, alpha, Pi, Phi,
                  Pidot1, Phidot1, Pidot2, Phidot2, tempPi, tempPhi)

        #filling in Ricci Scalar array
        RicciScalar[m] = Ricci(a[0], Phi[0], Pi[0])

    RicciPeak = np.amax(RicciScalar)
    if (RicciPeak == RicciScalar[nsteps-2]) and (not isnan(RicciScalar[nsteps-1])):
        print("The critical value of the Ricci Scalar is "+str(RicciPeak)+
              " for initial height "+str(phiheight))
        print(RicciPeak)
        return 0
    elif (RicciScalar[nsteps-1] == RicciPeak) or isnan(RicciScalar[nsteps-1]):
        print(str(phiheight)+" is supercritical")
        return 1
    print("The maximum value of the Ricci Scalar is "+str(RicciPeak)+
              " for initial height "+str(phiheight))
    return -1
    

def bisection(f, lowerbound, higherbound, maxiterations):
    f_a_n = f(lowerbound)
    f_b_n = f(higherbound)
    if f_a_n*f_b_n >= 0:
        print("Bisection method fails.")
        return None
    a_n = lowerbound
    b_n = higherbound
    for n in range(1,maxiterations+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f_a_n*f_m_n < 0:
            a_n = a_n
            b_n = m_n
            f_b_n = f_m_n
        elif f_b_n*f_m_n < 0:
            a_n = m_n
            f_a_n = f_m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n,0
        else:
            print("Bisection method fails.")
            return None
    return a_n, b_n
