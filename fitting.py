import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def bestfit3(x, amp, period, offset, slope, intercept): # function; all arguments but first are to be optimised
    return np.sin((2*np.pi*x)/(period)-offset)*amp+slope*x+intercept

def goodness(xdata, ydata, res): #Goodness-of-fit function, lower result = better
    sqintegral = 0
    sqresint = 0
    for i in range(len(xdata)-1):
        sqintegral += ((xdata[i+1]-xdata[i])*ydata[i])**2/ydata[i]
        sqresint += ((xdata[i+1]-xdata[i])*res[i])**2/ydata[i]
    return sqresint/sqintegral

#cutting out unimportant data
xdata = np.genfromtxt("maxriccix.txt") # replace with target file names
ydata = np.genfromtxt("maxricciy.txt")
##ymax = np.amax(ydatauncut)
##for i in range(len(xdatauncut)):
##    if ydatauncut[i] >= ymax*1e-2:
##        xdata = np.append(xdata, [xdatauncut[i]])
##        ydata = np.append(ydata, [ydatauncut[i]])


dp0 = [1, 4, 1, -0.758, 1.33]

#plotting actual data
plt.subplot(2, 1, 1)
plt.plot(xdata, ydata, 'b-', label='data')
plt.ylabel('Relative emissivity density')
#plt.title(str(file))

try: #fitting function to data
    popt3, pcov3 = curve_fit(bestfit3, xdata, ydata,
                             p0=dp0, maxfev=100000,
                             method='lm')
    
except Exception as e: #if data doesn't fit, plot initial guess instead
    plt.plot(xdata, bestfit3(xdata, *dp0), 'g-')
    print(e)

else: #plotting properly fitted data
    plt.plot(xdata, bestfit3(xdata, *popt3), 'r-',
             label='Optimised data: amp=%5.3e, period=%5.3f, offset=%5.3f,slope=%5.3f, intercept=%5.3f'
             % tuple(popt3))
    plt.plot(xdata, bestfit3(xdata, *dp0), 'g-',
             label='Guess')
    plt.legend()
    
    #plotting subplot
    res = ydata-bestfit3(xdata, *popt3)
    plt.subplot(2, 1, 2)
    plt.plot(xdata, res)
##    plt.xlabel('Frequency/30GHz')
##    plt.ylabel('Residual')
##    plt.xscale("log")
#    print('Goodness of fit = '+ str(goodness(xdata, ydata, res)))

finally:
    plt.show()
