import os
import pandas as pd
import matplotlib.pyplot as plt

from LWR import *

# smoothy fitting scattered data points 
# input: observed data of x and y, and the smoothness of fitting
# output: fitted x and y
def smoothFit(xData,yData,smoothness):
    # compute the prediction
    ## observed data
    x = np.array([[1,xi] for xi in xData])
    y = np.array(yData)
    ## to be predicted x
    xPredict = np.linspace(0.8*min(xData), 1.2*max(xData), num=100)
    xs = np.array([[1,xi] for xi in xPredict])
    
    ## run LWR/Normal Equation and compute the prediction of y
    thetas = [normalEquation(x,y,smoothness,xsi) for xsi in xs] 
    yPredict=[np.dot(xsi,thetasi) for xsi,thetasi in zip(xs,thetas)]
    return xPredict,yPredict


###### Example ######

# read data from file
filename = '081008.csv'
path = os.path.abspath('.')
file = os.path.join(path, filename)
data = pd.read_csv(file ,header=0,comment='#')

# value in logarithm
time = np.log10(data['time'])
luminosity = np.log10(data['luminosity'])

## initial parameters
smoothness = 0.08

## run fitting
xPredict,yPredict = smoothFit(time,luminosity,smoothness)

# plot
plt.plot(time,luminosity,'.')
plt.plot(xPredict,yPredict,'.-',markersize=3)

plt.xlabel('Log[Time (s)]',fontsize=12)
plt.ylabel('Log[Luminosity (erg/s)]',fontsize=12)
plt.savefig('LWRexample.pdf')