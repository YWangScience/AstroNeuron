import os
import pandas as pd

from LWR import *

## find the flat region where the spectrum index is almost a constant
## input: x coordinate, spectral index, and the threshold below which is considered as constant
## output: range of x

def findFlat(xData,index,smoothness,threshold):
    # compute the prediction
    ## observed data
    x = np.array([[1,xi] for xi in xData])
    ## to be predicted x
    xPredict = np.linspace(min(xData), max(xData), num=100)
    xs = np.array([[1,xi] for xi in xPredict])
    
    ## run LWR/Normal Equation and compute the prediction of slopes
    slopes = [normalEquation(x,index,smoothness,xsi) for xsi in xs]

    ## initiate parameters
    xsLeft = []
    xsRight = []
    flagLeft = 0
    flagRight =1
    
    for k in range(len(xs)):
        ## left boundary of flat range
        if (abs(slopes[k][1])<threshold and flagLeft==0):
            xsLeft.append(xs[k][1])
            flagLeft = 1
            flagRight = 0
            
        ## right boundary of flat range
        if (abs(slopes[k][1])>threshold and flagRight==0):
            xsRight.append(xs[k-1][1])
            flagLeft = 0
            flagRight = 1
            
        ## right boundary for last data point
        if (k== len(xs)-1 and flagRight==0):
            xsRight.append(xs[-1][1])
            flagLeft = 0
            flagRight = 1
            
    return xsLeft,xsRight


###### Example ######

# read data from file
filename = '081008.csv'
path = os.path.abspath('.')
file = os.path.join(path, filename)
data = pd.read_csv(file ,header=0,comment='#')

# value in logarithm
time = np.log10(data['time'])
index = data['index']

## initial parameters
smoothness = 0.08
threshold = 0.2

## run findFlat code and show the x ranges
xsLeft,xsRight = findFlat(time,index,smoothness,threshold)
[print(10**xLeft,'-', 10**xRight) for xLeft, xRight in zip(xsLeft,xsRight) if xLeft != xRight]

