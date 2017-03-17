import os
import pandas as pd

from LWR import *

## integrate power-laws
## input: observed data of x and y, the smoothness of fitting, and the range of x in logarithm
## output: integrated value
def integratePL(xData,yData,smoothness,xRangeMin,xRangeMax):
    # compute the prediction
    ## observed data
    x = np.array([[1,xi] for xi in xData])
    y = np.array(yData)
    ## to be predicted x
    xPredict = np.linspace(xRangeMin, xRangeMax, num=100)
    xs = np.array([[1,xi] for xi in xPredict])
    
    ## run LWR/Normal Equation and compute the prediction of y
    thetas = [normalEquation(x,y,smoothness,xsi) for xsi in xs]

    ## integrate energy
    accumulator = 0
    for k in range(len(xs)-1):
        #accumulator += 1/(1+thetas[k][1])*10**thetas[k][0]*((10**x[k+1][1])**(thetas[k][1]+1)-(10**x[k][1])**(thetas[k][1]+1))
        accumulator += 10**thetas[k][0]*(10**xs[k][1])**thetas[k][1]*(10**xs[k+1][1]-10**xs[k][1])
    return accumulator


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
xRangeMin = np.log10(10)
xRangeMax = np.log10(1e6)

## run fitting
energy = integratePL(time,luminosity,smoothness,xRangeMin,xRangeMax)

## print result
print(energy)


