# AstroNeuron


Data Fitting
============

The most common function to fit GRB spectrum and light-curve is
power-law $$y = C x^\alpha$$ C is the normalization, and $\alpha$ is the
power-law index, they can be found by fitting the observed data. The
logarithm of a power-law gives a linear function
$$log(y) = log(C) + \alpha log(x),$$ which enables to apply linear
regression, here we first introduce a popular algorithm in the compute
science of machine learning, gradient decedent.

Gradient Decedent
-----------------

Gradient decedent is universal in fitting functions which have partial
derivatives, here the example of fitting luminosity light-curve deals
with a simple linear function, which is written as $$L(t) = C t^\alpha$$
The logarithm linearizes the function
$$log(L) = log(C) + \alpha log(t)$$ From observation, we have $m$ data
points that the luminosity ($L^j_{obs}$ in unit of erg/s) at different
time ($t^j$ in unit of s), the superscript $j$ indicates different data
points, its upper boundary is $m$. We generalize the function as
$$h^j =\theta^i x^{ij}$$ where in our example, $h^j=log(L^j)$,
$\theta = \{log(C),\alpha\}$, and $x^{ij} = \{1, log(t^j)\}$. Einstein
summation convention is adopted to simplify the summing symbol if not
specified.

Now we define a function which measures the difference between the
theoretical value $h^j=log(L^j)$ and the observed value
$y^j = log(L^j_{obs})$, named *cost function*, which is a scalar
$$J = \frac{1}{2m} (h^j-y^j)^2.$$ To find the best fitting parameters
$\theta$ is equivalent to minimize the cost function $J$. In practice,
we first guess an initial $\theta^i$, then repeatedly update the
$\theta^i$ to obtain the minimal $J$, the updating method is to change
$\theta^i$ along the direction of the gradient of $J$
$$\theta^i = \theta^i - \epsilon \frac{\partial J}{\partial \theta^i}$$
here $\epsilon$ is the step of change, it is called *learning rate* in
machine learning, smaller $\epsilon$ increases the precision but it
takes longer time to converge. The gradient gives
$$\frac{\partial J}{\partial \theta_i} = \frac{1}{2m} \frac{\partial}{\partial \theta^i} (\theta^i x^{ij}-y^j)^2 = \frac{1}{m}(h^j-y^j) x^{ij}$$
indicators $i,j$ written as superscript or subscripts actually have no
difference since a flat metric is implicitly used. The final fitted
$\theta$ converges after iterations.

**Python sample code: gradient descent**

    import numpy as np

    # input: observed y at given x, initial theta and the learning rate episilon
    # output: fitted theta
    def gradientDescent(x, y, theta, episilon):
        # initiate cost function J, and JPre which stores previous cost
        J = 100
        JPre = J/2
        # relative tolerance absolute tolerance 
        rTol = 1e-5
        aTol = 1e-6 
        # number of data pairs
        m = len(x)
        # updating theta until stable (<tolerance)
        while (not np.allclose(J,JPre,rTol,aTol)):
            # prediction
            h = np.dot(x, np.transpose(theta))
            # cost function
            JPre = J
            J = np.dot(np.transpose(h-y),(h-y)) / (2*m)
            # gradient
            gradient = np.dot(np.transpose(h-y), x) / m
            # update theta
            thetaPre = theta
            theta = theta - alpha * gradient
        return theta

Locally Weighted Regression
---------------------------

The observational data is always discrete, in some work, we need a
smooth curve connecting all the data points to do integration or
extrapolation. An apparent way is to divide all the data points to
several segments, fit each segment with a simple function, then connect
all the simple functions as a curve, the smoothness of the curve depends
on the size of segment. An almost equivalent way of thinking is
predicting a set of y at a given set of x by fitting their nearby
observed data points, then connect all the predicted points. Locally
weighted regression (LWR) is an algorithm designed for the above
purpose, if we want to predict the value of $y^{i*}$ at a given
$x^{i*}$, we first need to compute the distance of observed data with
respect to $\hat{x}$ $$D = x^{ij} - x^{i*}$$ for having a local fitting,
the nearby data shall be more important than the distant ones, we use
*weight* to quantify, a common selection of weight is an exponential
function of the distance
$$\omega^j =  \underset{i}{\Sigma} \exp(-\frac{(x^{ij}-x^{i*})^2}{2 \tau^2})$$
the parameter $\tau$ controls the range of effective data points,
affecting the smoothness. The cost function which is going to be
minimized has an additional term of weight
$$J = \frac{1}{2m} \omega^j (h^j-y^j)^2.$$ its gradient becomes
$$\frac{\partial J}{\partial \theta_i} = \frac{1}{2m} \frac{\partial}{\partial \theta^i} \omega^j (\theta_k x^{kj}-y^j)^2 = \frac{1}{m} \omega^j (h^j-y^j) x^{ij}$$
the iteration of $\theta$ keeps the same form as before
$$\theta^i = \theta^i - \epsilon \frac{\partial J}{\partial \theta_i}$$

**Python sample code: locally weighted regression using gradient
descent**

    import numpy as np

    # input: observed y at given x, the smoothness increases from 0 to 1, and the x-coordinate xs where we want to predict its y value
    # output: the weight of each observed data point with respect to thew one xs to be predicted  
    def weight(x,xs,tau):
        # distance
        D = x - xs
        # weight
        w = np.exp(-np.diagonal(np.dot(D,np.transpose(D)))/(2*tau*tau))
        return w

    # input: observed y at given x, initial theta and the learning rate episilon, the smoothness factor tau, and the x-coordinate xs where we want to predict its y value
    # output: fitted theta at xs
    def gradientDescent(x, y, theta, episilon, tau, xs):
        # initiate cost function J, and JPre which stores previous cost
        J = 100
        JPre = J/2
        # relative tolerance and absolute tolerance 
        rtol = 1e-4
        atol = 1e-5
        # number of data pairs
        m = len(x)
        # updating theta until stable (<tolerance)
        while (not np.allclose(J,JPre,rtol,atol)):
            # weight
            w = weight(x,xs,tau)
            # prediction
            h = np.dot(x, theta)
            # cost function
            JPre = J
            J = np.dot(np.transpose(np.multiply(w,h-y)), h-y) /(2*m)
            # gradient
            gradient = np.dot(np.transpose(np.multiply(w,h-y)), x) / m
            # update theta
            theta = theta - episilon * gradient
        return theta

Normal Equation
---------------

For polynomial functions as we used for examples, $\theta$ can be solved
directly from the minimal of the cost function $J$,
$$\frac{\partial J}{\partial \theta^i} = 0$$ we expand
$$\frac{\partial J}{\partial \theta_i} = \frac{1}{m} \omega^j (h^j-y^j) x^{ij} = 0$$
and $\theta$ is obtained
$$\theta^i = (x^{kj}\omega^{j}x^{ji})^{-1}(x^{kj}\omega^{j}y^{j})$$ this
direct way needs to do the inverse of matrix, which is time consuming if
$\theta$ has a thousands of dimensions, but for limited number of
dimensions, as in our example, normal equation is faster than gradient
descent.

**Python sample code: Normal equation for locally weighted regression**

    import numpy as np

    # input: observed y at given x, the smoothness increases from 0 to 1, and the x-coordinate xs where we want to predict its y value
    # output: the computed theta 
    def normalEquation(x,y,smoothness,xs):
        # weight as diagnol matrix
        w = np.diag(weight(x,xs,smoothness))
        # compute theta using normal equation
        # theta=(x'wx)x′wy
        left = np.linalg.pinv(np.dot(np.transpose(x),np.dot(w,x)))
        right = np.dot(np.transpose(x),np.dot(w,y))
        theta = np.dot(left,right)
        return theta

Example of smoothly fitting light-curve
---------------------------------------

Let’s smoothly fit the luminosity light-curve of GRB 081008 in its
cosmological rest frame, data is saved in the file *081008.csv*, first
we refine the code for the smoothness, smoothness increases by setting
the parameter from $0$ to $1$, which kindly represents the percentage of
data that play effective role in predicting at a given point.

**Python sample code: refined weight function**


    def weight(x,xs,smoothness):
        # distance and its square
        D = x - xs
        DSquare = np.diagonal(np.dot(D,np.transpose(D)))    
        # select normalization factor for a given smoothness
        position = int(np.ceil(smoothness * len(x)))
        tau = np.sort(DSquare)[position]
        # weight
        w = np.exp(-DSquare/(2*tau*tau))
        # set small weights to 0 to speed up the computation
        w[w<1e-4] = 0 
        return w

now let’s read the data using *pandas* and apply our LWR, then export
and make a plot using *matplotlib*.

**Python sample code: example LWR**


    import os
    import pandas as pd
    import matplotlib.pyplot as plt


    # read data from file
    filename = '081008.csv'
    path = os.path.abspath('.')
    file = os.path.join(path, filename)
    data = pd.read_csv(file ,header=0,comment='#')

    # value in logarithm
    time = np.log10(data['time'])
    luminosity = np.log10(data['luminosity'])

    # compute the prediction
    ## observed data
    x = np.array([[1,xi] for xi in time])
    y = np.array(luminosity)
    ## to be predicted x
    xPredict = np.linspace(0.8*min(time), 1.2*max(time), num=100)
    xs = np.array([[1,xi] for xi in xPredict])

    ## initial parameters
    smoothness = 0.08

    ## run LWR and compute the prediction of y
    thetas = [normalEquation(x,y,smoothness,xsi) for xsi in xs] 
    yPredict=[np.dot(xsi,thetasi) for xsi,thetasi in zip(xs,thetas)]

    # plot
    plt.plot(time,luminosity,'.')
    plt.plot(xPredict,yPredict)

    plt.xlabel('Log[Time (s)]',fontsize=12)
    plt.ylabel('Log[Luminosity (erg/s)]',fontsize=12)

![Example of LWR. Fitting a light-curve from GRB 081008, blue points are
the data points, green line is the fitting using LWR with smoothness
0.08.[]{data-label="fig:LWRexample"}](LWRexample.pdf){width="85.00000%"}

Energy integration form scattered data points
=============================================

With the LWR algorithm, we are able to fit the observed scattered data
points with a smooth curve, which is composed of numerous linear
functions, the area of this curve covers
$$\int h dx  \simeq  \frac{1}{2} \theta_{ik} x^2 |_{x=x_{ik}}^{x=x_{ik+1}} ~~\text{or}~~ \theta_{ik} x_{ik} (x_{ik+1} -x_{ik})$$
here k indicates the k*th* linear function.

Continue with the luminosity example, we need to return linear to
logarithmic first
$$L(t) = C t^\alpha = 10^{\theta_{0k}} t_k^{\theta_{1k}}$$
$$t_k = 10^{x_{1k}}$$ The energy in the a given time interval is given
by
$$E = \int L(t) dt \simeq \frac{1}{1+\theta_{1k}} 10^{\theta_{0k}} t^{1+\theta_{1k}} |_{t=t_{k}}^{t=t_{k+1}}  ~~\text{or}~~ 10^{\theta_{0k}} t_{k}^{\theta_{1k}} (t_{k+1}-t_{k})$$

**Python sample code: Integration of power-law**

    ## integrate power-laws
    ## input: the x-coordinates and corresponding thetas to be integrated
    ## output: integrated value
    def integratePL(xs,thetas):
            accumulator = 0
            for k in range(len(xs)-1):
                accumulator += 10**thetas[k][0]*(10**xs[k][1])**thetas[k][1]*(10**xs[k+1][1]-10**xs[k][1])
            return accumulator

    ## running the integration       
    integratePL(xs,thetas)       
    2.1387610450550029e+52

The total energy in Fig. \[fig:LWRexample\] is $2.14 \times 10^{52}$
erg.

Static evolution from scattered data points
===========================================

Let’s start with an example, that we are planning to find a time range
that the spectra has no evolution except the amplitude, the spectrum can
always be fitted by a power-law, and we have already known the power-law
amplitude $A_k$ and the spectral index $\beta_k$ at different scattered
time $t_k$. $$F = A \epsilon^\beta$$ where F is the spectrum in unit of
$s^{-1}cm^{-2} keV^{-1}$ and $\epsilon$ is the energy of photon in unit
of $keV$.

The procedure is first to fit the power-law index at different time
$\beta_k$ with a smooth curve, then to find the time range of the curve
where its tangent’s slope is close to $0$.

**Python sample code: constant spectral index**

    ##  read spectral index beta from data of GRB 081008
    betas = data['index']
    slopes = [normalEquation(x,betas,smoothness,xsi) for xsi in xs]

    ## input: x coordinate, slopes, and the threshold below which is considered as flat
    ## output: range where
    def findFlat(xs,slopes,threshold):
        ## initiate parameters
        xsLeft = []
        xsRight = []
        flagLeft = 0
        flagRight =1
        
        for k in range(len(xs)):
            ## left boundary of flat range
            if (abs(slopes[k][1]) < threshold and flagLeft==0):
                xsLeft.append(xs[k][1])
                flagLeft = 1
                flagRight = 0
                
            ## right boundary of flat range
            if (abs(slopes[k][1]) > threshold and flagRight==0):
                xsRight.append(xs[k-1][1])
                flagLeft = 0
                flagRight = 1
                
            ## right boundary for last data point
            if (k == len(xs)-1 and flagRight == 0):
                xsRight.append(xs[-1][1])
                flagLeft = 0
                flagRight = 1
                
        return xsLeft,xsRight
     
    ## run findFlat function for spectral index and show the x ranges
    xsLeft,xsRight = findFlat(xs,slopes,0.2)
    [(xLeft, xRight) for xLeft, xRight in zip(xsLeft,xsRight)]
    [(1.4427274734747277, 1.4427274734747277),
     (2.0717011320238088, 2.0717011320238088),
     (2.5071444340962494, 2.5071444340962494),
     (2.7974399688112097, 3.3296484491219704),
     (3.5231788055986115, 3.5231788055986115),
     (4.1037698750285321, 5.9906908506757759)]

    [(xLeft, xRight) for xLeft, xRight in zip(xsLeft,xsRight) if xLeft != xRight]
    [(2.7974399688112097, 3.3296484491219704),
     (4.1037698750285321, 5.9906908506757759)]

In the above example, the curve is composed of 100 points, and the
threshold is that the slope of the spectral index curve is smaller than
$0.2$. Six time ranges are found in this example, but 4 of them contains
only one point, which can be neglected since the time interval is quite
small, probably just transitions, finally two time ranges we shall
consider.

If we have multi-wavelength observations, the static time range is the
intersection of the static time ranges in different energy bands. In the
optical, the evolution of spectra, counterpart of spectral index in
X-ray, is the color index between different bands.

**Python sample code: Intersection of two lists of time ranges**

    ## input: two time ranges lists
    ## output: intersected time ranges
    def intersect(timeRangesOne, timeRangesTwo):
        timeRanges = []
        for timeRangeOne in timeRangesOne:
            for timeRangeTwo in timeRangesTwo:
                if timeRangeOne[1] < timeRangeTwo[0] : continue
                if timeRangeOne[0] > timeRangeTwo[1] : continue
                timeRanges.append((max(timeRangeOne[0],timeRangeTwo[0]), min(timeRangeOne[1],timeRangeTwo[1])))
        return timeRanges
