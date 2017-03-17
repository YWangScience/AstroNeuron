# AstroNeuron


Data Fitting
============

The most common function to fit GRB spectrum and light-curve is power-law 
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/dab0fc48b71c47a17b071fb6c7db41b8.svg?invert_in_darkmode" align=middle width=61.243545pt height=14.9075025pt/></p> 
C is the normalization, and <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.537065000000004pt height=12.102549999999994pt/> is the power-law index, they can be found by fitting the observed data. The logarithm of a power-law gives a linear function
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/2952373812b34a8c8067a39cff1d8f53.svg?invert_in_darkmode" align=middle width=190.72019999999998pt height=16.376943pt/></p> 
which enables to apply linear regression, here we first introduce a popular algorithm in the compute science of machine learning, gradient decedent. 

Gradient Decedent
-----------------

Gradient decedent is universal in fitting functions which have partial
derivatives, here the example of fitting luminosity light-curve deals
with a simple linear function, which is written as <p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/f09c56c139c6283e72f821b5419d5478.svg?invert_in_darkmode" align=middle width=78.97559999999999pt height=16.376943pt/></p>
The logarithm linearizes the function
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/ecd9bfa44cb67d9a664ea00c70491815.svg?invert_in_darkmode" align=middle width=185.25209999999998pt height=16.376943pt/></p> From observation, we have <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.379255000000002pt height=14.102549999999994pt/> data
points that the luminosity (<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/d357c0f4e0aed6fa5d9e1de9b3fbc96d.svg?invert_in_darkmode" align=middle width=29.550345pt height=30.970500000000015pt/> in unit of erg/s) at different
time (<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/ad55554f53d626f2820a57a84bb7fc79.svg?invert_in_darkmode" align=middle width=11.995665pt height=27.102240000000002pt/> in unit of s), the superscript <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.681657500000003pt height=21.602129999999985pt/> indicates different data
points, its upper boundary is <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.379255000000002pt height=14.102549999999994pt/>. We generalize the function as
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/1b7da869b1d4d38e484f6368796ddf1b.svg?invert_in_darkmode" align=middle width=71.92465499999999pt height=14.372968499999997pt/></p> where in our example, <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/883b72489b0ca4c4cf94e4e46ae92455.svg?invert_in_darkmode" align=middle width=90.59094pt height=27.102240000000002pt/>,
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/b9d1138e09740efedb690afc244052fa.svg?invert_in_darkmode" align=middle width=111.38539499999999pt height=24.56552999999997pt/>, and <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/8ab2c2fecfc411a20a5e3ae6acf205ac.svg?invert_in_darkmode" align=middle width=121.78881pt height=27.102240000000002pt/>. Einstein
summation convention is adopted to simplify the summing symbol if not
specified.

Now we define a function which measures the difference between the
theoretical value <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/883b72489b0ca4c4cf94e4e46ae92455.svg?invert_in_darkmode" align=middle width=90.59094pt height=27.102240000000002pt/> and the observed value
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/5d8a055b6e149031113aa2cbbe1dc30a.svg?invert_in_darkmode" align=middle width=102.14358000000001pt height=30.970500000000015pt/>, named *cost function*, which is a scalar
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/0bbd07203c3b875fe4bca8832845780a.svg?invert_in_darkmode" align=middle width=135.742035pt height=32.950664999999994pt/></p> To find the best fitting parameters
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.143030500000002pt height=22.745910000000016pt/> is equivalent to minimize the cost function <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.656360000000001pt height=22.381919999999983pt/>. In practice,
we first guess an initial <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/f8fde23553817f1c9e46f796b07fd95a.svg?invert_in_darkmode" align=middle width=12.778260000000003pt height=27.102240000000002pt/>, then repeatedly update the
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/f8fde23553817f1c9e46f796b07fd95a.svg?invert_in_darkmode" align=middle width=12.778260000000003pt height=27.102240000000002pt/> to obtain the minimal <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.656360000000001pt height=22.381919999999983pt/>, the updating method is to change
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/f8fde23553817f1c9e46f796b07fd95a.svg?invert_in_darkmode" align=middle width=12.778260000000003pt height=27.102240000000002pt/> along the direction of the gradient of <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.656360000000001pt height=22.381919999999983pt/>
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/76f9574add291dab79605850bd410575.svg?invert_in_darkmode" align=middle width=101.05507499999999pt height=33.769394999999996pt/></p>
here <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.647503500000004pt height=14.102549999999994pt/> is the step of change, it is called *learning rate* in
machine learning, smaller <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.647503500000004pt height=14.102549999999994pt/> increases the precision but it
takes longer time to converge. The gradient gives
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/e7ed203a109373d944b2509f1cc936a0.svg?invert_in_darkmode" align=middle width=315.7704pt height=36.235155pt/></p>
indicators <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/4fe48dde86ac2d37419f0b35d57ac460.svg?invert_in_darkmode" align=middle width=20.612625000000005pt height=21.602129999999985pt/> written as superscript or subscripts actually have no
difference since a flat metric is implicitly used. The final fitted
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.143030500000002pt height=22.745910000000016pt/> converges after iterations.

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
purpose, if we want to predict the value of <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/bcdeda17201a1389003ceab61d2b0ff3.svg?invert_in_darkmode" align=middle width=19.962690000000002pt height=27.102240000000002pt/> at a given
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/47aed00bf7aee78e0e593b45608b7e42.svg?invert_in_darkmode" align=middle width=20.703540000000004pt height=27.102240000000002pt/>, we first need to compute the distance of observed data with
respect to <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/f84e86b97e20e45cc17d297dc794b3e8.svg?invert_in_darkmode" align=middle width=9.359955000000003pt height=22.745910000000016pt/> <p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/fb0c5098128fa55ec20a6b97c993fca0.svg?invert_in_darkmode" align=middle width=97.569285pt height=15.737732999999999pt/></p> for having a local fitting,
the nearby data shall be more important than the distant ones, we use
*weight* to quantify, a common selection of weight is an exponential
function of the distance
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/7b69c0f3d9159295c45f975e4a2ab759.svg?invert_in_darkmode" align=middle width=191.55345pt height=35.947559999999996pt/></p>
the parameter <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode" align=middle width=9.013125000000002pt height=14.102549999999994pt/> controls the range of effective data points,
affecting the smoothness. The cost function which is going to be
minimized has an additional term of weight
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/bf8c51d107c54f2e9998757937931315.svg?invert_in_darkmode" align=middle width=153.452145pt height=32.950664999999994pt/></p> its gradient becomes
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/ab08c3c0bb9b52c7f7074fcda80a26a6.svg?invert_in_darkmode" align=middle width=355.96275pt height=36.235155pt/></p>
the iteration of <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.143030500000002pt height=22.745910000000016pt/> keeps the same form as before
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/be95025c2ea7a5747384c2d5ff526c89.svg?invert_in_darkmode" align=middle width=100.598355pt height=36.235155pt/></p>

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

For polynomial functions as we used for examples, <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.143030500000002pt height=22.745910000000016pt/> can be solved
directly from the minimal of the cost function <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.656360000000001pt height=22.381919999999983pt/>,
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/ba42a32bf3b781c2e36da61f28cc7ec5.svg?invert_in_darkmode" align=middle width=55.317735pt height=33.769394999999996pt/></p> we expand
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/d8f3fb74fa2a778689b125a8ab6be0cb.svg?invert_in_darkmode" align=middle width=198.44549999999998pt height=36.235155pt/></p>
and <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.143030500000002pt height=22.745910000000016pt/> is obtained
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/5d4cfc755ec7ffbea1449d418af946cb.svg?invert_in_darkmode" align=middle width=197.61885pt height=18.842505pt/></p> this
direct way needs to do the inverse of matrix, which is time consuming if
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.143030500000002pt height=22.745910000000016pt/> has a thousands of dimensions, but for limited number of
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
the parameter from <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.188554000000002pt height=21.10812pt/> to <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.188554000000002pt height=21.10812pt/>, which kindly represents the percentage of
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

Example of LWR. Fitting a light-curve from GRB 081008, blue points are
the data points, green line is the fitting using LWR with smoothness
0.08.
![light-curve of GRB 081008](https://github.com/YWangScience/AstroNeuron/blob/master/LWRexample.png){width="85.00000%"}


Energy integration form scattered data points
=============================================

With the LWR algorithm, we are able to fit the observed scattered data
points with a smooth curve, which is composed of numerous linear
functions, the area of this curve covers
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/60ed6771f4da2b9d800ab445e8f7ef95.svg?invert_in_darkmode" align=middle width=346.40099999999995pt height=36.393555pt/></p>
here k indicates the k*th* linear function.

Continue with the luminosity example, we need to return linear to
logarithmic first
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/b838cbdda19d9326a2913e0945e044e5.svg?invert_in_darkmode" align=middle width=162.00459pt height=20.326185pt/></p>
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/26dd3ae438094ebc0bcecc14c8138b47.svg?invert_in_darkmode" align=middle width=71.58855pt height=14.1888285pt/></p> The energy in the a given time interval is given
by
<p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/31c88483ff7138c8534febd243410863.svg?invert_in_darkmode" align=middle width=476.1058499999999pt height=36.393555pt/></p>

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

The total energy in Fig. \[fig:LWRexample\] is <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/7ca51fc1935ba5d77fbfef820cbfb977.svg?invert_in_darkmode" align=middle width=78.59131500000001pt height=26.70657pt/>
erg.

Static evolution from scattered data points
===========================================

Let’s start with an example, that we are planning to find a time range
that the spectra has no evolution except the amplitude, the spectrum can
always be fitted by a power-law, and we have already known the power-law
amplitude <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/1f0aa5770083d7bade7ac8aafcbfc008.svg?invert_in_darkmode" align=middle width=19.521645000000003pt height=22.381919999999983pt/> and the spectral index <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/9c45015d0ef86d5f2d4cf3cc401f18b2.svg?invert_in_darkmode" align=middle width=16.502145000000002pt height=22.745910000000016pt/> at different scattered
time <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/509bf7d4f0f63616580a39c4ed8b527d.svg?invert_in_darkmode" align=middle width=13.152810000000004pt height=20.14650000000001pt/>. <p align="center"><img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/6044ce6c3b680a5a7be4d285d153cdd1.svg?invert_in_darkmode" align=middle width=61.629645pt height=14.748277499999999pt/></p> where F is the spectrum in unit of
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/21e72c507aa93b3cb7f81aebf3280864.svg?invert_in_darkmode" align=middle width=111.07882499999998pt height=26.70657pt/> and <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.647503500000004pt height=14.102549999999994pt/> is the energy of photon in unit
of <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/490bd9d7208632021d6f132bf88dd2af.svg?invert_in_darkmode" align=middle width=29.861535000000003pt height=22.745910000000016pt/>.

The procedure is first to fit the power-law index at different time
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/9c45015d0ef86d5f2d4cf3cc401f18b2.svg?invert_in_darkmode" align=middle width=16.502145000000002pt height=22.745910000000016pt/> with a smooth curve, then to find the time range of the curve
where its tangent’s slope is close to <img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.188554000000002pt height=21.10812pt/>.

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
<img src="https://rawgit.com/YWangScience/AstroNeuron/master/svgs/358d4d0949e47523757b4bc797ab597e.svg?invert_in_darkmode" align=middle width=20.92629pt height=21.10812pt/>. Six time ranges are found in this example, but 4 of them contains
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
