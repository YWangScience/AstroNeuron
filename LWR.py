import numpy as np
import random

# input: observed y at given x, initial theta and the learning rate episilon
# output: fitted theta
def gradientDescent(x, y, theta, episilon, smoothness, xs):
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
        w = weight(x,xs,smoothness)
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

def normalEquation(x,y,smoothness,xs):
    # weight as diagnol matrix
    w = np.diag(weight(x,xs,smoothness))
    # compute theta using normal equation
    # theta=(x'wx)x′wy
    left = np.linalg.pinv(np.dot(np.transpose(x),np.dot(w,x)))
    right = np.dot(np.transpose(x),np.dot(w,y))
    theta = np.dot(left,right)
    return theta
    
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
    
def weightOrigin(x,xs,tau):
    # distance
    D = x - xs
    # weight
    w = np.exp(-np.diagonal(np.dot(D,np.transpose(D)))/(2*tau*tau))
    return w

