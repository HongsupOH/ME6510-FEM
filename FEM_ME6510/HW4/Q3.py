import numpy as np
from sympy import integrate, Symbol
import matplotlib.pyplot as plt 

def f(x):
    return 0.2+25*x-200*x**2+675*x**3-900*x**4+400*x**5

def X(a,b,xd):
    return ((b + a) + (b - a)*xd)/2

def dx(a,b):
    return (b-a)/2

def order(num):
    if num==1:
        w = np.array([2])
        xd = np.array([0])
    if num==2:
        w = np.array([1,1])
        xd = np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    if num==3:
        w = np.array([8/9,5/9,5/9])
        xd = np.array([0,-np.sqrt(3/5),np.sqrt(3/5)])
    return w,xd

def exact(a,b):
    x = Symbol('x')
    fcn = 0.2+25*x-200*x**2+675*x**3-900*x**4+400*x**5
    int_fcn = integrate(fcn,(x,a,b))
    return int_fcn

def main(a,b):
    errors = []
    F = []
    for num in [1,2,3]:
        w,xds = order(num)
        for xd in xds:
            F.append(f(X(a,b,xd)))
        dX = dx(a,b)
        I = np.dot(w,F)*dX
        print("Order {}'s answer is {}".format(num,I))
        err = np.abs(I-exact(a,b))*100/exact(a,b)
        errors.append(err)
        F = []
    print("")
    for ind,err in enumerate(errors):
        print("True percent error of order {} is {}%".format(ind+1,err))

    plt.plot([1,2,3],errors)
    plt.grid()
    plt.xlabel("order")
    plt.ylabel("error(%)")
    plt.show()

if __name__=="__main__":
    a = 0
    b = 0.8
    main(a,b)
