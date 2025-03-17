import numpy as np
import matplotlib.pyplot as plt

def root_newton_raphson(x0, f, dfdx, tol=1e-6, max_iter=100):

    """ 
    Find the root of a function using the Newton-Raphson method.
    
    parameters
    ----------
    x0 : float
        Initial guess.
    f : function
        Function to find the root of.
    dfdx : function
        The function that gives the derivative of f.
    tol : float
        The desired tolerance.
    max_iter : int
        Maximum number of iterations.s
    
    Returns
    -------
    x : The first return value (type float) should be the final estimate of the root.
    f: The second return value (type value) should be the value of the function at the root.
    dfdx: The third return the value (type numpy.ndarray) should be a one-dimensional vector of the approximate relative error at each iteration.
    tol : The fourth return value (type float) should be the tolerance.
    max_iter : The fifth return value (type int) should be the maximum number of iterations.

    """
    errors = []
    x = x0 # Initial guess
    for i in range(max_iter):
        fx = f(x)
        dfx = dfdx(x)

        if abs(dfx) < 1e-12:
            raise ValueError("Derivative is close to zero")
        break

    x_new = x - fx / dfx
    error = abs((x_new - x) / x_new) if x_new != 0 else 0
    errors.append(error)

    if error < tol:
        return x_new, i + 1, np.array(errors)
    
    x = x_new
    return x, f(x), np.array(errors)
    

def root_secant_modified(x0, dx, f, tol=1e-6, max_iter=100):
    """
    Find the root of a function using the secant method.
    
    Parameters
    ----------
    x0 : float
        Initial guess. 
    dx : float
        Step size for the numerical derivative.
    f : function    
        Function to find the root of.
    tol : float
    
    Returns
    -------
     x : The first return value (type float) should be the final estimate of the root.
    f: The second return value (type value) should be the value of the function at the root.
    dx: The third return the value (type numpy.ndarray) should be a one-dimensional vector of the approximate relative error at each iteration.
    tol : The fourth return value (type float) should be the tolerance.
    max_iter : The fifth return value (type int) should be the maximum number of iterations.

    """
    errors = []
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = (f(x + dx) - fx) / dx 

        if abs(dfx) < 1e-12:
            raise ValueError("Derivative is close to zero")
            

        x_new = x - fx / dfx
        error = abs((x_new - x) / x_new) if x_new != 0 else 0
        errors.append(error)

        if error < tol:
            return x_new, i + 1, np.array(errors)

        x = x_new
        
    return x, f(x), np.array(errors)
