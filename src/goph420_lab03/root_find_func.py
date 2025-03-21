# This file contains the implementation of the Newton-Raphson method for root finding.
import numpy as np

# Define the function to find the root of using the Newton-Raphson method
# The function takes the initial guess x0, the function f, the derivative of f dfdx, the tolerance tol, and the maximum number of iterations max_iter as input arguments
def newton_raphson_root(x0, f, dfdx, tol=5e-6, max_iter=100):
    """ 
    Find the root of a function using the Newton-Raphson method.

    Parameters
    ----------
    x0 : float
        Initial guess.
    f : function
        Function to find the root of.
    dfdx : function
        The function that gives the derivative of f.
    tol : float, optional
        The desired tolerance (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 100).

    Returns
    -------
    x_new : float
        Final estimate of the root.
    iterations : int
        Number of iterations taken to converge.
    errors : numpy.ndarray
        A one-dimensional array of the approximate relative error at each iteration.
    """
    
    # Initialize the error list
    x = x0
    eps_a = 1.0
    iter = 0
    errors = []

    # Perform the Newton-Raphson method
    # The loop continues until the relative error is less than the tolerance or the maximum number of iterations is reached
    while eps_a > tol and iter < max_iter:
        delta_x = -f(x) / dfdx(x)
        x += delta_x
        eps_a = np.abs(delta_x / x)
        errors.append(eps_a)
        iter += 1

    # Check if the maximum number of iterations was reached without convergence
    # If the maximum number of iterations is reached without convergence, a warning message is printed
    if iter >= max_iter and eps_a > tol:
        print(f"Warning: {iter} iterations completed with a relative error of {eps_a}")

    # Return the final estimate of the root, the number of iterations, and the list of errors
    return x, iter, np.array(errors)
