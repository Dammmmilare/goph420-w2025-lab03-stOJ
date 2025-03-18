import numpy as np

def root_newton_raphson(x0, f, dfdx, tol=1e-6, max_iter=100):
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
    errors = []
    x = x0  

    for i in range(max_iter):
        fx = f(x)
        dfx = dfdx(x)

        if abs(dfx) < 1e-12:  # Prevent division by zero
            raise ValueError("Derivative is close to zero. Newton-Raphson method may fail.")

        x_new = x - fx / dfx
        error = abs((x_new - x) / x_new) if x_new != 0 else 0
        errors.append(error)

        if error < tol:
            return x_new, i + 1, np.array(errors)  # Converged

        x = x_new

    return x, max_iter, np.array(errors)  # If max iterations reached, return last x


def root_secant_modified(x0, dx, f, tol=1e-6, max_iter=100):
    """
    Find the root of a function using the Modified Secant method.

    Parameters
    ----------
    x0 : float
        Initial guess.
    dx : float
        Step size for numerical derivative.
    f : function    
        Function to find the root of.
    tol : float, optional
        Convergence tolerance (default: 1e-6).
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
    errors = []
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = (f(x + dx) - fx) / dx  # Approximate derivative

        if abs(dfx) < 1e-12:  # Prevent division by zero
            raise ValueError("Approximate derivative is close to zero. Method may fail.")

        x_new = x - fx / dfx
        error = abs((x_new - x) / x_new) if x_new != 0 else 0
        errors.append(error)

        if error < tol:
            return x_new, i + 1, np.array(errors)  # Converged

        x = x_new

    return x, max_iter, np.array(errors)  # If max iterations reached, return last x
