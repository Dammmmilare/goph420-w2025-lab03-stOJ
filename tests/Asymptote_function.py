import numpy as np
import matplotlib.pyplot as plt

# Computing the function using the graphichal method to visualize our data points to clearly find the root of the function
"""Maximum value equation: S_max = np.sqrt(H ** 2 * (B1 ** -2 - B2 ** -2)) and  Step size for constant F and multiple k values: S = (0.25 * f) * (2k + 1)"""
    
def Fz(Z, f):
   # density (kg/m^3)
   d1 = 1800
   d2 = 2500

   # velocity (m/s)
   B1 = 1900
   B2 = 3200

   # Area thickness (m)
   H = 4000
   return np.tan(2 * np.pi() * f * Z) - (d2 / d1) * np.sqrt((H**2) * (B1 **2 - B2 **2) -Z **2) / Z

def dFz(Z, f):
   # density (kg/m^3)
   d1 = 1800
   d2 = 2500

   # velocity (m/s)
   B1 = 1900
   B2 = 3200

   # Area thickness (m)
   H = 4000
   return - d2/d1 * np.sqrt(H**2 * ( B1 ** -1 - B2)-Z**2) / Z**2 - d2/1 * 1/Z * (H**2 * B1 ** -2 - B2 ** -2 - Z**2)**0.5 - 2 * np.pi * f * 1/np.cos(2 * np.pi * f * Z)**2

# Finding the asymptote of the function
def asymptote_finder():
    # Densities in kg/m^3
    p1 = 1800
    p2 = 2500

    # Velocity (m/s)
    B1 = 1900
    B2 = 3200

    # Area Thickness in (m)
    H = 4000

    # Computing the value of S_max
    S_max = np.sqrt(H ** 2 * (B1 ** -2 - B2 ** -2))

    # Initialize variables
    S_list = []
    k = 0  # Start from 0
    f = 1
    step = 1  # Define a small step size to increment k
    print(S_max)

    while k <= S_max:
        S = (0.25 * 1/f) * (2 * k + 1)
        if S > S_max:
            break
        else:
            S_list.append(S)
        k += step  # Increment k

    return S_list  # Return the computed values

# Call the function
result = asymptote_finder()

print(result) 