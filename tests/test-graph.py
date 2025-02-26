import numpy as np
import matplotlib.pyplot as plt

#Computing the function using the graphichal method to visualize our data points to clearly find the root of the function


#S_max = np.sqrt(H ** 2 * (B1 ** -2 - B2 ** -2))
#S = (0.25 * f) * (2k + 1)
    
def Fz(Z, f):
   # densities in kg/m^3
   p1 = 1800
   p2 = 2500

   # velocities in m/s
   B1 = 1900
   B2 = 3200

   # thicknesses in m
   H = 4000
   return np.tan(2 * np.pi() * f * Z) - (p2 / p1) * np.sqrt((H**2) * (B1 **2 - B2 **2) -Z **2) / Z


import numpy as np

def asymptote_finder():
    # Densities in kg/m^3
    p1 = 1800
    p2 = 2500

    # Velocities in m/s
    B1 = 1900
    B2 = 3200

    # Thickness in m
    H = 4000

    # Compute S_max
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