import numpy as np
import matplotlib.pyplot as plt
from root_finding_fun import root_newton_raphson

# Given parameters
ρ1, ρ2 = 1800, 2500  # Densities (kg/m³)
β1, β2 = 1900, 3200  # Shear wave velocities (m/s)
H = 4000  # Thickness of surface layer (m)

def g(ζ, f):
    """Dispersion function g(ζ) = 0 for Love waves."""
    term1 = np.tan((ζ * β1) / H)
    term2 = (ρ2 * β2**2) / (ρ1 * β1**2)
    term3 = ζ / np.sqrt(ζ**2 - (β2 / β1)**2)

    return term1 - term2 * term3

def dg_dζ(ζ, f, dx=1e-6):
    """Numerically approximate the derivative of g(ζ)."""
    return (g(ζ + dx, f) - g(ζ, f)) / dx

def compute_cl(ζ):
    """Compute Love wave velocity cL."""
    return ζ * β1

def compute_lambda_L(cL, f):
    """Compute Love wave wavelength λL."""
    return cL / f if f != 0 else np.nan

def solve_z(f, initial_guess=1.0):
    """Finds ζ using Newton-Raphson method with error handling."""
    try:
        z_root, iterations, errors = root_newton_raphson(initial_guess, lambda ζ: g(ζ, f), lambda ζ: dg_dζ(ζ, f))
        print(f"f = {f:.2f} Hz, ζ = {z_root:.6f}, Iterations = {iterations}")  # Debugging output
        return z_root
    except ValueError as e:
        print(f"Failed to find root for f = {f:.2f}: {e}")
        return None  # Prevents NaN from spreading

# Frequency range
freqs = np.linspace(0.1, 10, 100)  # Hz
cl_modes, lambda_L_modes = [], []

for f in freqs:
    ζ = solve_z(f, 1.0)  # Try initial guess ζ = 1.0
    
    if ζ is not None and not np.isnan(ζ):
        cl = compute_cl(ζ)
        lambda_L = compute_lambda_L(cl, f)
        cl_modes.append(cl)
        lambda_L_modes.append(lambda_L)
    else:
        print(f"Skipping f = {f:.2f} Hz due to invalid ζ")

# Debugging output
print("Computed Love wave velocities (first 10):", cl_modes[:10])
print("Computed Wavelengths (first 10):", lambda_L_modes[:10])
print("Total computed points:", len(cl_modes), len(lambda_L_modes))

# Plot Love wave velocity vs. frequency
plt.figure(figsize=(10, 6))
plt.plot(freqs, cl_modes, label='Love Wave Velocity (cL)', color='b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Velocity (m/s)')
plt.title('Love Wave Velocity vs Frequency')
plt.legend()
plt.grid()
plt.ylim(0, max(cl_modes) * 1.1 if cl_modes else 1)  # Ensure valid axis limits
plt.savefig('love_wave_velocity.png')
plt.show()

# Plot wavelength vs frequency
plt.figure(figsize=(10, 6))
plt.plot(freqs, lambda_L_modes, label='Wavelength vs Frequency', color='r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Wavelength (m)')
plt.title('Wavelength vs Frequency')
plt.legend()
plt.grid()
plt.ylim(0, max(lambda_L_modes) * 1.1 if lambda_L_modes else 1)  # Ensure valid axis limits
plt.savefig('wavelength_vs_frequency.png')
plt.show()


# End of src/goph420-lab03/love_wave.py