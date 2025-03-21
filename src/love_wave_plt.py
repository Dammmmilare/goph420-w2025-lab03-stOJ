import numpy as np
import matplotlib.pyplot as plt

def root_newton_raphson(x0, f, dfdx, tol=1e-6, max_iter=100):
    x = x0
    errors = []
    for i in range(max_iter):
        fx = f(x)
        dfx = dfdx(x)
        if abs(dfx) < 1e-10:
            break  # Avoid division by zero
        x_new = x - fx / dfx
        error = abs((x_new - x) / x_new)
        errors.append(error)
        if error < tol:
            return x_new, i + 1, np.array(errors)
        x = x_new
    return x, max_iter, np.array(errors)

def root_secant_modified(x0, dx, f, tol=1e-6, max_iter=100):
    x = x0
    errors = []
    for i in range(max_iter):
        fx = f(x)
        dfx = (f(x + dx) - fx) / dx
        if abs(dfx) < 1e-10:
            break  # Avoid division by zero
        x_new = x - fx / dfx
        error = abs((x_new - x) / x_new)
        errors.append(error)
        if error < tol:
            return x_new, i + 1, np.array(errors)
        x = x_new
    return x, max_iter, np.array(errors)

def love_wave_dispersion(zeta, f, rho1, rho2, beta1, beta2, H):
    term1 = np.tan(zeta * f) * (rho2 * beta2**2 - rho1 * beta1**2)
    term2 = rho2 * beta2**2 * np.tan(zeta * f) + rho1 * beta1**2
    return term1 - term2

def love_wave_velocity(zeta, H):
    return H / zeta

def compute_love_wave_modes(frequencies, rho1, rho2, beta1, beta2, H, method='newton'):
    zetas = []
    velocities = []
    wavelengths = []
    for f in frequencies:
        roots = []
        for x0 in np.linspace(0.1, 10, 5):  # Multiple initial guesses
            if method == 'newton':
                root, _, _ = root_newton_raphson(x0, 
                                                 lambda z: love_wave_dispersion(z, f, rho1, rho2, beta1, beta2, H), 
                                                 lambda z: (love_wave_dispersion(z + 1e-5, f, rho1, rho2, beta1, beta2, H) - 
                                                            love_wave_dispersion(z, f, rho1, rho2, beta1, beta2, H)) / 1e-5)
            else:
                root, _, _ = root_secant_modified(x0, 1e-5, 
                                                  lambda z: love_wave_dispersion(z, f, rho1, rho2, beta1, beta2, H))
            if root > 0 and not any(np.isclose(root, r, atol=1e-3) for r in roots):
                roots.append(root)
        if roots:  # Only store if roots are found
            zetas.append(roots)
            velocities.append([love_wave_velocity(z, H) for z in roots])
            wavelengths.append([love_wave_velocity(z, H) / f for z in roots])
    return frequencies[:len(zetas)], zetas, velocities, wavelengths

def plot_results(frequencies, velocities, wavelengths):
    if not velocities:
        print("No valid solutions found.")
        return
    
    plt.figure(figsize=(10, 5))
    for mode in range(min(len(v) for v in velocities)):
        plt.plot(frequencies, [v[mode] for v in velocities], label=f'Mode {mode+1}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Love Wave Velocity (m/s)')
    plt.title('Love Wave Velocity vs Frequency')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    for mode in range(min(len(w) for w in wavelengths)):
        plt.plot(frequencies, [w[mode] for w in wavelengths], label=f'Mode {mode+1}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wavelength (m)')
    plt.title('Wavelength vs Frequency')
    plt.legend()
    plt.show()

# Given parameters
rho1, rho2 = 1800, 2500
beta1, beta2 = 1900, 3200
H = 4000  # 4 km
frequencies = np.linspace(0.1, 10, 100)

# Compute and plot
frequencies, zetas, velocities, wavelengths = compute_love_wave_modes(frequencies, rho1, rho2, beta1, beta2, H)
plot_results(frequencies, velocities, wavelengths)
