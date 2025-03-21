import numpy as np
import matplotlib.pyplot as plt

def calculate_asymptotes(frequency, layer_thickness, shear_velocity_1, shear_velocity_2):
    """Calculates the asymptotes for the Love wave dispersion curve."""
    asymptotes = [0.0]
    a = 0.0
    k = 0
    max_zeta = np.sqrt(layer_thickness**2 * (shear_velocity_1**-2 - shear_velocity_2**-2))
    while a < max_zeta:
        a = (0.25 * 1 / frequency) * (2 * k + 1)
        if a < max_zeta:
            asymptotes.append(a)
        k += 1
    asymptotes.append(max_zeta)
    return asymptotes

def dispersion_function(zeta, frequency, layer_thickness, density_1, density_2, shear_velocity_1, shear_velocity_2):
    """Calculates the Love wave dispersion function."""
    const_c = layer_thickness**2 * (shear_velocity_1**-2 - shear_velocity_2**-2)
    if const_c - zeta**2 < 0:
        return np.nan
    return ((density_2 / density_1) * np.sqrt(const_c - zeta**2) / zeta) - np.tan(2 * np.pi * frequency * zeta)

def dispersion_derivative(zeta, frequency, layer_thickness, density_1, density_2, shear_velocity_1, shear_velocity_2):
    """Calculates the derivative of the Love wave dispersion function."""
    const_c = layer_thickness**2 * (shear_velocity_1**-2 - shear_velocity_2**-2)
    if const_c - zeta**2 < 0:
        return np.nan
    return (-(density_2 / density_1) * const_c / (zeta**2 * np.sqrt(const_c - zeta**2))) - 2 * np.pi * frequency * (1 / np.cos(2 * np.pi * frequency * zeta))**2

def newton_raphson(initial_guess, func, derivative, tolerance=1e-6, max_iterations=100):
    """Newton-Raphson method for finding roots."""
    x = initial_guess
    for _ in range(max_iterations):
        fx = func(x)
        if abs(fx) < tolerance:
            return x, _
        dfx = derivative(x)
        if dfx == 0:
            return None, _
        x_new = x - fx / dfx
        x = x_new
    return None, max_iterations

def main():
    density_1 = 1800  # kg/m^3
    density_2 = 2500  # kg/m^3
    shear_velocity_1 = 1900  # m/s
    shear_velocity_2 = 3200  # m/s
    layer_thickness = 4000  # m
    frequencies = [0.1, 0.5, 1.0, 1.5, 2.0]  # Hz

    mode_roots = [[], [], []]

    for frequency in frequencies:
        asymptotes = calculate_asymptotes(frequency, layer_thickness, shear_velocity_1, shear_velocity_2)
        initial_guesses = []
        for j, asymptote in enumerate(asymptotes):
            if asymptote == 0 or (asymptote == np.sqrt(layer_thickness**2 * (shear_velocity_1**-2 - shear_velocity_2**-2)) and dispersion_function(asymptote, frequency, layer_thickness, density_1, density_2, shear_velocity_1, shear_velocity_2) > 0):
                continue
            x0 = asymptote - 1e-3
            if layer_thickness**2 * (shear_velocity_1**-2 - shear_velocity_2**-2) - x0**2 >= 0:
                initial_guesses.append(x0)

        roots = []
        for guess in initial_guesses:
            root_val, _ = newton_raphson(guess, lambda z: dispersion_function(z, frequency, layer_thickness, density_1, density_2, shear_velocity_1, shear_velocity_2), lambda z: dispersion_derivative(z, frequency, layer_thickness, density_1, density_2, shear_velocity_1, shear_velocity_2))
            roots.append(root_val)

        for k, mode in enumerate(mode_roots):
            if k < len(roots):
                mode.append(roots[k])

    mode_roots = np.array(mode_roots, dtype=object)

    wave_speeds_0 = [np.sqrt(1 / (shear_velocity_1**-2 - (r / layer_thickness)**2)) for r in mode_roots[0]]
    wave_speeds_1 = [np.sqrt(1 / (shear_velocity_1**-2 - (r / layer_thickness)**2)) for r in mode_roots[1]]
    wave_speeds_2 = [np.sqrt(1 / (shear_velocity_1**-2 - (r / layer_thickness)**2)) for r in mode_roots[2]]

    wavelengths_0 = [speed / frequencies[i] for i, speed in enumerate(wave_speeds_0)]
    wavelengths_1 = [speed / frequencies[i + 1] for i, speed in enumerate(wave_speeds_1)]
    wavelengths_2 = [speed / frequencies[i + 2] for i, speed in enumerate(wave_speeds_2)]

    plt.plot(frequencies[-len(wavelengths_0):], wavelengths_0, label='Mode 0')
    plt.plot(frequencies[-len(wavelengths_1):], wavelengths_1, label='Mode 1')
    plt.plot(frequencies[-len(wavelengths_2):], wavelengths_2, label='Mode 2')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wavelength (m)')
    plt.title('Wavelength vs. Frequency for Modes 0, 1, 2')
    plt.grid()
    plt.legend()
    plt.savefig('figures/mode_wavelength.png')
    plt.show()

if __name__ == "__main__":
    main()