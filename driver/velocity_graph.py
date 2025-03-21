import numpy as np
import matplotlib.pyplot as plt

def compute_critical_slowness(layer_thickness, shear_velocity_top, shear_velocity_bottom):
    """Determines the maximum horizontal slowness for Love wave propagation."""
    return np.sqrt(layer_thickness**2 * (1/shear_velocity_top**2 - 1/shear_velocity_bottom**2))

def calculate_initial_slowness_estimates(frequency, layer_thickness, shear_velocity_top, shear_velocity_bottom):
    """Generates initial slowness estimates for root finding."""
    max_slowness = compute_critical_slowness(layer_thickness, shear_velocity_top, shear_velocity_bottom)
    slowness_estimates = [0.0]
    index = 0
    slowness = 0.0
    while slowness <= max_slowness:
        slowness = (0.25 / frequency) * (2 * index + 1)
        if slowness < max_slowness:
            slowness_estimates.append(slowness)
        index += 1
    slowness_estimates.append(max_slowness)
    return slowness_estimates

def love_wave_transcendental_equation(slowness, frequency, layer_thickness, density_top, density_bottom, shear_velocity_top, shear_velocity_bottom):
    """Defines the transcendental equation for Love wave dispersion."""
    critical_slowness_squared = layer_thickness**2 * (1/shear_velocity_top**2 - 1/shear_velocity_bottom**2)
    return (density_bottom / density_top) * np.sqrt(critical_slowness_squared - slowness**2) / slowness - np.tan(2 * np.pi * frequency * slowness)

def love_wave_transcendental_equation_derivative(slowness, frequency, layer_thickness, density_top, density_bottom, shear_velocity_top, shear_velocity_bottom):
    """Calculates the derivative of the transcendental equation."""
    critical_slowness_squared = layer_thickness**2 * (1/shear_velocity_top**2 - 1/shear_velocity_bottom**2)
    return (-density_bottom / density_top) * critical_slowness_squared / (slowness**2 * np.sqrt(critical_slowness_squared - slowness**2)) - 2 * np.pi * frequency / np.cos(2 * np.pi * frequency * slowness)**2

def newton_raphson(initial_guess, func, deriv, *args, tolerance=1e-8, max_iterations=100):
    """Performs Newton-Raphson root finding."""
    x = initial_guess
    for _ in range(max_iterations):
        fx = func(x, *args)
        dfx = deriv(x, *args)
        if dfx == 0:
            return x, False  # Avoid division by zero
        next_x = x - fx / dfx
        if abs(next_x - x) < tolerance:
            return next_x, True
        x = next_x
    return x, False  # Did not converge

def extract_phase_velocities(slowness_values, shear_velocity_top, layer_thickness):
    """Converts slowness values to phase velocities."""
    return [np.sqrt(1 / (1/shear_velocity_top**2 - (s/layer_thickness)**2)) for s in slowness_values]

def generate_dispersion_curve(frequencies, layer_thickness, density_top, density_bottom, shear_velocity_top, shear_velocity_bottom):
    """Calculates and plots Love wave dispersion curves."""
    modes = [[], [], []]
    for freq in frequencies:
        initial_slowness_guesses = calculate_initial_slowness_estimates(freq, layer_thickness, shear_velocity_top, shear_velocity_bottom)
        slowness_roots = []
        for initial_guess in initial_slowness_guesses:
          if initial_guess != 0 and (initial_guess != compute_critical_slowness(layer_thickness, shear_velocity_top, shear_velocity_bottom) or love_wave_transcendental_equation(initial_guess, freq, layer_thickness, density_top, density_bottom, shear_velocity_top, shear_velocity_bottom) <= 0):
            root, converged = newton_raphson(initial_guess, love_wave_transcendental_equation, love_wave_transcendental_equation_derivative, freq, layer_thickness, density_top, density_bottom, shear_velocity_top, shear_velocity_bottom)
            if converged:
                slowness_roots.append(root)

        for mode_index, mode in enumerate(modes):
            if mode_index < len(slowness_roots):
                mode.append(slowness_roots[mode_index])

    modes = np.array(modes, dtype=object)
    phase_velocities = [extract_phase_velocities(mode, shear_velocity_top, layer_thickness) for mode in modes]

    for index, velocity_mode in enumerate(phase_velocities):
        plt.plot(frequencies[-len(velocity_mode):], velocity_mode, label=f'Mode {index}')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Velocity (m/s)')
    plt.title('Love Wave Dispersion Characteristics')
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/mode_phase Velocity.png')
    plt.show()


def main():
    """Main execution function."""
    top_layer_density = 1800  # kg/m^3
    bottom_layer_density = 2500  # kg/m^3
    top_layer_shear_velocity = 1900  # m/s
    bottom_layer_shear_velocity = 3200  # m/s
    medium_thickness = 4000  # m
    frequency_range = [0.1, 0.5, 1.0, 1.5, 2.0]  # Hz

    generate_dispersion_curve(frequency_range, medium_thickness, top_layer_density, bottom_layer_density, top_layer_shear_velocity, bottom_layer_shear_velocity)

if __name__ == "__main__":
    main()