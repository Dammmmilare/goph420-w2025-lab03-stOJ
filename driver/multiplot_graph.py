import numpy as np
import matplotlib.pyplot as plt

def love_wave_dispersion(densities, velocities, thickness, frequencies):
    """
    Plots the dispersion relationship for Love waves.

    Args:
        densities: A tuple of (p1, p2) densities.
        velocities: A tuple of (v1, v2) velocities.
        thickness: The thickness H.
        frequencies: A list of frequencies.
    """
    p1, p2 = densities
    v1, v2 = velocities
    H = thickness
    n_freq = len(frequencies)

    zeta_max = np.sqrt(H**2 * (v1**-2 - v2**-2))

    fig, axes = plt.subplots(n_freq, 1, figsize=(8, 10), sharex=True, sharey=True)
    fig.suptitle("Love Wave Dispersion Curves", fontsize=16)

    for i, freq in enumerate(frequencies):
        def love_func(zeta):
            return (p2 / p1) * np.sqrt(zeta_max**2 - zeta**2) / zeta - np.tan(2 * np.pi * freq * zeta)

        asymptotes = [0]
        a, k = 0, 0
        while a < zeta_max:
            a = 0.25 * (2 * k + 1) / freq
            if a < zeta_max:
                asymptotes.append(a)
            k += 1
        asymptotes.append(zeta_max)
        n_asymptotes = len(asymptotes)

        ax = axes[i]
        for j, asymptote in enumerate(asymptotes):
            if j < n_asymptotes - 1:
                zeta_points = np.linspace(asymptote + 1e-3, asymptotes[j + 1] - 1e-3, 500)
                dispersion_values = love_func(zeta_points)
                ax.plot(zeta_points, dispersion_values, "-r")
            ax.plot([asymptote, asymptote], [-5, 5], "--b")
        ax.grid(True)
        ax.set_xlim(0, zeta_max)
        ax.set_ylim(-5, 5)

    fig.text(0.5, 0.04, "Zeta", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "Dispersion", va="center", rotation="vertical", fontsize=12)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("figures/love_wave_dispersion.png")
    plt.show()

def main():
    densities = (1800, 2500)
    velocities = (1900, 3200)
    thickness = 4000
    frequencies = [0.1, 0.5, 1, 1.5, 2]

    love_wave_dispersion(densities, velocities, thickness, frequencies)

if __name__ == "__main__":
    main()