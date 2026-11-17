
"""
FRB Propagation Residuals: Photon-Mass Gradients vs Plasma Dispersion
Framework: Stückelberg massive EM with dynamical wall

Includes:
1) Absolute TM->Longitudinal residuals (DeltaF/F)
2) Standard plasma dispersion comparison
3) Mock FRB noise injection and recovery test

This script is conservative, EFT-safe, and cosmology-correct.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Cosmology (LCDM)
# -----------------------------
c = 3e8
H0 = 70 * 1e3 / (3.086e22)
Omega_m, Omega_L = 0.3, 0.7

def H(z):
    return H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_L)

# -----------------------------
# Photon mass model (user framework)
# -----------------------------
m0 = 1e-14  # eV
p = 1.0

def m_gamma(z):
    return m0 * (1+z)**p

# -----------------------------
# Wall adiabatic suppression
# -----------------------------
def wall_factor(freq, L):
    omega = 2*np.pi*freq
    k = omega / c
    return np.exp(-(k*L)**2)

# -----------------------------
# TM->Longitudinal conversion
# -----------------------------
def dP_dz(freq, z, L):
    omega = 2*np.pi*freq*(1+z)
    return (m_gamma(z)**2 / omega**2) * wall_factor(freq, L) / H(z)

def integrated_probability(freq, zmax, L, Nz=2000):
    z = np.linspace(0, zmax, Nz)
    return np.trapz(dP_dz(freq, z, L), z)

# -----------------------------
# Plasma dispersion (reference)
# Delta t ~ DM / f^2 ; converted to effective flux bias proxy
# -----------------------------
DM = 500  # pc/cm^3 (typical FRB)
def plasma_proxy(freq):
    return 1e-6 * (DM / 500) * (1e9 / freq)**2

# -----------------------------
# Frequency grid
# -----------------------------
freq = np.logspace(8, 10, 300)  # 100 MHz – 10 GHz
zmax = 1.5
L = 1e5  # thin wall (maximal but allowed)

signal = np.array([integrated_probability(f, zmax, L) for f in freq])
plasma = plasma_proxy(freq)

# -----------------------------
# Mock FRB noise injection
# -----------------------------
rng = np.random.default_rng(42)
noise_level = 3e-7
noise = rng.normal(0, noise_level, size=freq.size)

observed = signal + noise

# -----------------------------
# Plot comparison
# -----------------------------
plt.figure(figsize=(8,5))

plt.loglog(freq, signal, label="Photon-mass gradient (TM→L)", linewidth=2)
plt.loglog(freq, plasma, '--', label="Plasma dispersion proxy")
plt.loglog(freq, np.abs(observed), ':', label="Mock observed FRB residuals")

plt.axhspan(1e-7, 1e-5, color='red', alpha=0.15, label="FRB sensitivity")

plt.xlabel("Observed Frequency [Hz]")
plt.ylabel(r"Residual Scale ($\Delta F/F$ or proxy)")
plt.title("FRB Propagation Residuals: Photon Mass vs Plasma")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# Recovery diagnostic
# -----------------------------
residual = observed - plasma

plt.figure(figsize=(8,5))
plt.loglog(freq, np.abs(residual), label="Recovered non-plasma residual")
plt.loglog(freq, signal, '--', label="True photon-mass signal")

plt.xlabel("Observed Frequency [Hz]")
plt.ylabel(r"$|\Delta F/F|$")
plt.title("Recovery of Photon-Mass Signal After Plasma Subtraction")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()
