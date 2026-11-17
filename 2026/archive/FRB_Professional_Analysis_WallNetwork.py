
"""
FRB Cosmological Propagation Analysis
====================================
Framework: Stückelberg massive electromagnetism with dynamical walls

Implements:
1) Plasma DM likelihood fit and subtraction (f^{-2})
2) Parameter scan over (m_gamma, wall thickness L)
3) Anisotropic wall-network statistics (Poisson crossings)

Outputs:
- Absolute residuals DeltaF/F
- Sensitivity contours
- Diagnostic plots suitable for PRD/PRL
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
# Photon mass model
# -----------------------------
def m_gamma(z, m0, p=1.0):
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
def dP_dz(freq, z, m0, L):
    omega = 2*np.pi*freq*(1+z)
    return (m_gamma(z, m0)**2 / omega**2) * wall_factor(freq, L) / H(z)

def integrated_probability(freq, zmax, m0, L, Nz=1500):
    z = np.linspace(0, zmax, Nz)
    return np.trapz(dP_dz(freq, z, m0, L), z)

# -----------------------------
# Plasma DM likelihood fit
# Fit A * f^{-2} to data
# -----------------------------
def fit_plasma_DM(freq, observed):
    x = freq**(-2)
    A = np.sum(observed * x) / np.sum(x**2)
    return A * x

# -----------------------------
# Anisotropic wall network
# Poisson-distributed wall crossings
# -----------------------------
def wall_network_factor(mean_crossings):
    N = np.random.poisson(mean_crossings)
    return np.sqrt(N) if N > 0 else 0.0

# -----------------------------
# Frequency grid
# -----------------------------
freq = np.logspace(8, 10, 250)   # FRB band
zmax = 1.5

# -----------------------------
# Parameter scan
# -----------------------------
m_vals = np.logspace(-15, -13, 30)   # photon mass [eV]
L_vals = np.logspace(4, 9, 30)       # wall thickness [m]

sensitivity = np.zeros((len(m_vals), len(L_vals)))

for i, m0 in enumerate(m_vals):
    for j, L in enumerate(L_vals):
        P = np.array([integrated_probability(f, zmax, m0, L) for f in freq])
        sensitivity[i, j] = P.max()

# -----------------------------
# Plot parameter-space sensitivity
# -----------------------------
plt.figure(figsize=(7,5))
X, Y = np.meshgrid(L_vals, m_vals)

plt.contourf(X, Y, sensitivity, levels=30)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Wall thickness L [m]")
plt.ylabel("Photon mass m_gamma [eV]")
plt.title("Max FRB Residual DeltaF/F")
cbar = plt.colorbar()
cbar.set_label("DeltaF/F")
plt.tight_layout()
plt.show()

# -----------------------------
# Single realization with wall network + noise
# -----------------------------
m0 = 3e-14
L = 1e5
mean_cross = 1.5

signal = np.array([integrated_probability(f, zmax, m0, L) for f in freq])
signal *= wall_network_factor(mean_cross)

# Add noise
rng = np.random.default_rng(1)
noise = rng.normal(0, 3e-7, size=freq.size)
observed = signal + noise

# Plasma subtraction
plasma_fit = fit_plasma_DM(freq, observed)
residual = observed - plasma_fit

# -----------------------------
# Plot recovery
# -----------------------------
plt.figure(figsize=(8,5))
plt.loglog(freq, np.abs(observed), label="Observed FRB residuals")
plt.loglog(freq, np.abs(plasma_fit), '--', label="Plasma DM fit")
plt.loglog(freq, np.abs(residual), label="Post-subtraction residual")
plt.loglog(freq, signal, ':', label="True photon-mass signal")

plt.axhspan(1e-7, 1e-5, color='red', alpha=0.15, label="FRB sensitivity")
plt.xlabel("Observed Frequency [Hz]")
plt.ylabel(r"$|\Delta F/F|$")
plt.title("FRB Recovery with Plasma Likelihood and Wall Network")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()
