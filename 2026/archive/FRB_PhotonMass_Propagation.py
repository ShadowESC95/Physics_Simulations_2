
"""
FRB / CMB Propagation Residuals from Spatially Varying Photon Mass
Framework: Stückelberg massive EM with dynamical wall (user theory)

This script computes ABSOLUTE observable residuals:
DeltaF/F ~ P_TM->L

No fine-tuning, EFT-safe parameters, cosmology-correct redshift integration.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Cosmology (LCDM, conservative)
# -----------------------------
c = 3e8
H0 = 70 * 1e3 / (3.086e22)
Omega_m, Omega_L = 0.3, 0.7

def H(z):
    return H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_L)

# -----------------------------
# Effective photon mass model
# -----------------------------
m0 = 1e-14  # eV (below all bounds)
p = 1.0     # mild redshift scaling

def m_gamma(z):
    return m0 * (1+z)**p

# -----------------------------
# Wall profile suppression
# -----------------------------
def wall_factor(freq, L):
    omega = 2*np.pi*freq
    k = omega / c
    return np.exp(-(k*L)**2)

# -----------------------------
# Differential probability
# -----------------------------
def dP_dz(freq, z, L):
    omega = 2*np.pi*freq*(1+z)
    return (m_gamma(z)**2 / omega**2) * wall_factor(freq, L) / H(z)

# -----------------------------
# Integrated probability
# -----------------------------
def integrated_probability(freq, zmax, L, Nz=2000):
    z = np.linspace(0, zmax, Nz)
    integrand = dP_dz(freq, z, L)
    return np.trapz(integrand, z)

# -----------------------------
# FRB regime
# -----------------------------
freq = np.logspace(8, 10, 300)  # 100 MHz – 10 GHz
zmax = 1.5
L_vals = [1e5, 1e7, 1e9]  # thin → thick walls (meters)

results = {L: np.array([integrated_probability(f, zmax, L) for f in freq]) for L in L_vals}

# -----------------------------
# Plot absolute residuals
# -----------------------------
plt.figure(figsize=(8,5))

for L in L_vals:
    plt.loglog(freq, results[L], label=f"Wall thickness L = {L:.0e} m")

# FRB sensitivity bands (conservative)
plt.axhspan(1e-7, 1e-5, color='red', alpha=0.15, label="FRB sensitivity window")

plt.xlabel("Observed Frequency [Hz]")
plt.ylabel(r"Absolute Residual $\Delta F/F$")
plt.title("Predicted FRB Flux Residuals from Photon-Mass Gradients")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()
