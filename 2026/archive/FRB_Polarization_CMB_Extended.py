
"""
Extended FRB + CMB Analysis
==========================
Adds:
1) Polarization-dependent observables (TM vs TE)
2) CMB consistency exclusion region
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Cosmology
# -----------------------------
c = 3e8
H0 = 70 * 1e3 / (3.086e22)
Omega_m, Omega_L = 0.3, 0.7

def H(z):
    return H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_L)

# -----------------------------
# Photon mass model
# -----------------------------
def m_gamma(z, m0):
    return m0 * (1+z)

# -----------------------------
# Wall factor
# -----------------------------
def wall_factor(freq, L):
    omega = 2*np.pi*freq
    return np.exp(-(omega*L/c)**2)

# -----------------------------
# Conversion probabilities
# -----------------------------
def P_TM(freq, zmax, m0, L, Nz=1500):
    z = np.linspace(0, zmax, Nz)
    omega = 2*np.pi*freq*(1+z)
    integrand = (m_gamma(z, m0)**2 / omega**2) * wall_factor(freq, L) / H(z)
    return np.trapz(integrand, z)

def P_TE(freq):
    return np.zeros_like(freq)  # no longitudinal coupling

# -----------------------------
# Frequency grids
# -----------------------------
freq_FRB = np.logspace(8, 10, 300)
freq_CMB = np.logspace(10, 12, 200)

# -----------------------------
# Parameters
# -----------------------------
m_vals = np.logspace(-15, -13, 40)
L_vals = np.logspace(4, 9, 40)

# -----------------------------
# CMB exclusion threshold
# -----------------------------
CMB_limit = 1e-8

excluded = np.zeros((len(m_vals), len(L_vals)))
allowed = np.zeros_like(excluded)

for i, m0 in enumerate(m_vals):
    for j, L in enumerate(L_vals):
        P_cmb = max(P_TM(f, 1100, m0, L) for f in freq_CMB)
        if P_cmb > CMB_limit:
            excluded[i, j] = 1
        else:
            allowed[i, j] = 1

# -----------------------------
# Plot exclusion region
# -----------------------------
plt.figure(figsize=(7,5))
X, Y = np.meshgrid(L_vals, m_vals)

plt.contourf(X, Y, excluded, levels=[0.5,1.5], colors=['#ffcccc'])
plt.contour(X, Y, allowed, levels=[0.5], colors='k')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Wall thickness L [m]")
plt.ylabel("Photon mass m_gamma [eV]")
plt.title("CMB Exclusion (red) and Allowed Region (white)")
plt.tight_layout()
plt.show()

# -----------------------------
# Polarization diagnostic
# -----------------------------
m0 = 3e-14
L = 1e5

Ptm = np.array([P_TM(f, 1.5, m0, L) for f in freq_FRB])
Pte = P_TE(freq_FRB)

plt.figure(figsize=(8,5))
plt.loglog(freq_FRB, Ptm, label="TM (longitudinal-coupled)")
plt.loglog(freq_FRB, Pte + 1e-20, '--', label="TE (null channel)")

plt.xlabel("Observed Frequency [Hz]")
plt.ylabel(r"$\Delta F/F$")
plt.title("Polarization-Dependent FRB Residuals")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
