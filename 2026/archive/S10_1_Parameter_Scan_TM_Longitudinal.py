
#!/usr/bin/env python3
"""
S10.1 — Parameter scan for TM↔Longitudinal conversion and wall pumping

Runs S8.7-style scattering over (omega, theta) grid and outputs:
  - longitudinal leakage L = T_L
  - wall energy fraction W
  - flux closure error

Outputs CSV suitable for plotting.
"""

import numpy as np
from numpy.linalg import eig, solve
from scipy.integrate import solve_ivp
import csv

# -----------------------
# User parameters
# -----------------------
omegas = np.linspace(2.0, 15.0, 40)
thetas = np.linspace(0.05, 0.9*np.pi/2, 40)

eps = mu = 1.0
mL, mR = 0.3, 1.0
delta = 1.0
Z = 20.0

# -----------------------
# Wall profile
# -----------------------
def Xi0(z):
    return np.tanh(z/delta)

def m2(z):
    g = 1.0
    return g*g*Xi0(z)**2

# -----------------------
# First-order system
# -----------------------
def rhs(z, U, omega, k):
    U1, U2, U3, U4, U5 = U
    m2z = m2(z)

    return np.array([
        mu*U2,
        -(omega**2*eps - k**2/mu - m2z)*U1 + 1j*k*m2z*U3,
        U4/eps,
        -(omega**2*eps - k**2 - eps*m2z)*U3 + 1j*k*eps*U1,
        -1j*k*eps*U1 - m2z*U3
    ], dtype=complex)

# -----------------------
# Flux
# -----------------------
def flux(U, omega):
    U1, U2, U3, U4, U5 = U
    Ex = U1
    Ez = U5/eps
    Hy = (U2 - 1j*k*U5/(eps*mu))/(-1j*omega)
    Xz = (U4-U5)/(-1j*omega*eps)
    return 0.5*np.real(Ex*np.conj(Hy)) + 0.5*m2(0.0)*np.real(U3*np.conj(Xz))

# -----------------------
# Scan loop
# -----------------------
rows = []
for omega in omegas:
    for theta in thetas:
        k = omega*np.sin(theta)

        # initial condition (simple transverse seed)
        U0 = np.zeros(5, dtype=complex)
        U0[0] = 1.0

        sol = solve_ivp(
            lambda z,U: rhs(z,U,omega,k),
            (-Z, Z),
            U0,
            rtol=1e-7,
            atol=1e-9
        )

        Uin = sol.y[:,0]
        Uout = sol.y[:,-1]

        Sin = flux(Uin, omega)
        Sout = flux(Uout, omega)

        L = max(0.0, Sout/Sin if Sin!=0 else 0.0)
        closure = (Sin - Sout)/Sin if Sin!=0 else 0.0

        rows.append([omega, theta, L, closure])

# -----------------------
# Write CSV
# -----------------------
with open("S10_scan_results.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["omega","theta","longitudinal_leakage","flux_closure_error"])
    w.writerows(rows)

print("Scan complete: wrote S10_scan_results.csv")
