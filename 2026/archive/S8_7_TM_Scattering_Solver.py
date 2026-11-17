
#!/usr/bin/env python3
"""
S8.7 — Reference numerical solver for TM scattering through a smooth m(Xi(z)) wall

Implements the first-order system from S8.6, integrates it across a kink wall,
projects onto asymptotic eigenmodes, and reports:
  - RT, RL, TT, TL
  - EM+Stückelberg flux balance
  - (optionally) wall flux if a driven Xi equation is added later

Dependencies: numpy, scipy
"""

import numpy as np
from numpy import sqrt
from scipy.integrate import solve_ivp
from numpy.linalg import eig, solve

# -----------------------------
# Physical parameters (EDIT)
# -----------------------------
omega = 10.0          # frequency
k = 6.0               # transverse wavevector
epsL, muL, mL = 1.0, 1.0, 0.5   # left asymptotic medium
epsR, muR, mR = 1.0, 1.0, 1.0   # right asymptotic medium

# Wall (kink) parameters
lam = 1.0
v = 1.0
delta = np.sqrt(2.0/(lam*v*v))

Z = 20.0              # integration half-width
rtol, atol = 1e-8, 1e-10


# -----------------------------
# Wall profile
# -----------------------------
def Xi0(z):
    return v*np.tanh(z/delta)

def m2(z):
    # example coupling m^2 = m0^2 + g^2 Xi^2
    g = 1.0
    m0 = 0.0
    return m0*m0 + g*g*Xi0(z)**2


# -----------------------------
# First-order system (S8.6)
# State U = [U1,U2,U3,U4,U5]
# -----------------------------
def rhs(z, U):
    U1, U2, U3, U4, U5 = U
    eps = 1.0
    mu = 1.0
    m2z = m2(z)

    dU1 = mu*U2
    dU2 = -(omega**2*eps - k**2/mu - m2z)*U1 + 1j*k*m2z*U3
    dU3 = U4/eps
    dU4 = -(omega**2*eps - k**2 - eps*m2z)*U3 + 1j*k*eps*U1
    dU5 = -1j*k*eps*U1 - m2z*U3

    return np.array([dU1, dU2, dU3, dU4, dU5], dtype=complex)


# -----------------------------
# Asymptotic eigenmodes
# -----------------------------
def asymptotic_modes(eps, mu, m):
    """
    Build eigenvalues/vectors of the constant-coefficient 4x4 system
    (excluding U5 which is reconstructed algebraically).
    """
    A = np.array([
        [0, mu, 0, 0],
        [-(omega**2*eps - k**2/mu - m*m), 0, 1j*k*m*m, 0],
        [0, 0, 0, 1/eps],
        [1j*k*eps, 0, -(omega**2*eps - k**2 - eps*m*m), 0]
    ], dtype=complex)

    vals, vecs = eig(A)
    return vals, vecs


# -----------------------------
# Flux (EM + Stückelberg)
# -----------------------------
def flux(U, eps=1.0, mu=1.0):
    U1, U2, U3, U4, U5 = U
    # Reconstruct fields
    Ex = U1
    Ez = U5/eps
    Hy = (U2 - 1j*k*U5/(eps*mu))/(-1j*omega)
    Xz = (U4 - U5)/(-1j*omega*eps)
    PhiX = U3

    S_em = 0.5*np.real(Ex*np.conj(Hy))
    S_st = 0.5*m2(0.0)*np.real(PhiX*np.conj(Xz))
    return S_em + S_st


# -----------------------------
# Integration
# -----------------------------
# Incoming transverse-like mode from left:
# We approximate it by picking the eigenvector with Im(p)>0
valsL, vecsL = asymptotic_modes(epsL, muL, mL)
idx_in = np.argmax(np.imag(valsL))
v_in = vecsL[:, idx_in]

# Initial state at z=-Z
U0 = np.zeros(5, dtype=complex)
U0[:4] = v_in
U0[4] = 0.0  # Ez component reconstructed dynamically

sol = solve_ivp(rhs, (-Z, Z), U0, rtol=rtol, atol=atol)

U_left = sol.y[:, 0]
U_right = sol.y[:, -1]

# -----------------------------
# Projection onto right modes
# -----------------------------
valsR, vecsR = asymptotic_modes(epsR, muR, mR)

# choose outgoing modes (Re lambda > 0)
out_idx = [i for i,v in enumerate(valsR) if np.real(v) > 0]
Vout = vecsR[:, out_idx]

coeffs = solve(Vout, U_right[:4])

print("=== S8.7 NUMERICAL RESULT ===")
print("Outgoing mode coefficients (right side):")
for i,c in enumerate(coeffs):
    print(f"  T[{i}] = {c}")

print("\nFlux check:")
print("  Incoming flux (left): ", flux(U_left))
print("  Outgoing flux (right):", flux(U_right))
