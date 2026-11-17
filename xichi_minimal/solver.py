"""
Time-domain solver for the 1+1 dimensional Xi-chi model.

The right-hand sides come from model.numeric_rhs, which generates them from the
symbolic Euler-Lagrange equations. Scheme: second-order centred differences in
space and velocity-Verlet (leapfrog) in time, which is symplectic, so the
discrete energy has no secular drift. Stable for dt < dx (Courant condition,
tightened slightly by the mass terms).

Boundaries: the end values of both fields are held fixed. Use a domain large
enough that nothing reaches them during the run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import model


@dataclass
class Params:
    g: float = 1.0
    lam: float = 2.0
    v: float = 1.0

    @property
    def photon_mass(self):
        return self.g * self.v

    @property
    def delta(self):
        return np.sqrt(2.0 / self.lam) / self.v


@dataclass
class State:
    x: np.ndarray
    xi: np.ndarray
    a: np.ndarray
    xi_t: np.ndarray
    a_t: np.ndarray
    time: float = 0.0


class Solver:
    def __init__(self, params: Params, x: np.ndarray, dt: float):
        self.p = params
        self.x = np.asarray(x, dtype=float)
        self.dx = float(self.x[1] - self.x[0])
        if dt >= self.dx:
            raise ValueError("dt must be smaller than dx (Courant condition)")
        self.dt = float(dt)
        self._acc_xi, self._acc_a = model.numeric_rhs(params.g, params.lam, params.v)

    def laplacian(self, f):
        out = np.zeros_like(f)
        out[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / self.dx**2
        return out

    def accelerations(self, xi, a):
        acc_xi = self._acc_xi(xi, a, self.laplacian(xi), self.laplacian(a))
        acc_a = self._acc_a(xi, a, self.laplacian(xi), self.laplacian(a))
        acc_xi = np.asarray(acc_xi, dtype=float) + np.zeros_like(xi)
        acc_a = np.asarray(acc_a, dtype=float) + np.zeros_like(a)
        acc_xi[[0, -1]] = 0.0          # fixed end values
        acc_a[[0, -1]] = 0.0
        return acc_xi, acc_a

    def step(self, s: State, n: int = 1) -> State:
        dt = self.dt
        acc_xi, acc_a = self.accelerations(s.xi, s.a)
        for _ in range(n):
            s.xi_t += 0.5 * dt * acc_xi
            s.a_t += 0.5 * dt * acc_a
            s.xi += dt * s.xi_t
            s.a += dt * s.a_t
            acc_xi, acc_a = self.accelerations(s.xi, s.a)
            s.xi_t += 0.5 * dt * acc_xi
            s.a_t += 0.5 * dt * acc_a
            s.time += dt
        return s

    # ---- diagnostics ------------------------------------------------------
    def energy_density(self, s: State, part: str = "total"):
        """Discrete energy per cell boundary-to-boundary link.

        Gradient energy uses forward differences, which is the discrete
        energy conserved by the centred Laplacian. Returns an array of
        length len(x) - 1 (one value per link, averaged onto links).
        """
        p = self.p
        gx_xi = np.diff(s.xi) / self.dx
        gx_a = np.diff(s.a) / self.dx
        mid = lambda f: 0.5 * (f[1:] + f[:-1])
        e_a = (mid(0.5 * s.a_t**2) + 0.5 * gx_a**2
               + mid(0.5 * p.g**2 * s.xi**2 * s.a**2))
        e_xi = (mid(0.5 * s.xi_t**2) + 0.5 * gx_xi**2
                + mid(0.25 * p.lam * (s.xi**2 - p.v**2) ** 2))
        return {"a": e_a, "xi": e_xi, "total": e_a + e_xi}[part]

    def energy(self, s: State, part: str = "total", mask=None):
        e = self.energy_density(s, part)
        if mask is not None:
            e = e[mask[:-1]]
        return float(np.sum(e) * self.dx)


# --------------------------------------------------------------------------
# Initial data
# --------------------------------------------------------------------------
def vacuum(x, params: Params, sign: float = 1.0) -> State:
    z = np.zeros_like(x)
    return State(x, sign * params.v + z, z.copy(), z.copy(), z.copy())


def kink(x, params: Params) -> State:
    z = np.zeros_like(x)
    return State(x, params.v * np.tanh(x / params.delta), z.copy(), z.copy(), z.copy())


def add_wave_packet(s: State, amplitude, x0, k0, sigma, mass):
    """Add a right-moving packet of the a field with dispersion w^2 = k^2 + mass^2.

    The packet is built in Fourier space so that each mode moves to the right
    with its own frequency. `mass` is the photon mass where the packet starts.
    """
    x = s.x
    dx = x[1] - x[0]
    psi = amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * (x - x0))
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)
    psi_k = np.fft.fft(psi)
    psi_k[k <= 0] = 0.0                      # keep right-movers only
    w = np.sqrt(k**2 + mass**2)
    s.a = s.a + np.fft.ifft(psi_k).real
    s.a_t = s.a_t + np.fft.ifft(-1j * w * psi_k).real
    weight = (w**2) * np.abs(psi_k) ** 2     # energy per mode
    return k[k > 0], weight[k > 0]
