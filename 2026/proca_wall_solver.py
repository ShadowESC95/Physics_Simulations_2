#!/usr/bin/env python3
"""
Proca / Stueckelberg photon scattering through a planar mass wall m(z).

Replaces S8_7_TM_Scattering_Solver.py and S10_1_Parameter_Scan_TM_Longitudinal.py.

Model (c = hbar = 1, metric +---):
    L = -1/4 F^2 + 1/2 m(z)^2 (A_mu - d_mu theta)^2
Unitary gauge gives the Proca equations with a position-dependent mass:
    d_mu F^{mu nu} + m^2 A^nu = 0,   d_nu (m^2 A^nu) = 0.

Fields ~ exp(i(k x - omega t)), z-dependence to be solved.  Write
    q(z)^2 = omega^2 - k^2 - m(z)^2.

Exact result used here
----------------------
The three polarisations reduce to three *decoupled* scalar equations in
gauge-invariant variables:

    TE :   E_y'' + q^2 E_y = 0
    TM :   E_x'' + q^2 E_x = 0                       ("E channel")
           m^2 (Pi'/m^2)' + q^2 Pi = 0,  Pi = m^2 A_z ("Pi channel")

(E_x obeys the plain Klein-Gordon operator because (box + m^2) F^{tx} picks up
no gradient-of-m term; the Pi equation is the z-component of the Proca
equation after eliminating div A with the constraint.)

The z-energy flux is the sum of two separately conserved channel fluxes:

    S_z = omega / (2 (omega^2 - k^2)) * [ Im(E_x* E_x') + Im(Pi* Pi') / m^2 ].

Transverse-TM (T) and longitudinal (L) photons are *mixtures* of the two
channels, with a mixing angle  tan(alpha) = k m / (omega q)  that depends on
the local mass.  T <-> L conversion is therefore the mismatch of the two
channel responses plus the change of alpha across the wall.

Numerics
--------
The Pi equation has a regular singular point where m = 0 (kink walls).  To
stay regular we integrate the equivalent 't Hooft-Feynman-gauge pair

    A_z'' + q^2 A_z            = -2 m' chi
    chi'' + (q^2 - m''/m) chi  = -2 m' A_z

and form the gauge-invariant  Pi = m^2 A_z + m chi' - m' chi
(with Pi' = m^2 A_z' - m q^2 chi).
m''/m is finite for a tanh kink (-2 sech^2(z/d)/d^2), as in an Abelian-Higgs
wall where it equals V'(|Phi|^2).

Dependencies: numpy, scipy
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------
# Wall profiles: each returns m, m', and U = m''/m
# --------------------------------------------------------------------------
@dataclass
class Step:
    """m(z) = mean + half * tanh(z/delta): mass step from mL (z<0) to mR (z>0)."""
    mL: float
    mR: float
    delta: float

    def m(self, z):
        return 0.5 * (self.mL + self.mR) + 0.5 * (self.mR - self.mL) * np.tanh(z / self.delta)

    def dm(self, z):
        return 0.5 * (self.mR - self.mL) / self.delta / np.cosh(z / self.delta) ** 2

    def U(self, z):
        t = np.tanh(z / self.delta)
        d2 = -(self.mR - self.mL) / self.delta ** 2 * t / np.cosh(z / self.delta) ** 2
        return d2 / self.m(z)

    @property
    def m_left(self):
        return self.mL

    @property
    def m_right(self):
        return self.mR


@dataclass
class Kink:
    """m(z) = m0 tanh(z/delta): passes through m = 0 at the wall centre."""
    m0: float
    delta: float

    def m(self, z):
        return self.m0 * np.tanh(z / self.delta)

    def dm(self, z):
        return self.m0 / self.delta / np.cosh(z / self.delta) ** 2

    def U(self, z):
        return -2.0 / self.delta ** 2 / np.cosh(z / self.delta) ** 2

    @property
    def m_left(self):
        return -self.m0

    @property
    def m_right(self):
        return self.m0


@dataclass
class Bump:
    """m(z) = m0 + dm sech^2(z/delta): localised mass barrier (dm>0) or well (dm<0)."""
    m0: float
    dmass: float
    delta: float

    def m(self, z):
        return self.m0 + self.dmass / np.cosh(z / self.delta) ** 2

    def dm(self, z):
        return -2.0 * self.dmass / self.delta * np.tanh(z / self.delta) / np.cosh(z / self.delta) ** 2

    def U(self, z):
        s2 = 1.0 / np.cosh(z / self.delta) ** 2
        d2 = self.dmass / self.delta ** 2 * (4.0 * s2 - 6.0 * s2 ** 2)
        return d2 / self.m(z)

    @property
    def m_left(self):
        return self.m0

    @property
    def m_right(self):
        return self.m0


# --------------------------------------------------------------------------
# Channel solvers
# --------------------------------------------------------------------------
def _csqrt(x):
    """Square root with Im >= 0, so exp(i q z) decays for evanescent q."""
    r = np.sqrt(complex(x))
    return r if r.imag >= 0 else -r


def _split(f, df, q, z):
    """f = a e^{iqz} + b e^{-iqz}  ->  (a, b)."""
    a = 0.5 * (f + df / (1j * q)) * np.exp(-1j * q * z)
    b = 0.5 * (f - df / (1j * q)) * np.exp(1j * q * z)
    return a, b


def channel_coefficients(wall, omega, k, Z=None, rtol=1e-10, atol=1e-12):
    """
    Amplitude reflection/transmission (r, t) of the E channel and the Pi channel
    for a wave incident from z = -infinity.

    Returns dict with rE, tE, rPi, tPi, qL, qR.  The t's are the coefficient of
    exp(i qR z) on the right for unit incident amplitude of that variable.
    """
    if Z is None:
        Z = 14.0 * wall.delta
    mL, mR = wall.m_left, wall.m_right
    qL = _csqrt(omega ** 2 - k ** 2 - mL ** 2)
    qR = _csqrt(omega ** 2 - k ** 2 - mR ** 2)
    if abs(qL.imag) > 0:
        raise ValueError("incident wave is evanescent: need omega^2 > k^2 + mL^2")

    def q2(z):
        return omega ** 2 - k ** 2 - wall.m(z) ** 2

    # E channel:  y = (E, E')
    def rhsE(z, y):
        return [y[1], -q2(z) * y[0]]

    e0 = np.exp(1j * qR * Z)
    sE = solve_ivp(rhsE, (Z, -Z), [e0, 1j * qR * e0], method="DOP853", rtol=rtol, atol=atol)
    aE, bE = _split(sE.y[0, -1], sE.y[1, -1], qL, -Z)

    # Pi channel via the regular gauge pair:  y = (Az, Az', chi, chi')
    def rhsP(z, y):
        Az, dAz, chi, dchi = y
        dm = wall.dm(z)
        return [dAz, -q2(z) * Az - 2 * dm * chi,
                dchi, -(q2(z) - wall.U(z)) * chi - 2 * dm * Az]

    sP = solve_ivp(rhsP, (Z, -Z), [e0, 1j * qR * e0, 0j, 0j],
                   method="DOP853", rtol=rtol, atol=atol)
    Az, dAz, chi, dchi = sP.y[:, -1]
    m, dm = wall.m(-Z), wall.dm(-Z)
    Pi = m * m * Az + m * dchi - dm * chi
    dPi = m * m * dAz - m * q2(-Z) * chi
    aP, bP = _split(Pi, dPi, qL, -Z)

    return dict(rE=bE / aE, tE=1.0 / aE, rPi=bP / aP, tPi=mR ** 2 / aP, qL=qL, qR=qR)


# --------------------------------------------------------------------------
# Polarisation basis  (a_T, a_L)  <->  channel amplitudes (E_x, Pi)
# --------------------------------------------------------------------------
def _mix(omega, k, m, kappa):
    """Columns: T and L unit-flux polarisations with z-wavenumber kappa."""
    K = np.sqrt(complex(k * k + kappa * kappa))
    return np.array([[1j * omega * kappa / K, 1j * k * m / K],
                     [-m * m * k / K, m * omega * kappa / K]], dtype=complex)


def scatter(wall, omega, theta, **kw):
    """
    Power coefficients for a photon of frequency omega hitting the wall at
    incidence angle theta (measured from the wall normal, in the left medium).

    Returns dict with
        R[i][j], T[i][j] : power from incident polarisation j into i,
                           indices 0 = transverse TM, 1 = longitudinal
        R_TE, T_TE       : the decoupled TE polarisation
        closure          : max |1 - sum of outgoing power| over the two inputs
    """
    mL, mR = wall.m_left, wall.m_right
    KL = np.sqrt(omega ** 2 - mL ** 2)
    k = KL * np.sin(theta)
    c = channel_coefficients(wall, omega, k, **kw)
    qL, qR = c["qL"], c["qR"]

    ML_in = _mix(omega, k, mL, qL)
    ML_out = _mix(omega, k, mL, -qL)
    refl = np.linalg.solve(ML_out, np.diag([c["rE"], c["rPi"]]) @ ML_in)
    R = np.abs(refl) ** 2

    if abs(qR.imag) < 1e-14 and qR.real > 0:
        MR = _mix(omega, k, mR, qR)
        tran = np.linalg.solve(MR, np.diag([c["tE"], c["tPi"]]) @ ML_in)
        T = np.abs(tran) ** 2 * (qR.real / qL.real)
        T_TE = abs(c["tE"]) ** 2 * (qR.real / qL.real)
    else:  # total reflection
        T = np.zeros((2, 2))
        T_TE = 0.0

    closure = np.max(np.abs(1.0 - R.sum(axis=0) - T.sum(axis=0)))
    return dict(R=R, T=T, R_TE=abs(c["rE"]) ** 2, T_TE=T_TE, closure=closure,
                k=k, qL=qL, qR=qR, channels=c)


# --------------------------------------------------------------------------
# Closed forms for checking
# --------------------------------------------------------------------------
def sharp_step_channels(omega, k, mL, mR):
    """delta -> 0 limit: Fresnel-type coefficients of the two channels."""
    qL = _csqrt(omega ** 2 - k ** 2 - mL ** 2)
    qR = _csqrt(omega ** 2 - k ** 2 - mR ** 2)
    rE = (qL - qR) / (qL + qR)
    tE = 2 * qL / (qL + qR)
    pL, pR = qL / mL ** 2, qR / mR ** 2
    rPi = (pL - pR) / (pL + pR)
    tPi = 2 * pL / (pL + pR)
    return dict(rE=rE, tE=tE, rPi=rPi, tPi=tPi, qL=qL, qR=qR)


def adiabatic_conversion(omega, theta, mL, mR):
    """
    Thick-wall (WKB) limit of the T -> L transmitted power for a mass step:
    the polarisation rotates by alpha_R - alpha_L, tan(alpha) = k m / (omega q).
    """
    k = np.sqrt(omega ** 2 - mL ** 2) * np.sin(theta)
    qL = np.sqrt(omega ** 2 - k ** 2 - mL ** 2)
    qR = np.sqrt(omega ** 2 - k ** 2 - mR ** 2)
    return np.sin(np.arctan2(k * mR, omega * qR) - np.arctan2(k * mL, omega * qL)) ** 2


# --------------------------------------------------------------------------
# Fast vectorised solver (used by the desktop app and for parameter scans)
# --------------------------------------------------------------------------
def _wall_arrays(kind, p1, p2, delta, z):
    """m, m', m''/m for arrays of wall parameters.  kind: 'step' | 'kink' | 'bump'."""
    t = np.tanh(z / delta)
    s2 = 1.0 - t * t
    if kind == "kink":                       # p1 = m0
        return p1 * t, p1 / delta * s2, -2.0 / delta ** 2 * s2
    if kind == "bump":                       # p1 = m0, p2 = extra mass at centre
        m = p1 + p2 * s2
        return m, -2.0 * p2 / delta * t * s2, p2 / delta ** 2 * (4 * s2 - 6 * s2 * s2) / m
    a, b = 0.5 * (p1 + p2), 0.5 * (p2 - p1)  # step: p1 = mL, p2 = mR
    m = a + b * t
    return m, b / delta * s2, -2.0 * b / delta ** 2 * t * s2 / m


def wall_ends(kind, p1, p2):
    """Asymptotic (signed) masses on the left and right."""
    if kind == "kink":
        return -p1, p1
    if kind == "bump":
        return p1, p1
    return p1, p2


def _reduce(M):
    """Ordered product M[N-1] ... M[1] M[0] along axis 0 by pairwise multiplication."""
    while M.shape[0] > 1:
        n = M.shape[0]
        head = M[1:n - n % 2 + 1:2] @ M[0:n - n % 2:2]
        M = np.concatenate([head, M[-1:]]) if n % 2 else head
    return M[0]


def scatter_fast(kind, p1, p2, delta, omega, theta, ppw=60, span=12.0,
                 max_steps=40000, profile=False):
    """
    Same physics as scatter(), using fixed-step RK4 transfer matrices.  Every
    argument except `kind` may be an array; they are broadcast together, so one
    call evaluates a whole parameter scan.

    Returns dict of arrays: R, T with shape (2, 2, P) [out, in, point],
    R_TE, T_TE, closure, thick (adiabatic T<->L prediction), valid (bool).
    With profile=True (single point only) also z, m, absE, absPi normalised to
    the incident amplitude of each channel.
    """
    p1, p2, delta, omega, theta = [np.atleast_1d(np.asarray(v, float)) for v in
                                   np.broadcast_arrays(p1, p2, delta, omega, theta)]
    P = p1.size
    mL, mR = wall_ends(kind, p1, p2)
    valid = omega > np.abs(mL)
    om = np.where(valid, omega, np.abs(mL) + 1.0)        # placeholder where invalid
    k = np.sqrt(om ** 2 - mL ** 2) * np.sin(theta)
    w2 = om ** 2 - k ** 2
    qL = np.sqrt(w2 - mL ** 2)
    qR2 = w2 - mR ** 2
    qR = np.where(qR2 >= 0, np.sqrt(np.abs(qR2)), 1j * np.sqrt(np.abs(qR2)))

    Z = span * delta
    h0 = np.minimum(delta / 40.0, 2 * np.pi / np.sqrt(w2) / ppw)
    N = int(np.clip(np.ceil(np.max(2 * Z / h0)), 400, max_steps))
    h = (-2 * Z / N)[None, :]                           # integrate from +Z to -Z
    n = np.arange(N)[:, None]

    def A_at(z):
        m, dm, U = _wall_arrays(kind, p1, p2, delta, z)
        q2 = w2 - m * m
        AE = np.zeros(z.shape + (2, 2))
        AE[..., 0, 1] = 1.0
        AE[..., 1, 0] = -q2
        AP = np.zeros(z.shape + (4, 4))
        AP[..., 0, 1] = 1.0
        AP[..., 1, 0] = -q2
        AP[..., 1, 2] = -2 * dm
        AP[..., 2, 3] = 1.0
        AP[..., 3, 0] = -2 * dm
        AP[..., 3, 2] = -(q2 - U)
        return AE, AP

    z0 = Z[None, :] + n * h
    stages = [A_at(z0), A_at(z0 + 0.5 * h), A_at(z0 + h)]
    hh = h[..., None, None]

    def step_matrices(i):
        A0, Am, A1 = stages[0][i], stages[1][i], stages[2][i]
        I = np.eye(A0.shape[-1])
        K1 = A0
        K2 = Am @ (I + 0.5 * hh * K1)
        K3 = Am @ (I + 0.5 * hh * K2)
        K4 = A1 @ (I + hh * K3)
        return I + hh / 6.0 * (K1 + 2 * K2 + 2 * K3 + K4)

    SE, SP = step_matrices(0), step_matrices(1)          # shape (N, P, n, n)
    yE0 = np.stack([np.ones(P, complex), 1j * qR], axis=-1)[..., None]
    yP0 = np.stack([np.ones(P, complex), 1j * qR, np.zeros(P, complex), np.zeros(P, complex)],
                   axis=-1)[..., None]
    yE = (_reduce(SE) @ yE0)[..., 0]
    yP = (_reduce(SP) @ yP0)[..., 0]

    zl = -Z
    m, dm, _ = _wall_arrays(kind, p1, p2, delta, zl)
    Pi = m * m * yP[:, 0] + m * yP[:, 3] - dm * yP[:, 2]
    dPi = m * m * yP[:, 1] - m * (w2 - m * m) * yP[:, 2]

    def split(f, df):
        return (0.5 * (f + df / (1j * qL)) * np.exp(-1j * qL * zl),
                0.5 * (f - df / (1j * qL)) * np.exp(1j * qL * zl))

    aE, bE = split(yE[:, 0], yE[:, 1])
    aP, bP = split(Pi, dPi)
    rE, tE, rPi, tPi = bE / aE, 1.0 / aE, bP / aP, mR ** 2 / aP

    def mix(mass, kappa):
        K = np.sqrt(k * k + kappa * kappa + 0j)
        M = np.empty((P, 2, 2), complex)
        M[:, 0, 0] = 1j * om * kappa / K
        M[:, 0, 1] = 1j * k * mass / K
        M[:, 1, 0] = -mass * mass * k / K
        M[:, 1, 1] = mass * om * kappa / K
        return M

    Min = mix(mL, qL)
    D = np.zeros((P, 2, 2), complex)
    D[:, 0, 0], D[:, 1, 1] = rE, rPi
    R = np.abs(np.linalg.solve(mix(mL, -qL), D @ Min)) ** 2
    is_open = (qR.imag == 0) & (qR.real > 0)
    qRs = np.where(is_open, qR.real, 1.0)
    D[:, 0, 0], D[:, 1, 1] = tE, tPi
    flux = np.where(is_open, qRs / qL, 0.0)
    T = np.abs(np.linalg.solve(mix(mR, qRs), D @ Min)) ** 2 * flux[:, None, None]
    closure = np.max(np.abs(1.0 - R.sum(axis=1) - T.sum(axis=1)), axis=1)
    thick = np.where(is_open, np.sin(np.arctan2(k * mR, om * qRs) - np.arctan2(k * mL, om * qL)) ** 2, np.nan)

    bad = ~valid
    out = dict(R=np.where(bad[None, None, :], np.nan, np.moveaxis(R, 0, -1)),
               T=np.where(bad[None, None, :], np.nan, np.moveaxis(T, 0, -1)),
               R_TE=np.where(bad, np.nan, np.abs(rE) ** 2),
               T_TE=np.where(bad, np.nan, np.abs(tE) ** 2 * flux),
               closure=np.where(bad, np.nan, closure), thick=np.where(bad, np.nan, thick),
               valid=valid, open=is_open, steps=N, k=k, qL=qL, qR=qR)

    if profile:
        if P != 1:
            raise ValueError("profile=True needs a single parameter point")
        every = max(1, N // 500)
        ye, yp = yE0[0, :, 0].copy(), yP0[0, :, 0].copy()
        zs, eabs, pabs = [Z[0]], [abs(ye[0])], [abs(mR[0] ** 2)]
        for i in range(N):
            ye, yp = SE[i, 0] @ ye, SP[i, 0] @ yp
            if (i + 1) % every == 0 or i == N - 1:
                zi = Z[0] + (i + 1) * h[0, 0]
                mi, dmi, _ = _wall_arrays(kind, p1[0], p2[0], delta[0], zi)
                zs.append(zi)
                eabs.append(abs(ye[0]))
                pabs.append(abs(mi * mi * yp[0] + mi * yp[3] - dmi * yp[2]))
        zs = np.array(zs)[::-1]
        out.update(z=zs, m=_wall_arrays(kind, p1[0], p2[0], delta[0], zs)[0],
                   absE=np.array(eabs)[::-1] / abs(aE[0]), absPi=np.array(pabs)[::-1] / abs(aP[0]))
    return out


def scan_fast(kind, p1, p2, delta, omega, theta, ppw=40, budget=150000, max_steps=150000, **kw):
    """
    scatter_fast over a scan, with points grouped by how many steps they need so
    that cheap points are not integrated on the most expensive point's grid.
    Returns the same arrays as scatter_fast (without profiles).
    """
    p1, p2, delta, omega, theta = [np.atleast_1d(np.asarray(v, float)) for v in
                                   np.broadcast_arrays(p1, p2, delta, omega, theta)]
    P = p1.size
    need = np.maximum(40.0, np.maximum(omega, 1e-9) * delta * ppw / (2 * np.pi)) * 24.0
    order = np.argsort(need)
    keys = ("R", "T", "R_TE", "T_TE", "closure", "thick", "valid", "open")
    parts, i = [], 0
    while i < P:
        j = i + 1
        while j < P and need[order[j]] * (j + 1 - i) <= budget and need[order[j]] <= 2.0 * need[order[i]]:
            j += 1
        idx = order[i:j]
        parts.append((idx, scatter_fast(kind, p1[idx], p2[idx], delta[idx], omega[idx], theta[idx],
                                        ppw=ppw, max_steps=max_steps, **kw)))
        i = j
    out = {}
    for key in keys:
        sample = parts[0][1][key]
        arr = np.empty(sample.shape[:-1] + (P,), sample.dtype)
        for idx, res in parts:
            arr[..., idx] = res[key]
        out[key] = arr
    return out


def sharp_wall_conversion(kind, p1, p2, omega, theta):
    """Thin-wall limit of the transmitted T<->L power (zero unless the mass changes)."""
    mL, mR = wall_ends(kind, p1, p2)
    if kind != "step" or omega <= abs(mL):
        return 0.0
    k = np.sqrt(omega ** 2 - mL ** 2) * np.sin(theta)
    c = sharp_step_channels(omega, k, mL, mR)
    if abs(c["qR"].imag) > 0:
        return 0.0
    tran = np.linalg.solve(_mix(omega, k, mR, c["qR"]), np.diag([c["tE"], c["tPi"]]) @ _mix(omega, k, mL, c["qL"]))
    return float(abs(tran[1, 0]) ** 2 * (c["qR"].real / c["qL"].real))


if __name__ == "__main__":
    wall = Step(mL=0.5, mR=1.0, delta=1.0)
    out = scatter(wall, omega=3.0, theta=np.radians(50))
    np.set_printoptions(precision=6, suppress=True)
    print("Step wall mL=0.5 mR=1.0 delta=1, omega=3, theta=50 deg")
    print("  R (rows out T,L; cols in T,L):\n", out["R"])
    print("  T (rows out T,L; cols in T,L):\n", out["T"])
    print("  TE: R = %.6f  T = %.6f" % (out["R_TE"], out["T_TE"]))
    print("  flux closure error: %.2e" % out["closure"])
