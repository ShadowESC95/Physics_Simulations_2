#!/usr/bin/env python3
"""
FRB Polarization Null-Test with Redshift Binning and Geometric Scaling
--------------------------------------------------------------------
Adds redshift binning and tests whether polarization-differential
residuals scale with cosmological distance as expected for a
propagation effect.

Input CSV adds:
    z  : source redshift

Dependencies:
numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

H0 = 70.0  # km/s/Mpc
c = 299792.458  # km/s

def comoving_distance(z, Om=0.3):
    # Flat LCDM, analytic integral (simple quadrature)
    zs = np.linspace(0, z, 500)
    Ez = np.sqrt(Om*(1+zs)**3 + (1-Om))
    return c/H0 * np.trapz(1/Ez, zs)  # Mpc

def load_data(path):
    df = pd.read_csv(path)
    if "z" not in df.columns:
        raise ValueError("Redshift column z required")
    return df

def bin_by_redshift(df, zbins):
    df = df.copy()
    df["zbin"] = pd.cut(df["z"], zbins)
    return df

def analyze(df):
    results = []
    for zb, sub in df.groupby("zbin"):
        Dbar = sub["D_res"].mean()
        Dc = np.mean([comoving_distance(z) for z in sub["z"]])
        results.append((zb, Dc, Dbar))
    return results

def plot(results):
    Dc = [r[1] for r in results]
    Dbar = [r[2] for r in results]
    plt.figure(figsize=(6,4))
    plt.plot(Dc, Dbar, "o-")
    plt.xlabel("Mean Comoving Distance (Mpc)")
    plt.ylabel(r"$\langle \mathcal{D} \rangle$ [ms]")
    plt.tight_layout()
    plt.show()

# Expected usage:
# df = load_data("frb_with_D_and_z.csv")
# df = bin_by_redshift(df, [0,0.3,0.6,1.0,2.0])
# results = analyze(df)
# plot(results)
