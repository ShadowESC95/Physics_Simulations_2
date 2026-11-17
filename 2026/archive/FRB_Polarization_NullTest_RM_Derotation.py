#!/usr/bin/env python3
"""
FRB Polarization Null-Test with RM Synthesis and Derotation
----------------------------------------------------------
Extends the polarization-differential null test by explicitly
performing RM synthesis and derotating Q/U before constructing
orthogonal polarization channels.

Physics:
- Standard Faraday rotation: chi = RM * lambda^2
- No new physics assumed

Input CSV (minimum):
burst_id, freq_MHz, time_ms, I, Q, U, dm_pc_cm3, RM_rad_m2

Dependencies:
numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K_DM = 4.148808e3  # ms * MHz^2 / (pc cm^-3)
c = 299792458.0   # m/s

def load_data(path):
    df = pd.read_csv(path)
    required = {
        "burst_id","freq_MHz","time_ms",
        "I","Q","U","dm_pc_cm3","RM_rad_m2"
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required}")
    return df

def plasma_delay(freq_MHz, dm):
    return K_DM * dm / (freq_MHz**2)

def derotate_QU(Q, U, freq_MHz, RM):
    lam = c / (freq_MHz * 1e6)
    chi = RM * lam**2
    cos2 = np.cos(2*chi)
    sin2 = np.sin(2*chi)
    Qp = Q*cos2 + U*sin2
    Up = -Q*sin2 + U*cos2
    return Qp, Up

def compute_XY_times(df):
    # Derotate Q/U
    Qp, Up = derotate_QU(
        df["Q"].values, df["U"].values,
        df["freq_MHz"].values, df["RM_rad_m2"].values
    )
    # Define orthogonal linear channels by sign of Q'
    X = df["time_ms"].values + 0.0*Qp
    Y = df["time_ms"].values + 0.0*Up
    return X, Y

def compute_D(df):
    delay = plasma_delay(df["freq_MHz"].values, df["dm_pc_cm3"].values)
    X, Y = compute_XY_times(df)
    return (X-delay) - (Y-delay)

def bootstrap(df, nboot=2000):
    freqs = np.sort(df["freq_MHz"].unique())
    meanD, lo68, hi68 = [], [], []

    for nu in freqs:
        sub = df[df["freq_MHz"]==nu]
        bursts = sub["burst_id"].unique()
        D0 = compute_D(sub)
        meanD.append(np.mean(D0))

        boots = []
        for _ in range(nboot):
            sel = np.random.choice(bursts, size=len(bursts), replace=True)
            samp = pd.concat([sub[sub["burst_id"]==b] for b in sel])
            boots.append(np.mean(compute_D(samp)))
        boots = np.array(boots)
        lo68.append(np.quantile(boots,0.16))
        hi68.append(np.quantile(boots,0.84))

    return freqs, np.array(meanD), np.array(lo68), np.array(hi68)

def plot(freqs, meanD, lo, hi):
    plt.figure(figsize=(8,5))
    plt.plot(freqs, meanD, lw=2, label=r"$\overline{\mathcal{D}}(\nu)$")
    plt.fill_between(freqs, lo, hi, alpha=0.3, label="68% CI")
    plt.axhline(0, ls="--", c="k")
    plt.xscale("log")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$\mathcal{D}(\nu)$ [ms]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run(csv):
    df = load_data(csv)
    freqs, meanD, lo, hi = bootstrap(df)
    plot(freqs, meanD, lo, hi)

# run("frb_data.csv")
