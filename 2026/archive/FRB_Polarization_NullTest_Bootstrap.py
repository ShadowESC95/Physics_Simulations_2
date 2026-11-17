#!/usr/bin/env python3
"""
FRB Polarization Null-Test with Bootstrap Confidence Bands
---------------------------------------------------------
This script ingests an FRB burst table and computes the
polarization-differential residual

    D(nu) = Δt_X(nu) - Δt_Y(nu)

after standard plasma dispersion removal, and estimates
bootstrap confidence intervals by resampling over bursts.

Purpose:
    Decisive falsification test for polarization-selective
    vacuum propagation effects.

Input CSV (minimum columns):
    burst_id        : unique burst identifier
    freq_MHz        : observing frequency [MHz]
    time_ms_X       : arrival time, polarization X [ms]
    time_ms_Y       : arrival time, polarization Y [ms]
    dm_pc_cm3       : dispersion measure [pc cm^-3]

Dependencies:
    numpy, pandas, matplotlib

Author:
    Generated for Xi–Chi observer-algebra falsification test
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Physical constant (standard cold plasma dispersion)
# -------------------------------------------------
K_DM = 4.148808e3  # ms * MHz^2 / (pc cm^-3)

# -------------------------------------------------
# Data loading
# -------------------------------------------------
def load_burst_table(csv_path):
    df = pd.read_csv(csv_path)
    required = {
        "burst_id",
        "freq_MHz",
        "time_ms_X",
        "time_ms_Y",
        "dm_pc_cm3",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    return df

# -------------------------------------------------
# Plasma dispersion delay
# -------------------------------------------------
def plasma_delay(freq_MHz, dm):
    return K_DM * dm / (freq_MHz ** 2)

# -------------------------------------------------
# Compute polarization residual D(nu)
# -------------------------------------------------
def compute_residuals(df):
    delay = plasma_delay(df["freq_MHz"].values,
                          df["dm_pc_cm3"].values)
    res_X = df["time_ms_X"].values - delay
    res_Y = df["time_ms_Y"].values - delay
    return res_X - res_Y

# -------------------------------------------------
# Bootstrap confidence bands (resample bursts)
# -------------------------------------------------
def bootstrap_bands(df, n_boot=2000, conf=(0.68, 0.95)):
    freqs = np.sort(df["freq_MHz"].unique())
    mean_D = []
    bands = {c: [] for c in conf}

    for nu in freqs:
        sub = df[df["freq_MHz"] == nu]
        burst_ids = sub["burst_id"].unique()

        # Mean residual at this frequency
        D0 = compute_residuals(sub)
        mean_D.append(np.mean(D0))

        boot_means = []
        for _ in range(n_boot):
            sample_ids = np.random.choice(
                burst_ids, size=len(burst_ids), replace=True
            )
            sample = pd.concat(
                [sub[sub["burst_id"] == i] for i in sample_ids]
            )
            boot_means.append(np.mean(compute_residuals(sample)))

        boot_means = np.asarray(boot_means)

        for c in conf:
            alpha = (1 - c) / 2
            lo = np.quantile(boot_means, alpha)
            hi = np.quantile(boot_means, 1 - alpha)
            bands[c].append((lo, hi))

    return freqs, np.asarray(mean_D), bands

# -------------------------------------------------
# Plotting
# -------------------------------------------------
def plot_result(freqs, mean_D, bands):
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, mean_D, lw=2,
             label=r"$\overline{\mathcal{D}}(\nu)$")

    for c, vals in bands.items():
        lo = [v[0] for v in vals]
        hi = [v[1] for v in vals]
        plt.fill_between(
            freqs, lo, hi, alpha=0.25,
            label=f"{int(c*100)}% CI"
        )

    plt.axhline(0.0, ls="--", lw=1, color="k")
    plt.xscale("log")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$\mathcal{D}(\nu)$  [ms]")
    plt.title(
        "Polarization-Differential FRB Residual\n"
        "Bootstrap Confidence Bands"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Main entry point
# -------------------------------------------------
def run_analysis(csv_path):
    df = load_burst_table(csv_path)
    freqs, mean_D, bands = bootstrap_bands(df)
    plot_result(freqs, mean_D, bands)

# Example usage:
# run_analysis("frb_burst_table.csv")
