#!/usr/bin/env python3
"""
Proca Wall Solver - interactive desktop app.

    python3 proca_wall_app.py              open the window
    python3 proca_wall_app.py --selftest   render one frame to PNG without a display

Solves the scattering of a massive (Stueckelberg/Proca) photon off a planar wall
where the photon mass m(z) changes, and shows how much power ends up in the
transverse and longitudinal polarisations on each side.  All physics lives in
proca_wall_solver.py; this file is only the window.

Units: c = hbar = 1.  Masses, frequency and 1/length share one arbitrary unit.

Needs: numpy, scipy, matplotlib, tkinter (python3-tk).  Runs fully offline.
"""
import sys
import csv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from proca_wall_solver import scatter_fast, scan_fast, sharp_wall_conversion, wall_ends

HBAR_EV_S = 6.582119569e-16
C_M_S = 299792458.0

# Fixed colour per output channel (colour follows the quantity, never its rank)
COL = dict(TT="#2a78d6", TL="#eb6834", RT="#1baf7a", RL="#eda100",
           E="#6250d6", Pi="#d55181", ink="#2b2b28", muted="#7a7972", grid="#e1e0d9")
OUT = [("TT", "transmitted, transverse"), ("TL", "transmitted, longitudinal"),
       ("RT", "reflected, transverse"), ("RL", "reflected, longitudinal")]
PROFILES = {"Mass step": "step", "Kink (mass crosses zero)": "kink", "Barrier / well": "bump"}
SCANS = ["Incidence angle", "Frequency", "Wall thickness"]

DEFAULTS = dict(profile="Mass step", mL=0.5, mR=1.0, extra=0.8, delta=1.0, omega=3.0,
                theta=50.0, incident=0, scan="Incidence angle", logy=True)


# ----------------------------------------------------------------------------
# Computation (no GUI code below this line until the App class)
# ----------------------------------------------------------------------------
def wall_params(st):
    kind = PROFILES[st["profile"]]
    if kind == "step":
        return kind, st["mL"], st["mR"]
    if kind == "kink":
        return kind, st["mL"], 0.0
    return kind, st["mL"], max(st["extra"], -0.9 * st["mL"])


def outputs(res, j, i=None):
    """The four outgoing powers for incident polarisation j (0 transverse, 1 longitudinal)."""
    sl = (slice(None),) if i is None else (i,)
    return dict(TT=res["T"][(0, j) + sl], TL=res["T"][(1, j) + sl],
                RT=res["R"][(0, j) + sl], RL=res["R"][(1, j) + sl])


def solve_point(st):
    kind, p1, p2 = wall_params(st)
    mL, _ = wall_ends(kind, p1, p2)
    if st["omega"] <= abs(mL):
        return None
    th = np.radians(st["theta"])
    res = scatter_fast(kind, p1, p2, st["delta"], st["omega"], th, ppw=60,
                       max_steps=150000, profile=True)
    res["sharp"] = sharp_wall_conversion(kind, p1, p2, st["omega"], th)
    return res


def solve_scan(st, n=60):
    kind, p1, p2 = wall_params(st)
    mL, _ = wall_ends(kind, p1, p2)
    th = np.radians(st["theta"])
    if st["scan"] == "Incidence angle":
        x = np.linspace(0.5, 89.0, n)
        res = scan_fast(kind, p1, p2, st["delta"], st["omega"], np.radians(x))
        meta = dict(label="incidence angle (degrees)", log=False, mark=st["theta"])
    elif st["scan"] == "Frequency":
        x = np.geomspace(1.02 * abs(mL), 30.0, n)
        res = scan_fast(kind, p1, p2, st["delta"], x, th)
        meta = dict(label="frequency  omega", log=True, mark=st["omega"])
    else:
        x = np.geomspace(0.02, 20.0, 48)
        res = scan_fast(kind, p1, p2, x, st["omega"], th)
        meta = dict(label="wall thickness  delta", log=True, mark=st["delta"])
    return dict(x=x, res=res, **meta)


def frb_estimate(dm_ev, f_ghz, theta_deg):
    """Thick-wall T->L conversion per wall crossing in SI terms."""
    omega = 2 * np.pi * f_ghz * 1e9
    ratio = (dm_ev / HBAR_EV_S) / omega
    t = np.tan(np.radians(theta_deg))
    return dict(ratio=ratio, P=(t * ratio) ** 2, wavelength=C_M_S / (f_ghz * 1e9))


def fmt(v):
    if v is None or not np.isfinite(v):
        return "   -   "
    return f"{v:.5f}" if v >= 1e-3 else f"{v:.2e}"


def readout(st, res):
    if res is None:
        return ("The incident wave cannot propagate:\nthe frequency must exceed the mass on\n"
                "the left side.  Raise omega or lower m.")
    j = st["incident"]
    o = outputs(res, j, 0)
    name = "transverse" if j == 0 else "longitudinal"
    conv = res["T"][1 - j, j, 0]
    lines = [f"Incident: {name} TM photon", ""]
    lines += [f"  {lab:<26s} {fmt(o[k])}" for k, lab in OUT]
    lines += ["", f"  flux closure error         {res['closure'][0]:.1e}",
              f"  grid steps                 {res['steps']}", "",
              "Converted power (transmitted):",
              f"  this wall                  {fmt(conv)}",
              f"  thin-wall limit            {fmt(res['sharp'])}",
              f"  thick-wall limit           {fmt(res['thick'][0])}", "",
              f"TE photon:  R = {fmt(res['R_TE'][0])}   T = {fmt(res['T_TE'][0])}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def build_figure():
    fig = Figure(figsize=(9.6, 6.6), dpi=100, layout="constrained")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 2.4, 2.4], width_ratios=[1.0, 1.15])
    ax = dict(mass=fig.add_subplot(gs[0, 0]), bar=fig.add_subplot(gs[0, 1]))
    ax["field"] = fig.add_subplot(gs[1, 0], sharex=ax["mass"])
    ax["scan"] = fig.add_subplot(gs[1:, 1])
    ax["te"] = fig.add_subplot(gs[2, 0])
    return fig, ax


def _style(a, title):
    a.set_title(title, loc="left", fontsize=9.5, color=COL["ink"], pad=5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(COL["grid"])
    a.tick_params(colors=COL["muted"], labelsize=8, length=3)
    a.grid(True, color=COL["grid"], linewidth=0.6)
    a.set_axisbelow(True)


def draw(ax, st, res, scan):
    for a in ax.values():
        a.clear()
    j = st["incident"]

    # --- mass profile -------------------------------------------------------
    a = ax["mass"]
    _style(a, "Photon mass across the wall,  m(z)")
    if res is not None:
        a.plot(res["z"], res["m"], color=COL["ink"], lw=1.6)
        a.fill_between(res["z"], 0, res["m"], color=COL["ink"], alpha=0.08, lw=0)
        a.axhline(0, color=COL["muted"], lw=0.6)
    a.tick_params(labelbottom=False)

    # --- channel amplitudes ---------------------------------------------------
    a = ax["field"]
    _style(a, "The two independent wave channels")
    if res is not None:
        a.plot(res["z"], res["absE"], color=COL["E"], lw=1.6, label="|E_x|  (E channel)")
        a.plot(res["z"], res["absPi"], color=COL["Pi"], lw=1.6, label="|Pi| = |m^2 A_z|  (Pi channel)")
        a.set_ylim(bottom=0)
        a.legend(fontsize=8, frameon=False, loc="best")
    else:
        a.text(0.5, 0.5, "no propagating incident wave", ha="center", va="center",
               transform=a.transAxes, color=COL["muted"])
    a.set_xlabel("z   (wave arrives from the left; incident amplitude = 1)", fontsize=8.5, color=COL["muted"])

    # --- power budget ---------------------------------------------------------
    a = ax["bar"]
    _style(a, "Where the incident power goes")
    a.grid(False)
    if res is not None:
        o = outputs(res, j, 0)
        left = 0.0
        for k, lab in OUT:
            v = float(o[k])
            a.barh(0, v, left=left, color=COL[k], edgecolor="white", linewidth=1.5, height=0.55)
            if v > 0.07:
                a.text(left + v / 2, 0, f"{100 * v:.1f}%", ha="center", va="center",
                       fontsize=8.5, color="#0b0b0b" if k == "RL" else "white")
            left += v
        handles = [a.barh(0, 0, color=COL[k]) for k, _ in OUT]
        a.legend(handles, [f"{lab}  {fmt(float(o[k]))}" for k, lab in OUT], fontsize=7.5,
                 frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                 handlelength=1.0, columnspacing=1.2)
    a.set_xlim(0, 1)
    a.set_ylim(-0.5, 0.5)
    a.set_yticks([])
    a.spines["left"].set_visible(False)

    # --- TE comparison --------------------------------------------------------
    a = ax["te"]
    _style(a, "Converted fraction against the two limits")
    a.grid(False)
    if res is not None:
        conv = float(res["T"][1 - j, j, 0])
        vals = [("thin-wall limit", res["sharp"]), ("this wall", conv), ("thick-wall limit", float(res["thick"][0]))]
        vals = [(n, v) for n, v in vals if v is not None and np.isfinite(v)]
        ys = np.arange(len(vals))[::-1]
        vmax = max(max(v for _, v in vals), 1e-30)
        for y, (n, v) in zip(ys, vals):
            a.barh(y, v, color=COL["TL"] if n == "this wall" else COL["muted"],
                   alpha=1.0 if n == "this wall" else 0.45, height=0.5)
            a.text(v + 0.02 * vmax, y, fmt(v), va="center", fontsize=8.5, color=COL["ink"])
        a.set_yticks(ys)
        a.set_yticklabels([n for n, _ in vals], fontsize=8.5, color=COL["ink"])
        a.set_xlim(0, vmax * 1.28)
        a.set_ylim(-0.6, len(vals) - 0.4)
    a.set_xlabel("power converted between transverse and longitudinal (transmitted)",
                 fontsize=8.5, color=COL["muted"])

    # --- scan ------------------------------------------------------------------
    a = ax["scan"]
    _style(a, f"Outgoing power against {scan['label'].split('  ')[0]}" if scan else "Scan")
    if scan is not None:
        o = outputs(scan["res"], j)
        floor = 1e-12
        for k, lab in OUT:
            y = np.asarray(o[k], float)
            a.plot(scan["x"], np.maximum(y, floor) if st["logy"] else y, color=COL[k], lw=1.8, label=lab)
        if PROFILES[st["profile"]] != "bump":
            y = scan["res"]["thick"]
            a.plot(scan["x"], np.maximum(y, floor) if st["logy"] else y, color=COL["ink"], lw=1.0,
                   ls=(0, (4, 3)), label="thick-wall formula (converted)")
        a.axvline(scan["mark"], color=COL["muted"], lw=0.8, ls=":")
        if scan["log"]:
            a.set_xscale("log")
        if st["logy"]:
            a.set_yscale("log")
            lo = np.nanmin([np.nanmin(np.maximum(v, floor)) for v in o.values()])
            a.set_ylim(max(floor, lo * 0.5), 2.0)
        else:
            a.set_ylim(-0.02, 1.02)
        a.set_xlabel(scan["label"], fontsize=8.5, color=COL["muted"])
        a.set_ylabel("fraction of incident power", fontsize=8.5, color=COL["muted"])
        a.legend(fontsize=8, frameon=False, loc="best")


def write_scan_csv(path, st, scan):
    o = outputs(scan["res"], st["incident"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# " + ", ".join(f"{k}={v}" for k, v in st.items())])
        w.writerow([scan["label"].split("  ")[0].replace(" ", "_")] + [lab.replace(", ", "_") for _, lab in OUT]
                   + ["thick_wall_formula", "flux_closure_error"])
        for i, x in enumerate(scan["x"]):
            w.writerow([f"{x:.8g}"] + [f"{o[k][i]:.8e}" for k, _ in OUT]
                       + [f"{scan['res']['thick'][i]:.8e}", f"{scan['res']['closure'][i]:.2e}"])


# ----------------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------------
def run_app(smoke=False):
    import tkinter as tk
    from tkinter import ttk, filedialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title("Proca Wall Solver")
    root.geometry("1380x800")
    root.minsize(1000, 620)
    st = dict(DEFAULTS)
    cache = dict(res=None, scan=None, jobs={})

    side = ttk.Frame(root, padding=10)
    side.pack(side="left", fill="y")
    main = ttk.Frame(root)
    main.pack(side="right", fill="both", expand=True)

    fig, ax = build_figure()
    canvas = FigureCanvasTkAgg(fig, master=main)
    NavigationToolbar2Tk(canvas, main, pack_toolbar=True)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    out = tk.Text(side, width=44, height=15, font="TkFixedFont", relief="flat",
                  background=root.cget("background"), state="disabled")

    def redraw():
        draw(ax, st, cache["res"], cache["scan"])
        canvas.draw_idle()
        out.configure(state="normal")
        out.delete("1.0", "end")
        out.insert("1.0", readout(st, cache["res"]))
        out.configure(state="disabled")

    def do_point():
        cache["res"] = solve_point(st)
        redraw()

    def do_scan():
        cache["scan"] = solve_scan(st) if cache["res"] is not None else None
        status.set("")
        redraw()

    def schedule(*_):
        for name, (fn, delay) in dict(point=(do_point, 25), scan=(do_scan, 350)).items():
            if name in cache["jobs"]:
                root.after_cancel(cache["jobs"][name])
            cache["jobs"][name] = root.after(delay, fn)
        status.set("updating scan...")

    # ---- controls ----
    ttk.Label(side, text="Wall", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    prof = ttk.Combobox(side, values=list(PROFILES), state="readonly", width=30)
    prof.set(st["profile"])
    prof.pack(anchor="w", pady=(2, 6))

    sliders = {}

    def slider(key, label, lo, hi, log=False, fmt_="{:.3g}"):
        row = ttk.Frame(side)
        row.pack(fill="x", pady=1)
        name = tk.StringVar(value=label)
        ttk.Label(row, textvariable=name, width=24).pack(side="left")
        val = tk.StringVar()
        to = (lambda v: np.log10(v)) if log else (lambda v: v)
        frm = (lambda s: 10 ** s) if log else (lambda s: s)
        var = tk.DoubleVar(value=to(st[key]))

        def changed(_=None):
            st[key] = float(frm(var.get()))
            val.set(fmt_.format(st[key]))
            schedule()

        sc = ttk.Scale(row, from_=to(lo), to=to(hi), variable=var, command=changed, length=150)
        sc.pack(side="left", padx=4)
        ttk.Label(row, textvariable=val, width=7).pack(side="left")
        val.set(fmt_.format(st[key]))
        sliders[key] = dict(scale=sc, var=var, val=val, name=name, to=to, fmt=fmt_)

    slider("mL", "Mass on the left  m_L", 0.05, 3.0)
    slider("mR", "Mass on the right  m_R", 0.05, 3.0)
    slider("extra", "Extra mass at centre", -2.5, 3.0)
    slider("delta", "Wall thickness  delta", 0.02, 20.0, log=True)

    ttk.Label(side, text="Incoming photon", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(10, 0))
    slider("omega", "Frequency  omega", 0.1, 30.0, log=True)
    slider("theta", "Incidence angle (deg)", 0.0, 89.0, fmt_="{:.1f}")
    inc = tk.IntVar(value=st["incident"])
    row = ttk.Frame(side)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Polarisation", width=24).pack(side="left")

    def set_inc():
        st["incident"] = inc.get()
        schedule()

    ttk.Radiobutton(row, text="transverse", variable=inc, value=0, command=set_inc).pack(side="left")
    ttk.Radiobutton(row, text="longitudinal", variable=inc, value=1, command=set_inc).pack(side="left")

    ttk.Label(side, text="Scan", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(10, 0))
    row = ttk.Frame(side)
    row.pack(fill="x", pady=2)
    scan_box = ttk.Combobox(row, values=SCANS, state="readonly", width=18)
    scan_box.set(st["scan"])
    scan_box.pack(side="left")
    logy = tk.BooleanVar(value=st["logy"])

    def set_logy():
        st["logy"] = logy.get()
        redraw()

    ttk.Checkbutton(row, text="log scale", variable=logy, command=set_logy).pack(side="left", padx=8)

    def set_scan(_=None):
        st["scan"] = scan_box.get()
        schedule()

    scan_box.bind("<<ComboboxSelected>>", set_scan)

    def set_profile(_=None):
        st["profile"] = prof.get()
        kind = PROFILES[st["profile"]]
        sliders["mL"]["name"].set("Mass on the left  m_L" if kind == "step" else "Mass far from wall  m0")
        sliders["mR"]["scale"].state(["!disabled"] if kind == "step" else ["disabled"])
        sliders["extra"]["scale"].state(["!disabled"] if kind == "bump" else ["disabled"])
        schedule()

    prof.bind("<<ComboboxSelected>>", set_profile)

    ttk.Separator(side).pack(fill="x", pady=8)
    out.pack(anchor="w")

    # ---- physical units ----
    ttk.Separator(side).pack(fill="x", pady=8)
    ttk.Label(side, text="In physical units (thick wall, per crossing)",
              font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    row = ttk.Frame(side)
    row.pack(fill="x", pady=2)
    ent = {}
    for key, label, default in (("dm", "mass change (eV)", "1e-14"), ("f", "freq (GHz)", "1.0"), ("th", "angle (deg)", "45")):
        ttk.Label(row, text=label).pack(side="left")
        e = ttk.Entry(row, width=7)
        e.insert(0, default)
        e.pack(side="left", padx=(2, 8))
        ent[key] = e
    frb = tk.StringVar()
    ttk.Label(side, textvariable=frb, font="TkFixedFont", justify="left").pack(anchor="w")

    def update_frb(*_):
        try:
            r = frb_estimate(float(ent["dm"].get()), float(ent["f"].get()), float(ent["th"].get()))
            frb.set(f"m c^2 / (hbar omega) = {r['ratio']:.2e}\n"
                    f"converted fraction   = {r['P']:.2e}\n"
                    f"valid for walls thicker than {r['wavelength']:.3g} m")
        except ValueError:
            frb.set("enter numbers in all three boxes")

    for e in ent.values():
        e.bind("<KeyRelease>", update_frb)
    update_frb()

    # ---- buttons ----
    ttk.Separator(side).pack(fill="x", pady=8)
    row = ttk.Frame(side)
    row.pack(fill="x")

    def save_fig():
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            fig.savefig(path, dpi=200)

    def save_csv():
        if cache["scan"] is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            write_scan_csv(path, st, cache["scan"])

    def reset():
        st.update(DEFAULTS)
        for key, s in sliders.items():
            s["var"].set(s["to"](st[key]))
            s["val"].set(s["fmt"].format(st[key]))
        prof.set(st["profile"])
        scan_box.set(st["scan"])
        inc.set(st["incident"])
        logy.set(st["logy"])
        set_profile()

    ttk.Button(row, text="Save figure", command=save_fig).pack(side="left")
    ttk.Button(row, text="Export scan CSV", command=save_csv).pack(side="left", padx=6)
    ttk.Button(row, text="Reset", command=reset).pack(side="left")
    status = tk.StringVar()
    ttk.Label(side, textvariable=status, foreground="#7a7972").pack(anchor="w", pady=(6, 0))

    set_profile()
    if smoke:                       # build everything, run both solves, close
        root.withdraw()
        do_point()
        do_scan()
        for name in SCANS[1:]:
            st["scan"] = name
            do_scan()
        for name in list(PROFILES)[1:]:
            st["profile"] = name
            do_point()
            do_scan()
        root.update()
        root.destroy()
        print("window smoke test ok")
        return
    root.mainloop()


def selftest(path="proca_wall_selftest.png"):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    st = dict(DEFAULTS)
    fig, ax = build_figure()
    FigureCanvasAgg(fig)
    res = solve_point(st)
    draw(ax, st, res, solve_scan(st))
    fig.savefig(path, dpi=110)
    print(readout(st, res))
    print("wrote", path)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(*[a for a in sys.argv[1:] if not a.startswith("--")][:1])
    else:
        run_app(smoke="--smoke" in sys.argv)
