# -*- coding: utf-8 -*-
"""
Ride-hailing entry and household vehicle ownership in U.S. metro counties, 2010-2017
====================================================================================

One script, run from top to bottom:

    python roc_pipeline.py                      # uses the paths in CONFIG below
    python roc_pipeline.py --data <csv> --out <folder> --boot 999

Outputs (created if missing):
    <out>/Figures/*.png    300 dpi
    <out>/Tables/*.tex     ready for \\input{Tables/...} in Overleaf
    <out>/Tables/all_estimates.csv   every estimate, for checking
    <out>/run_log.tex      the full console log as a compilable LaTeX file

File names follow the paper: T1-T4 main tables, A1-A3 appendix tables, F1-F2 main figures, FA1 appendix
figure. D1-D2 are documentation tables that are not in the paper.

Estimators (all written with numpy, no extra packages):
    * Callaway and Sant'Anna (2021) group-time ATTs, base year = year before entry, aggregated by event time
      with cohort-size weights (counties, or households for the household-weighted estimand).
      Comparison: counties not yet reached by ride-hailing. Metro areas without a date in Lee et al. (2022)
      stay in the comparison group only until their driver supply first rises (see below).
    * Extended two-way fixed effects (Wooldridge 2025): county and year effects estimated from untreated
      observations only, cohort x year effects after entry, optional 2006-2010 characteristics x year,
      time-varying controls, and treatment x characteristic interactions for heterogeneity.
    * Long difference over the non-overlapping ACS windows 2008-2012 and 2013-2017.
    * Static two-way fixed effects (for contrast only).
    Inference: MSA block bootstrap (Callaway-Sant'Anna) and MSA-clustered standard errors (regressions).

Entry dates for undated metro areas: the year in which MSA driver supply (nonemployer taxi and limousine
establishments per 10,000 residents, population-weighted over the MSA's counties) first rises by at least
DETECT_JUMP over the previous year and stays at least that much above it in every later year. The threshold
is chosen on the 184 dated metro areas (Table A3). Upstate New York is dated by law (29 June 2017).

Vehicle outcomes: ACS_vehicles_2010-2017.csv (written by get_acs_vehicles.py: tables B25044 and B25046) is read from
the folder of the county CSV, or from this script's folder, or from --veh. It gives the shares with 3 and with 4+
vehicles and the mean number of vehicles per household (aggregate vehicles / households). Without it the script
falls back to the capped index (three or more counted as three).

Requirements: Python 3.9+, numpy, pandas, scipy, matplotlib. Runs as a script or pasted into a Jupyter cell.

Version 4 (17 Oct 2026): extended TWFE covariates interacted with every post-entry cell (now equal to imputation);
leads for all pre-entry years; terciles on cohorts 2012-2014 through 2016; uncapped vehicle outcomes; benchmark
table removed; Table 3 is now the heterogeneity table.

Version 3 (17 Oct 2026), after an external audit of version 2:
    * comparison group cleaned: undated metro areas leave it in the year their driver supply rises
    * summary window 0-2 years after entry: in 2017 only 37 metro counties had no ride-hailing
    * extended TWFE replaces Sun and Abraham in the robustness table and gives the heterogeneity estimates,
      so that characteristic-specific trends are absorbed before effects are compared across counties
    * fixed: household-weighted Sun and Abraham aggregated cohorts by county counts (now household sums)
    * dropped: treated-only comparison (in 2016 its only comparison counties are one MSA) and the
      outcome-regression covariate design (extrapolates far outside the comparison counties' range);
      a common-support check takes their place
    * new: cohort-balanced event study, common-support subsample, count of bootstrap draws that lose a cell
"""
import argparse
import io
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- CONFIG
OUT_ROOT = r"C:\Users\L03565094\Dropbox\Francisco\Papers2023\Uber_Lyft_Ownership\04-Results"
# Put the CSV next to this script, or change this path, or pass --data
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:          # running inside Jupyter: use the notebook's folder
    HERE = os.getcwd()
DATA_PATH = os.path.join(HERE, "Counties_2010-2017_full_dataset_new.csv")
# ACS tables B25044 and B25046 (written by get_acs_vehicles.py); used when present, otherwise the capped index
VEH_NAME = "ACS_vehicles_2010-2017.csv"   # looked for next to the county CSV (e.g. 01-Data), then next to this script
N_BOOT = 999
SEED = 20261017
DETECT_JUMP = 1.0            # drivers per 10,000 residents; chosen on the dated metro areas (Table A3)
DETECT_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

YEARS = list(range(2010, 2018))
COHORTS = [2012, 2013, 2014, 2015, 2016]   # treated cohorts that have at least one post-entry year
EVENTS = list(range(-4, 4))                # event times reported; -1 is the reference year
POST = [0, 1, 2]                           # summary window
POST4 = [0, 1, 2, 3]
PRE = [-4, -3, -2]

OUTCOMES = {
    "drivers": "Taxi and limousine nonemployers (per 10,000 residents)",
    "s0": "Zero vehicles (% of households)",
    "s1": "One vehicle (% of households)",
    "s2": "Two vehicles (% of households)",
    "s3": "Three or more vehicles (% of households)",
    "s2p": "Two or more vehicles (% of households)",
    "veh": "Capped vehicle index (three or more counted as three)",
    "c_car": "Drove alone (% of workers)",
    "c_cp": "Carpooled (% of workers)",
    "c_pt": "Public transit (% of workers)",
}
VEH_OUT = {"s3x": "Three vehicles (% of households)", "s4p": "Four or more vehicles (% of households)",
           "vmean": "Vehicles per household (aggregate vehicles / households)"}
HAS_VEH = False
SHORT = {"drivers": "Taxi and limousine nonemployers", "s0": "Zero vehicles", "s1": "One vehicle", "s2": "Two vehicles",
         "s3": "Three or more", "s2p": "Two or more", "veh": "Capped vehicle index", "s3x": "Three vehicles",
         "s4p": "Four or more", "vmean": "Vehicles per household",
         "c_car": "Drove alone", "c_cp": "Carpooled", "c_pt": "Public transit"}
UNITS = {"drivers": "per 10,000 residents", "veh": "vehicles", "vmean": "vehicles", "c_car": "pp of workers",
         "c_cp": "pp of workers", "c_pt": "pp of workers"}

# colours for print figures: one accent, one quiet grey
BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#8c8b86", "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- logging
class Log:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.buf.write(s + "\n")

    def section(self, title):
        self("")
        self("=" * 100)
        self(title)
        self("=" * 100)

    def write_tex(self, path):
        txt = self.buf.getvalue().replace("\\end{verbatim}", "\\end{verbatim }")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\\documentclass[10pt]{article}\n\\usepackage[margin=1.2cm,landscape]{geometry}\n"
                    "\\usepackage[T1]{fontenc}\n\\begin{document}\n\\scriptsize\n\\begin{verbatim}\n")
            f.write(txt)
            f.write("\\end{verbatim}\n\\end{document}\n")


log = Log()

TV = {"ln_pop": "Log population", "ln_hh": "Log households", "ln_inc": "Log mean household income",
      "pp_lf": "Labor force participation (\\%)", "pp_poverty": "Poverty rate (\\%)",
      "pp_college": "College share (\\%)"}
RECOVERY = ["ln_pop", "ln_inc", "pp_lf", "pp_poverty"]


# ----------------------------------------------------------------------------- data
def load_panel(path, scope="metro", veh_path=None):
    """scope='metro': counties in metro areas (main sample). scope='all': every county; non-metro counties
    never get an entry date and form their own clusters."""
    df = pd.read_csv(path, dtype={"GEO.id2": str}, encoding="latin-1")
    m = (df[df.Metropolitan_Area.notna()] if scope == "metro" else df).copy()
    m["metro"] = m.Metropolitan_Area.notna()
    m["Non_emp_transp"] = m["Non_emp_transp"].fillna(0)
    m["drivers"] = m.Non_emp_transp / m.population * 10000
    hh = m.total_households
    m["s0"] = m.hou_no_car / hh * 100
    m["s1"] = m.hou_1car / hh * 100
    m["s2"] = m.hou_2cars / hh * 100
    m["s3"] = m.hou_3or_more_cars / hh * 100
    m["s2p"] = m.s2 + m.s3
    m["veh"] = (m.s1 + 2 * m.s2 + 3 * m.s3) / 100       # three or more counted as three (lower bound)
    global HAS_VEH
    outs = list(OUTCOMES)
    if veh_path and os.path.exists(veh_path):
        v = pd.read_csv(veh_path, dtype={"GEO.id2": str})
        v["GEO.id2"] = v["GEO.id2"].str.zfill(5)
        m = m.merge(v[["GEO.id2", "year", "hh_total", "hh_3", "hh_4", "hh_5p", "veh_aggregate"]],
                    on=["GEO.id2", "year"], how="left")
        m["s3x"] = m.hh_3 / m.hh_total * 100
        m["s4p"] = (m.hh_4 + m.hh_5p) / m.hh_total * 100
        m["vmean"] = m.veh_aggregate / m.hh_total
        outs += list(VEH_OUT)
        if scope == "metro":
            HAS_VEH = True
    m["G"] = m.Platform.fillna(0).astype(int)
    m["cty"] = m["GEO.id2"]
    m["msa"] = np.where(m.metro, m.Metropolitan_Area, "NM_" + m.cty)
    m = m.sort_values(["cty", "year"])
    need = outs + ["pp_poverty", "hous_inc_mean", "pop_density", "total_households", "population",
                             "pp_lf", "pp_college"]
    bad = m.loc[m[need].isna().any(axis=1), "cty"].unique()
    if len(bad):
        log(f"[{scope}] dropping {len(bad)} counties with missing values: {list(bad)}")
        m = m[~m.cty.isin(bad)]
    counts = m.groupby("cty").size()
    m = m[m.cty.isin(counts[counts == len(YEARS)].index)]
    units = m[m.year == 2010].set_index("cty")[["msa", "G", "State", "GEO.display.label", "pp_poverty",
                                                "hous_inc_mean", "pop_density", "c_pt", "c_car",
                                                "total_households", "population", "income_Gini", "metro",
                                                "xcoord", "ycoord"]].copy()
    units.columns = ["msa", "G", "state", "name", "b_pov", "b_inc", "b_dens", "b_pt", "b_car", "b_hh",
                     "b_pop", "b_gini", "metro", "lon", "lat"]
    units["upstate_ny"] = (units.state == "New York") & (units.G == 0) & units.metro
    wide = {k: m.pivot(index="cty", columns="year", values=k).loc[units.index, YEARS].values for k in outs}
    # time-varying county characteristics: placebo outcomes and recovery controls
    m["ln_pop"] = np.log(m.population)
    m["ln_hh"] = np.log(m.total_households)
    m["ln_inc"] = np.log(m.hous_inc_mean)
    m["ln_dens"] = np.log(m.pop_density)
    for k in list(TV) + ["ln_dens", "income_Gini", "price_gas"]:
        wide[k] = m.pivot(index="cty", columns="year", values=k).loc[units.index, YEARS].values
    return m, units, wide


def msa_driver_series(units, Y):
    """Population-weighted driver supply per MSA: dict msa -> (county positions, series over YEARS)."""
    pop = units.b_pop.values
    out = {}
    for msa, idx in units.groupby("msa").indices.items():
        out[msa] = (idx, (Y["drivers"][idx] * pop[idx, None]).sum(0) / pop[idx].sum())
    return out


def detect_year(s, c):
    """First year whose driver supply exceeds the previous year by at least c and never falls back below
    previous-year level + c afterwards; 0 if no such year."""
    for i in range(1, len(s)):
        if s[i] - s[i - 1] >= c and s[i:].min() - s[i - 1] >= c:
            return YEARS[i]
    return 0


def comparison_dates(units, Y, c=DETECT_JUMP, shift=0):
    """Year from which each county can no longer serve as a comparison: the recorded entry year for dated MSAs,
    2017 for upstate New York (service allowed from 29 June 2017), the detected rise (+shift) for other undated
    MSAs, and 0 for counties with no detected rise by 2017."""
    G = units.G.values
    det = np.zeros(len(units), int)
    for msa, (idx, s) in msa_driver_series(units, Y).items():
        det[idx] = detect_year(s, c)
    Gc = G.copy()
    und = G == 0
    Gc[und] = np.where(det[und] > 0, det[und] + shift, 0)
    Gc[units.upstate_ny.values] = 2017
    return Gc, det


def detection_calibration(units, Y, grid=DETECT_GRID):
    """Agreement between detected and recorded entry years across dated MSAs, for each threshold."""
    ser = msa_driver_series(units, Y)
    G, ups = units.G.values, units.upstate_ny.values
    rows = []
    for c in grid:
        diff, up, und = [], [], []
        for msa, (idx, s) in ser.items():
            d = detect_year(s, c)
            if ups[idx].any():
                up.append(d == 2017)
            elif G[idx][0] > 0:
                diff.append(d - G[idx][0] if d > 0 else 99)
            else:
                und.append(d)
        diff, und = np.array(diff), np.array(und)
        rows.append({"c": c, "n": len(diff), "same": np.mean(diff == 0), "within1": np.mean(np.abs(diff) <= 1),
                     "earlier": np.mean(diff < 0), "later1": np.mean(diff == 1),
                     "later2": np.mean((diff > 1) & (diff < 99)), "none": np.mean(diff == 99),
                     "upstate2017": int(np.sum(up)), "n_up": len(up), "und_detected": np.mean(und > 0),
                     "n_und": len(und)})
    return pd.DataFrame(rows)


def near_treated(units, k=5):
    """True for undated counties whose k nearest counties (centroid distance) include a dated one."""
    lat, lon = np.radians(units.lat.values), np.radians(units.lon.values)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k]
    tr = units.G.values > 0
    return (units.G.values == 0) & tr[nn].any(axis=1)


# ----------------------------------------------------------------------------- CS estimator (vectorised over bootstrap draws)
def control_mask(G, g, t, design, extra):
    """Comparison counties for cohort g in year t. extra['Gc'] holds the year from which each county stops
    being a valid comparison (0 = never)."""
    b = g - 1
    if design in ("notyet", "notyet16"):
        Gc = extra["Gc"]
        c = ((Gc == 0) | (Gc > max(t, b))) & (G != g)
    elif design == "never":
        c = extra["Gc"] == 0
    elif design == "upstate":
        c = extra["upstate"].copy()
    else:
        raise ValueError(design)
    if extra.get("exclude") is not None:
        c = c & ~extra["exclude"]
    return c


def tmax_of(design):
    return 2016 if design in ("upstate", "notyet16") else 2017


def cs_event(y, G, W, design="notyet", extra=None, subset=None):
    """Event-time ATT(e) for e in EVENTS; returns dict e -> array over W rows (row 0 = point estimate).
    Key '_lost' gives the share of bootstrap draws in which at least one cell of the point estimate is missing
    (all its treated or comparison counties left out of the resample)."""
    B1, N = W.shape
    sub = np.ones(N, bool) if subset is None else subset
    out = {}
    lost = np.zeros(B1, bool)
    for e in EVENTS:
        if e == -1:
            out[e] = np.zeros(B1)
            continue
        num = np.zeros(B1)
        den = np.zeros(B1)
        for g in COHORTS:
            t = g + e
            if t < YEARS[0] or t > tmax_of(design):
                continue
            tr = (G == g) & sub
            co = control_mask(G, g, t, design, extra) & sub
            if tr.sum() == 0 or co.sum() < 2:
                continue
            dy = y[:, YEARS.index(t)] - y[:, YEARS.index(g - 1)]
            wt, wc = W[:, tr], W[:, co]
            nt, nc = wt.sum(1), wc.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                att = (wt @ dy[tr]) / nt - (wc @ dy[co]) / nc
            ok = (nt > 0) & (nc > 0) & np.isfinite(att)
            lost |= ~ok
            num += np.where(ok, att * nt, 0.0)
            den += np.where(ok, nt, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[e] = np.where(den > 0, num / den, np.nan)
    out["_lost"] = lost[1:].mean()
    return out


def summarize(res, post=POST, pre=PRE):
    """Point, SE, pre-trend Wald test, post average, window-adjusted constant effect."""
    R = pd.DataFrame({e: res[e] for e in EVENTS})
    point, boot = R.iloc[0], R.iloc[1:]
    s = {"e": {e: (point[e], boot[e].std(ddof=1)) for e in EVENTS}, "lost": res.get("_lost", 0.0)}
    pm = R[post].mean(axis=1)
    s["post"] = (pm.iloc[0], pm.iloc[1:].std(ddof=1))
    # ACS 5-year: the estimate e years after entry covers (e+1) post-entry years out of 5,
    # so with a constant annual effect tau, ATT(e) = tau*(e+1)/5. Least squares through the origin:
    a = np.array([(e + 1) / 5 for e in post])
    tau = (R[post].values @ a) / (a @ a)
    s["tau"] = (tau[0], tau[1:].std(ddof=1))
    lead = R[pre].values
    L = lead[0]
    V = np.atleast_2d(np.cov(lead[1:].T))
    try:
        s["pre_p"] = 1 - stats.chi2.cdf(float(L @ np.linalg.solve(V, L)), len(pre))
    except np.linalg.LinAlgError:
        s["pre_p"] = np.nan
    return s


def county_tau(y, G, W, extra, post=POST):
    """County-level post-entry effect (average over post) for each treated county; array (B1, N)."""
    B1, N = W.shape
    tot = np.zeros((B1, N))
    cnt = np.zeros(N)
    for g in COHORTS:
        tr = G == g
        for e in post:
            t = g + e
            if t > YEARS[-1]:
                continue
            co = control_mask(G, g, t, "notyet", extra)
            dy = y[:, YEARS.index(t)] - y[:, YEARS.index(g - 1)]
            with np.errstate(invalid="ignore", divide="ignore"):
                cm = (W[:, co] @ dy[co]) / W[:, co].sum(1)
            tot[:, tr] += dy[tr][None, :] - cm[:, None]
            cnt[tr] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def wslope(tau, x, W, mask):
    """Weighted OLS slope of tau on x across counties in mask, for every bootstrap row."""
    w = W[:, mask]
    t = tau[:, mask]
    xx = x[mask][None, :]
    sw = w.sum(1, keepdims=True)
    xm = (w * xx).sum(1, keepdims=True) / sw
    tm = (np.nan_to_num(w * t)).sum(1, keepdims=True) / sw
    return ((w * (xx - xm) * (t - tm)).sum(1)) / ((w * (xx - xm) ** 2).sum(1))


# ----------------------------------------------------------------------------- regression helpers
def cluster_vcov(Xw, e, cl):
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    codes = pd.factorize(cl)[0]
    S = np.zeros((codes.max() + 1, Xw.shape[1]))
    np.add.at(S, codes, Xw * e[:, None])
    Gc = S.shape[0]
    return XtXi @ (S.T @ S) @ XtXi * Gc / (Gc - 1)


def ols_cluster(y, X, cl):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    V = cluster_vcov(X, e, np.asarray(cl)) * (n - 1) / (n - k)
    return beta, V


def year_dummies(N, T):
    Yd = np.zeros((N, T, T - 1))
    for t in range(1, T):
        Yd[:, t, t - 1] = 1.0
    return Yd


def etwfe(y, units, Gt, Gc, weights=None, zb=None, tv=(), leads=False, subset=None, post=POST):
    """Extended two-way fixed effects (Wooldridge 2025), written so that it equals the imputation estimator.
    Every treated county-year after entry has its own cohort x year indicator, and so does every comparison
    county-year after the county's own entry date. When covariates are included, each of these indicators is
    also interacted with the covariates, centred within the cell, so that county effects, year effects and the
    covariate coefficients are estimated from untreated county-years only.
        zb     : 2006-2010 characteristics, each interacted with year for all counties; scaled to one SD among
                 treated counties, so the cell interactions give the change in the effect per SD ('mod')
        tv     : time-varying controls (arrays N x T)
        leads  : indicators for every pre-entry year except the year before entry (e <= -2), so that e = -1 is
                 the reference for all cohorts; pre-trend test on e in PRE
    County effects are removed by within-county demeaning (exact: balanced panel, weights constant within
    county). Weights, if given, also weight cohorts in the aggregation. MSA-clustered standard errors."""
    treated_all = np.isin(Gt, COHORTS)
    keep = np.ones(len(units), bool) if subset is None else subset.copy()
    keep &= ~((Gc > 0) & (Gc <= YEARS[0]) & ~treated_all)   # never untreated in the panel
    gt, gc = Gt[keep], Gc[keep]
    N, T = int(keep.sum()), len(YEARS)
    w = np.ones(N) if weights is None else weights[keep] / weights[keep].mean()
    # dummied cells: (kind, group, year) with membership masks
    cells = [(g, t) for g in COHORTS if (gt == g).any() for t in YEARS if t >= g]
    nm = (gc > 0) & ~np.isin(gt, COHORTS)
    ncells = [(h, t) for h in sorted(set(gc[nm])) for t in YEARS if t >= h]
    lcells = [(g, e) for g in COHORTS if (gt == g).any() for e in range(YEARS[0] - g, -1) if e <= -2] if leads else []
    allcells = ([(gt == g, YEARS.index(t)) for g, t in cells] + [(nm & (gc == h), YEARS.index(t)) for h, t in ncells]
                + [(gt == g, YEARS.index(g + e)) for g, e in lcells])
    D = np.zeros((N, T, len(allcells)))
    for j, (mask, ti) in enumerate(allcells):
        D[mask, ti, j] = 1.0
    blocks = [D]
    covs = []                                              # covariate values N x T, one per covariate
    k_z = 0
    if zb is not None:
        Zall = (zb - zb[treated_all].mean(0)) / zb[treated_all].std(0, ddof=1)
        Z = Zall[keep]
        k_z = Z.shape[1]
        Yd = year_dummies(N, T)
        blocks.append((Z[:, None, :, None] * Yd[:, :, None, :]).reshape(N, T, -1))
        covs += [np.repeat(Z[:, q][:, None], T, axis=1) for q in range(k_z)]
    for arr in tv:
        blocks.append(arr[keep][:, :, None])
        covs.append(arr[keep])
    # saturation: every dummied cell gets its own slope on every covariate (centred within the cell)
    if covs:
        S = np.zeros((N, T, len(allcells) * len(covs)))
        for j, (mask, ti) in enumerate(allcells):
            if not mask.any():
                continue
            for q, cv in enumerate(covs):
                x = cv[mask, ti]
                S[mask, ti, j * len(covs) + q] = x - (x * w[mask]).sum() / w[mask].sum()
        blocks.append(S)
    blocks.append(year_dummies(N, T))
    X3 = np.concatenate(blocks, axis=2)
    K = X3.shape[2]
    Xd = (X3 - X3.mean(1, keepdims=True)).reshape(N * T, K)
    yd = (y[keep] - y[keep].mean(1, keepdims=True)).ravel()
    sw = np.sqrt(np.repeat(w, T))
    Xw, yw = Xd * sw[:, None], yd * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    V = cluster_vcov(Xw, yw - Xw @ beta, np.repeat(units.msa.values[keep], T))
    off_l = len(cells) + len(ncells)
    off_s = len(allcells) + (k_z * (T - 1)) + len(tv)
    size = {g: w[gt == g].sum() for g in COHORTS}

    def evvec(e, q=None):
        v = np.zeros(K)
        gs = ([g for g in COHORTS if (g, g + e) in cells] if e >= 0 else [g for g in COHORTS if (g, e) in lcells])
        tot = sum(size[g] for g in gs)
        for g in gs:
            j = cells.index((g, g + e)) if e >= 0 else off_l + lcells.index((g, e))
            v[j if q is None else off_s + j * len(covs) + q] = size[g] / tot
        return v

    est = lambda v: (float(v @ beta), float(np.sqrt(v @ V @ v)))
    out = {"e": {e: est(evvec(e)) for e in POST4}}
    out["post"] = est(sum(evvec(e) for e in post) / len(post))
    a = np.array([(e + 1) / 5 for e in post])
    out["tau"] = est(sum(a[i] * evvec(e) for i, e in enumerate(post)) / (a @ a))
    out["pre_p"] = np.nan
    if leads:
        Wl = np.vstack([evvec(e) for e in PRE])
        Lb = Wl @ beta
        out["pre_p"] = 1 - stats.chi2.cdf(float(Lb @ np.linalg.solve(Wl @ V @ Wl.T, Lb)), len(PRE))
        out["e"].update({e: est(evvec(e)) for e in PRE})
    if k_z:
        out["mod"] = [est(sum(evvec(e, q) for e in post) / len(post)) for q in range(k_z)]
    return out


def sa_dummies(y, units, Yall, weights=None, controls=(), zb=None):
    """Sun and Abraham (2021) interaction-weighted event study with never-treated comparison (G = 0), for the
    documentation table that reproduces the parallel analysis. Cohort weights are county counts, or household
    sums when weights are given (fixed in version 3). MSA-clustered standard errors."""
    keep = units.G.isin([0] + COHORTS).values
    G, N, T = units.G.values[keep], int(keep.sum()), len(YEARS)
    cols = [(g, e) for g in COHORTS for e in range(YEARS[0] - g, YEARS[-1] - g + 1) if e != -1]
    D = np.zeros((N, T, len(cols)))
    for j, (g, e) in enumerate(cols):
        D[G == g, YEARS.index(g + e), j] = 1.0
    parts = [D]
    for c in controls:
        parts.append(Yall[c][keep][:, :, None])
    Yd = year_dummies(N, T)
    if zb is not None:
        Z = zb[keep]
        Z = (Z - Z.mean(0)) / Z.std(0)
        parts.append((Z[:, None, :, None] * Yd[:, :, None, :]).reshape(N, T, -1))
    parts.append(Yd)
    X3 = np.concatenate(parts, axis=2)
    Xd = (X3 - X3.mean(1, keepdims=True)).reshape(N * T, -1)
    yd = (y[keep] - y[keep].mean(1, keepdims=True)).ravel()
    w = np.ones(N) if weights is None else weights[keep] / weights[keep].mean()
    sw = np.sqrt(np.repeat(w, T))
    Xw, yw = Xd * sw[:, None], yd * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    V = cluster_vcov(Xw, yw - Xw @ beta, np.repeat(units.msa.values[keep], T))
    ncoh = {g: w[G == g].sum() for g in COHORTS}
    wv = {}
    for ev in EVENTS:
        if ev == -1:
            continue
        gs = [g for g in COHORTS if (g, ev) in cols]
        tot = sum(ncoh[g] for g in gs)
        v = np.zeros(Xw.shape[1])
        for g in gs:
            v[cols.index((g, ev))] = ncoh[g] / tot
        wv[ev] = v
    wp = sum(wv[ev] for ev in POST) / len(POST)
    Wl = np.vstack([wv[ev] for ev in PRE])
    L = Wl @ beta
    pre_p = 1 - stats.chi2.cdf(float(L @ np.linalg.solve(Wl @ V @ Wl.T, L)), len(PRE))
    return {"post": (wp @ beta, np.sqrt(wp @ V @ wp)), "pre_p": pre_p}


def pooled_twfe_event(y, units, Yall, controls=(), weights=None):
    """Event study with relative-time dummies pooled across cohorts (not cohort-specific), county and year fixed
    effects, never-treated comparison, optional time-varying controls, county-clustered SEs (the specification
    of the parallel analysis). Returns average of POST and pre-trend p."""
    G, N, T = units.G.values, len(units), len(YEARS)
    D = np.array([[yr - g if g > 0 else -999 for yr in YEARS] for g in G])
    levels = sorted(set(D[G > 0].ravel()) - {-1})
    P = np.stack([(D == l).astype(float) for l in levels], axis=2)
    parts = [P] + [Yall[c][:, :, None] for c in controls]
    X3 = np.concatenate(parts + [year_dummies(N, T)], axis=2)
    Xd = (X3 - X3.mean(1, keepdims=True)).reshape(N * T, -1)
    yd = (y - y.mean(1, keepdims=True)).ravel()
    w = np.ones(N) if weights is None else weights / weights.mean()
    sw = np.sqrt(np.repeat(w, T))
    Xw, yw = Xd * sw[:, None], yd * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    V = cluster_vcov(Xw, yw - Xw @ beta, np.repeat(np.arange(N), T))
    vp = np.zeros(Xw.shape[1])
    for l in POST:
        vp[levels.index(l)] = 1 / len(POST)
    Wl = np.zeros((len(PRE), Xw.shape[1]))
    for i, l in enumerate(PRE):
        Wl[i, levels.index(l)] = 1
    L = Wl @ beta
    pre_p = 1 - stats.chi2.cdf(float(L @ np.linalg.solve(Wl @ V @ Wl.T, L)), len(PRE))
    return vp @ beta, np.sqrt(vp @ V @ vp), pre_p


def static_twfe(y, units, Gt, Gc):
    """One post-entry indicator (dated entry, detected rise, or upstate New York in 2017), county and year fixed
    effects, MSA-clustered SEs. For contrast only."""
    on = np.where(np.isin(Gt, COHORTS), Gt, np.where(Gc > 0, Gc, 9999))
    N, T = len(units), len(YEARS)
    D = (np.array(YEARS)[None, :] >= on[:, None]).astype(float)
    X3 =np.concatenate([D[:, :, None], year_dummies(N, T)], axis=2)
    Xd = (X3 - X3.mean(1, keepdims=True)).reshape(N * T, -1)
    yd = (y - y.mean(1, keepdims=True)).ravel()
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    V = cluster_vcov(Xd, yd - Xd @ beta, np.repeat(units.msa.values, T))
    return beta[0], np.sqrt(V[0, 0])


def long_difference(y, units, Gc, weighted=False):
    """2013-2017 window minus 2008-2012 window; dated cohorts 2013-2015 against counties with no ride-hailing
    before 2017 (no detected rise, detected only in 2017, or upstate New York); baseline covariates."""
    G = units.G.values
    tr = np.isin(G, [2013, 2014, 2015])
    co = ~np.isin(G, COHORTS) & ((Gc == 0) | (Gc >= 2017))
    sel = tr | co
    u = units[sel]
    dy = y[sel, YEARS.index(2017)] - y[sel, YEARS.index(2012)]
    X = np.column_stack([np.ones(len(u)), tr[sel].astype(float), np.log(u.b_inc), np.log(u.b_dens),
                         u.b_pov, u.b_pt])
    if weighted:
        sw = np.sqrt(u.b_hh.values / u.b_hh.mean())
        beta, V = ols_cluster(dy * sw, X * sw[:, None], u.msa.values)
    else:
        beta, V = ols_cluster(dy, X, u.msa.values)
    return beta[1], np.sqrt(V[1, 1]), int(tr.sum()), int(co.sum())


def old_design_normalization(m):
    """Earlier manuscript design: treated counties only, MSA + year dummies, event dummies, time-varying
    controls. Returns event coefficients under R's implicit normalization (D=-1 and D=+5 at zero) and an
    alternative one (D=-1 and D=-7 at zero). Both are equally valid because the design is rank deficient."""
    d = m[m.G > 0].copy()
    d["D"] = d.year - d.G
    d["internet"] = d.internet_levels.where(d.internet_levels != 1, 2)
    base = [pd.get_dummies(d.msa, prefix="M", dtype=float),
            pd.get_dummies(d.year, prefix="Y", dtype=float).drop(columns=["Y_2010"]),
            pd.DataFrame({"drv": d.drivers, "gas": d.price_gas, "gini": d.income_Gini,
                          "lninc": np.log(d.hous_inc_mean), "lnd": np.log(d.pop_density)}),
            pd.get_dummies(d.internet.astype(int), prefix="I", dtype=float).drop(columns=["I_2"])]
    Dd = pd.get_dummies(d.D.astype(int), prefix="D", dtype=float)
    full = pd.concat(base + [Dd.drop(columns=["D_-1"])], axis=1)
    rank = np.linalg.matrix_rank(full.values)
    out = {}
    for lab, drop in [("R default", ["D_-1", "D_5"]), ("Alternative", ["D_-1", "D_-7"])]:
        X = pd.concat(base + [Dd.drop(columns=drop)], axis=1)
        for k in ["s0", "s2"]:
            b = np.linalg.lstsq(X.values, d[k].values, rcond=None)[0]
            s = pd.Series(b, index=X.columns)
            out[(lab, k)] = {e: (s[f"D_{e}"] if f"D_{e}" in s.index else 0.0) for e in range(-7, 6)}
    return out, full.shape[1], rank


# ----------------------------------------------------------------------------- formatting
def stars(b, se):
    if not np.isfinite(b) or not np.isfinite(se) or se == 0:
        return ""
    p = 2 * (1 - stats.norm.cdf(abs(b / se)))
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(b, se, d=3):
    if not np.isfinite(b):
        return "--"
    return f"{b:.{d}f} ({se:.{d}f}){stars(b, se)}"


def fp(p):
    return "--" if not np.isfinite(p) else f"{p:.2f}"


def dec(k):
    return 4 if k in ("veh", "vmean") else 2 if k == "drivers" else 3


HDR = {"s0": "Zero (pp)", "s1": "One (pp)", "s2": "Two (pp)", "s3": "Three or more (pp)", "s3x": "Three (pp)",
       "s4p": "Four or more (pp)", "vmean": "Vehicles per household", "veh": "Capped vehicle index",
       "drivers": "Taxi and limousine nonemployers"}


def tex_table(path, caption, label, header, rows, colspec, notes):
    lines = ["\\begin{table}[!h]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\begin{adjustbox}{width=1\\textwidth}", f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
             " & ".join(header) + " \\\\ \\hline"]
    for r in rows:
        if r == "HLINE":
            lines.append("\\hline")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{adjustbox}",
              "\\par\\vspace{2pt}\\begin{minipage}{\\textwidth}\\footnotesize " + notes + "\\end{minipage}",
              "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  wrote {path}")


# ----------------------------------------------------------------------------- figures
def style_axes(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  wrote {path}")


def event_panel2(ax, series, title="", unit="", ylim=None):
    """series: list of (summary, label, style) with style 'county' (grey hollow) or 'household' (blue)."""
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(-1, color=GREY, linewidth=0.8, linestyle=(0, (3, 3)))
    xs = np.array(EVENTS)
    offs = {"county": -0.14, "household": 0.14}
    for s, _, sty in series:
        b = np.array([s["e"][e][0] for e in EVENTS])
        se = np.array([s["e"][e][1] for e in EVENTS])
        if sty == "household":
            ax.errorbar(xs + offs[sty], b, yerr=1.96 * se, fmt="o", color=BLUE, ms=4.5, elinewidth=1.4, zorder=3)
        else:
            ax.errorbar(xs + offs[sty], b, yerr=1.96 * se, fmt="o", mfc="white", mec=GREY, color=GREY, ms=4.5,
                        elinewidth=1.2, zorder=2)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=14)
    ax.text(0, 1.02, unit, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(EVENTS)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes(ax)


def legend_two(fig):
    h1 = plt.Line2D([], [], marker="o", mfc="white", mec=GREY, color=GREY, linestyle="", ms=5)
    h2 = plt.Line2D([], [], marker="o", color=BLUE, linestyle="", ms=5)
    fig.legend([h1, h2], ["County average (each county counts once)", "Household-weighted (each household counts once)"],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--veh", default=None, help="ACS_vehicles_2010-2017.csv from get_acs_vehicles.py")
    args, _ = ap.parse_known_args()   # ignores the extra arguments Jupyter passes
    if args.veh is None:
        cands = [os.path.join(os.path.dirname(os.path.abspath(args.data)), VEH_NAME), os.path.join(HERE, VEH_NAME)]
        args.veh = next((c for c in cands if os.path.exists(c)), cands[0])
    t0 = time.time()
    figdir, tabdir = os.path.join(args.out, "Figures"), os.path.join(args.out, "Tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    log.section("RUN")
    log("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("Python", sys.version.split()[0], "| numpy", np.__version__, "| pandas", pd.__version__)
    log("Data:", args.data)
    log("ACS vehicle tables:", args.veh)
    log("Output:", args.out)
    log("Bootstrap draws:", args.boot, "| seed:", SEED, "| summary window: years", POST, "after entry")

    m, units, Y = load_panel(args.data, "metro", args.veh)
    TK = ["s0", "s1", "s2", "s3x", "s4p"] if HAS_VEH else ["s0", "s1", "s2", "s3"]
    MK = "vmean" if HAS_VEH else "veh"
    if HAS_VEH:
        d3 = np.abs(Y["s3"] - Y["s3x"] - Y["s4p"])
        log(f"ACS tables B25044/B25046 read from {args.veh}. Outcomes: shares with 0, 1, 2, 3 and 4+ vehicles and "
            f"vehicles per household = aggregate vehicles / households. Check: |three or more - (three + four or "
            f"more)| max {np.nanmax(d3):.3f} pp; mean vehicles per household minus capped index: "
            f"{np.nanmean(Y['vmean'] - Y['veh']):.3f}")
    else:
        log(f"ACS vehicle tables not found at {args.veh}: using the capped vehicle index (three or more = three). "
            "Run get_acs_vehicles.py to obtain the actual mean and the split between three and four or more.")
    G = units.G.values
    N = len(units)
    ups = units.upstate_ny.values
    msa_codes, msa_idx = np.unique(units.msa.values, return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = np.stack([np.bincount(rng.integers(0, len(msa_codes), len(msa_codes)), minlength=len(msa_codes))
                      for _ in range(args.boot)])
    Wb = np.vstack([np.ones(N), draws[:, msa_idx]]).astype(float)
    hh = units.b_hh.values / units.b_hh.mean()
    WTS = {"county": Wb, "household": Wb * hh[None, :]}
    HHW = {"county": None, "household": units.b_hh.values}
    ELAB = {"county": "County average", "household": "Household-weighted"}
    treated = np.isin(G, COHORTS)
    records = []

    # comparison dates
    Gc, det = comparison_dates(units, Y)
    Gc_shift, _ = comparison_dates(units, Y, shift=-1)
    Gc_listed = np.where(G > 0, G, 0)                       # undated counties as never treated (version 2)
    G_all = np.where(np.isin(Gc, COHORTS), Gc, 0)           # undated counties dated by their rise and treated
    extra = {"Gc": Gc, "upstate": ups}
    near = near_treated(units, k=5)
    ZB = np.column_stack([np.log(units.b_inc), np.log(units.b_dens), units.b_pov, units.b_pt, np.log(units.b_hh)])
    ZB_names = ["Log mean household income", "Log population density", "Poverty rate", "Transit commute share",
                "Log households (county size)"]
    und = G == 0
    lo, hi = ZB[und].min(0), ZB[und].max(0)
    support = ((ZB >= lo) & (ZB <= hi)).all(1)
    cs_sub = support | ~treated                              # treated counties inside the comparison range

    def rec(table, design, outcome, b, se, p=np.nan):
        records.append({"table": table, "design": design, "outcome": outcome, "b": b, "se": se, "p_or_pretrend": p})

    # ------------------------------------------------------------------ sample and entry dates
    log.section("SAMPLE")
    log(f"Metro counties: {N} in {len(msa_codes)} MSAs; years {YEARS[0]}-{YEARS[-1]} (ACS 5-year windows)")
    log("Counties by recorded entry cohort (0 = no date in Lee et al. 2022):", pd.Series(G).value_counts().sort_index().to_dict())
    log("MSAs by recorded entry cohort:", units.drop_duplicates("msa").G.value_counts().sort_index().to_dict())
    log(f"Counties used as treated (cohorts {COHORTS[0]}-{COHORTS[-1]}): {treated.sum()}; "
        f"households in them (2006-2010): {units.b_hh[treated].sum() / 1e6:.1f} million "
        f"({100 * units.b_hh[treated].sum() / units.b_hh.sum():.1f}% of metro households)")
    log(f"Undated counties: {int(und.sum())} in {units[und].msa.nunique()} MSAs, "
        f"{units.b_hh[und].sum() / 1e6:.1f} million households")
    log(f"Upstate New York counties (service allowed from 29 June 2017): {int(ups.sum())} in "
        f"{units[ups].msa.nunique()} MSAs")

    log.section("ENTRY DATES FOR UNDATED METRO AREAS FROM DRIVER SUPPLY")
    cal = detection_calibration(units, Y)
    log("Agreement of the detected year with the recorded year across dated MSAs, by threshold (drivers per 10,000):")
    log(cal.round(3).to_string(index=False))
    log(f"Threshold used: {DETECT_JUMP} (rise over the previous year, sustained in all later years)")
    ud = units[und].assign(det=det[und], Gc=Gc[und]).drop_duplicates("msa")
    log("Undated MSAs by detected year (0 = no rise by 2017), upstate New York excluded:",
        ud[~ud.msa.isin(units[ups].msa)].det.value_counts().sort_index().to_dict())
    log("Undated counties by comparison end year (Gc):", pd.Series(Gc[und]).value_counts().sort_index().to_dict())
    nv = Gc == 0
    log(f"Counties with no ride-hailing by 2017 (no date, no rise): {int(nv.sum())} in {units[nv].msa.nunique()} MSAs, "
        f"{units.b_hh[nv].sum() / 1e3:.0f} thousand households; MSAs: {sorted(units[nv].msa.unique())}")
    ser = msa_driver_series(units, Y)
    big_und = (units[und].groupby("msa").b_hh.sum().sort_values(ascending=False)).index[:30]
    log("Thirty largest undated MSAs: driver supply 2010-2017 and detected year")
    for msa in big_und:
        idx, s = ser[msa]
        log(f"  {msa[:44]:44s} " + " ".join(f"{v:6.2f}" for v in s) + f"  -> {det[idx][0] if det[idx][0] else 'none'}")
    if "Anchorage, AK" in ser:
        log(f"Anchorage, AK (service allowed in Alaska from June 2017): detected {det[ser['Anchorage, AK'][0]][0]}")
    for grp, mask in [("dated, entry 2012-2016", treated), ("undated, rise detected", und & ~ups & (det > 0)),
                      ("undated, no rise by 2017", nv), ("upstate NY", ups)]:
        w = units.b_pop.values[mask]
        series = (Y["drivers"][mask] * w[:, None]).sum(0) / w.sum()
        log(f"Driver supply per 10,000 (population-weighted), {grp}: " +
            ", ".join(f"{y}: {v:.2f}" for y, v in zip(YEARS, series)))
    for t in YEARS[2:]:
        c = (Gc == 0) | (Gc > t)
        log(f"  comparison pool in {t} (any cohort): {int(c.sum())} counties in {units[c].msa.nunique()} MSAs")
    log(f"Common support: treated counties within the range of undated counties on all five baseline "
        f"characteristics: {int((support & treated).sum())} of {int(treated.sum())}, holding "
        f"{100 * units.b_hh[support & treated].sum() / units.b_hh[treated].sum():.0f}% of treated households")
    for j, nm in enumerate(ZB_names):
        log(f"  {nm:32s} undated range {lo[j]:.2f} to {hi[j]:.2f}; treated outside: {int(((ZB[:, j] < lo[j]) | (ZB[:, j] > hi[j]))[treated].sum())}")

    rows = []
    for _, r in cal.iterrows():
        rows.append([f"{r.c:.2f}" + (" (used)" if r.c == DETECT_JUMP else ""), f"{100 * r.same:.0f}",
                     f"{100 * r.within1:.0f}", f"{100 * r.earlier:.0f}", f"{100 * r.later1:.0f}",
                     f"{100 * (r.later2 + r.none):.0f}", f"{int(r.upstate2017)} of {int(r.n_up)}",
                     f"{100 * r.und_detected:.0f}"])
    tex_table(os.path.join(tabdir, "A3_entry_detection.tex"),
              "Dating entry from taxi and limousine nonemployers: agreement with recorded entry years", "tab:detect",
              ["Rise (per 10,000)", "Same year (\\%)", "Within one year (\\%)", "Earlier (\\%)",
               "One year later (\\%)", "Two or more later or none (\\%)", "Upstate New York in 2017",
               "Undated MSAs with a rise (\\%)"], rows, "l|c|c|c|c|c|c|c",
              f"Detected entry year: first year in which MSA taxi and limousine nonemployers (nonemployer "
              f"establishments in NAICS 4853 per 10,000 residents, population-weighted) exceed the previous year by at least the "
              f"threshold and stays at least that much above it in all later years. Columns 2 to 6 compare the "
              f"detected year with the recorded year for the {int(cal.n[0])} MSAs dated by \\citet{{Lee2022}}. Upstate "
              f"New York: MSAs where ride-hailing was allowed from 29 June 2017. Last column: share of the "
              f"{int(cal.n_und[0])} other undated MSAs with a detected rise by 2017.")

    # ------------------------------------------------------------------ T1 descriptives
    log.section("T1 DESCRIPTIVES (2010 = ACS 2006-2010)")
    groups = [("Entry 2012-2016", treated), ("No recorded entry", und), ("No ride-hailing by 2017", nv),
              ("Upstate New York", ups)]
    i2010 = YEARS.index(2010)
    desc_vars = [("Households (thousands)", units.b_hh.values / 1000, 1),
                 ("Population density (per km$^2$)", units.b_dens.values, 1),
                 ("Poverty rate (\\%)", units.b_pov.values, 1),
                 ("Mean household income (thousand USD)", units.b_inc.values / 1000, 1),
                 ("Public transit commute share (\\%)", units.b_pt.values, 2),
                 ("Zero vehicles (\\% of households)", Y["s0"][:, i2010], 2),
                 ("One vehicle (\\%)", Y["s1"][:, i2010], 2),
                 ("Two vehicles (\\%)", Y["s2"][:, i2010], 2),
                 ("Three or more vehicles (\\%)", Y["s3"][:, i2010], 2)]
    if HAS_VEH:
        desc_vars += [("\\quad Three vehicles (\\%)", Y["s3x"][:, i2010], 2),
                      ("\\quad Four or more vehicles (\\%)", Y["s4p"][:, i2010], 2),
                      ("Vehicles per household", Y["vmean"][:, i2010], 3)]
    else:
        desc_vars += [("Capped vehicle index", Y["veh"][:, i2010], 3)]
    desc_vars += [("Taxi and limousine nonemployers (per 10,000), 2010", Y["drivers"][:, i2010], 2),
                  ("Taxi and limousine nonemployers (per 10,000), 2017", Y["drivers"][:, YEARS.index(2017)], 2)]
    rows = []
    for lab, v, d in desc_vars:
        r = [lab] + [f"{np.mean(v[mk]):.{d}f} ({np.std(v[mk], ddof=1):.{d}f})" for _, mk in groups]
        rows.append(r)
        log(f"{lab:45s} " + " | ".join(r[1:]))
    rows += ["HLINE", ["Counties"] + [str(int(mk.sum())) for _, mk in groups],
             ["MSAs"] + [str(units[mk].msa.nunique()) for _, mk in groups]]
    tex_table(os.path.join(tabdir, "T1_descriptives.tex"),
              "Baseline characteristics of metropolitan counties (ACS 2006-2010)", "tab:descriptives",
              ["", "Entry 2012-2016", "No recorded entry", "No ride-hailing by 2017", "Upstate New York"], rows,
              "l|c|c|c|c",
              "Mean (standard deviation) across counties. Entry years from \\citet{Lee2022}. No ride-hailing by 2017: "
              "counties without a recorded entry year with no sustained rise in taxi and limousine nonemployers by 2017. Upstate "
              "New York: counties where ride-hailing was not allowed until 29 June 2017. Both are subsets of the "
              "no-recorded-entry group. Poverty rate: percentage of families and people whose income in the past 12 "
              "months is below the poverty level (ACS DP03). " +
              ("Vehicles per household: aggregate vehicles available (ACS B25046) divided by households. "
               if HAS_VEH else "Capped vehicle index: households with three or more vehicles counted as three. ") +
              "Taxi and limousine nonemployers: nonemployer establishments in NAICS 4853 per 10,000 residents.")

    # ------------------------------------------------------------------ main CS event studies, two estimands
    main_r = {"county": {}, "household": {}}
    for est in ["county", "household"]:
        log.section(f"MAIN: CALLAWAY-SANT'ANNA, CLEAN NOT-YET-TREATED COMPARISON, {ELAB[est].upper()}")
        for k in list(OUTCOMES) + (list(VEH_OUT) if HAS_VEH else []):
            r = cs_event(Y[k], G, WTS[est], "notyet", extra)
            s = summarize(r)
            s4 = summarize(r, post=POST4)
            main_r[est][k] = s
            log(f"{SHORT[k]:24s} " + "  ".join(f"e{e:+d}: {fmt(*s['e'][e], dec(k))}" for e in EVENTS if e != -1))
            log(f"{'':24s} avg 0-2 {fmt(*s['post'], dec(k))} | constant effect {fmt(*s['tau'], dec(k))} | "
                f"pre-trend p = {s['pre_p']:.3f} | avg 0-3 {fmt(*s4['post'], dec(k))}, constant effect 0-3 "
                f"{fmt(*s4['tau'], dec(k))} | draws losing a cell: {100 * s['lost']:.1f}%")
            for e in EVENTS:
                rec(f"event_{est}", f"e{e}", k, *s["e"][e])
            rec(f"summary_{est}", "avg_0_2", k, *s["post"], s["pre_p"])
            rec(f"summary_{est}", "constant_effect_0_2", k, *s["tau"], s["pre_p"])
            rec(f"summary_{est}", "avg_0_3", k, *s4["post"], s["pre_p"])
    rows = []
    for est in ["county", "household"]:
        rows.append([ELAB[est]] + [""] * 6)
        for k in ["drivers"] + TK + [MK]:
            s = main_r[est][k]
            rows.append(["\\quad " + SHORT[k]] + [fmt(*s["e"][e], dec(k)) for e in POST4] +
                        [fmt(*s["post"], dec(k)), f"{s['pre_p']:.2f}"])
        if est == "county":
            rows.append("HLINE")
    tex_table(os.path.join(tabdir, "T2_main_event.tex"),
              "Effect of ride-hailing entry on taxi and limousine nonemployers and household vehicle holdings",
              "tab:main", ["", "Entry year", "1 year after", "2 years after", "3 years after", "Average 0-2",
                           "Pre-trend $p$"], rows, "l|c|c|c|c|c|c",
              "Callaway and Sant'Anna group-time effects with the year before entry as base, aggregated by years "
              "since entry with cohort-size weights. Comparison: counties not yet reached by ride-hailing, where "
              "counties without a recorded date leave the comparison group in the year their taxi and limousine "
              "nonemployers rise (Table~\\ref{tab:detect}). County average: each county counts once. Household-weighted: "
              "counties and cohorts weighted by 2006-2010 households. Vehicle shares in percentage points of "
              "households, nonemployers per 10,000 residents. " +
              ("Vehicles per household: aggregate vehicles available divided by households. " if HAS_VEH else
               "Capped vehicle index: three or more vehicles counted as three. ") +
              "Each ACS estimate pools five survey years, so the estimate $e$ years after entry contains $e+1$ "
              "years with ride-hailing. Average 0-2: mean of the first three columns. Three years after entry, most comparisons are made in 2017, "
              f"when {int(nv.sum())} counties remained without ride-hailing. Pre-trend $p$: joint Wald test that the effects 2 to "
              "4 years before entry are zero. MSA block bootstrap standard errors in parentheses (" +
              str(args.boot) + " draws). * $p<0.10$; ** $p<0.05$; *** $p<0.01$.")

    tier_lim = (-0.9, 0.9)
    panels = [("drivers", None)] + [(k, tier_lim) for k in TK] + [(MK, None)]
    ncol = 3 if len(panels) <= 6 else 4
    fig, axs = plt.subplots(2, ncol, figsize=(10 if ncol == 3 else 12.5, 6.2))
    for ax, (k, lim) in zip(axs.ravel(), panels):
        event_panel2(ax, [(main_r["county"][k], "", "county"), (main_r["household"][k], "", "household")],
                     SHORT[k], UNITS.get(k, "pp of households"), lim)
    for ax in axs.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax in axs.ravel()[len(panels) - ncol:len(panels)]:
        ax.set_xlabel("Years relative to entry", fontsize=8.5, color=INK2)
    legend_two(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, os.path.join(figdir, "F1_event_study.png"))

    # ------------------------------------------------------------------ robustness
    log.section("ROBUSTNESS: AVERAGE EFFECT 0-2 YEARS AFTER ENTRY")
    rob_keys = TK + [MK, "drivers"]
    log(f"Undated counties with a dated county among their 5 nearest counties: {int(near.sum())} of "
        f"{int(und.sum())} (dropped in the spillover check)")
    big10 = units.b_hh.rank(ascending=False).values <= 10
    log("Ten largest counties (households):", "; ".join(units.name[big10]))
    G_bal = np.where(np.isin(G, [2012, 2013, 2014]), G, 0)
    nld = long_difference(Y["s0"], units, Gc)
    log(f"Long difference sample: {nld[2]} treated (cohorts 2013-2015), {nld[3]} comparison counties")

    def cs_d(est, Gt=G, ex=None, design="notyet", subset=None, pre=PRE, post=POST, tag=""):
        ex = extra if ex is None else ex

        def f(k):
            r = cs_event(Y[k], Gt, WTS[est], design, ex, subset)
            s = summarize(r, post=post, pre=pre)
            if k in ("s0", MK) and r["_lost"] > 0:
                log(f"    [{tag} {k}] bootstrap draws losing a cell: {100 * r['_lost']:.1f}%")
            if k == MK and tag.startswith("balanced"):
                log(f"    [{tag}] vehicles per household by event time: " +
                    "  ".join(f"e{e:+d}: {fmt(*s['e'][e], 4)}" for e in range(-2, 4)))
            return s["post"][0], s["post"][1], s["pre_p"]
        return f

    etc = {}

    def et_d(est, zb=None, tv=(), subset=None):
        def f(k):
            key = (est, zb is not None, tuple(id(a) for a in tv), None if subset is None else int(subset.sum()), k)
            if key not in etc:
                r = etwfe(Y[k], units, G, Gc, HHW[est], zb, tv, False, subset)
                rl = etwfe(Y[k], units, G, Gc, HHW[est], zb, tv, True, subset)
                r["pre_p"] = rl["pre_p"]
                etc[key] = r
            s = etc[key]
            return s["post"][0], s["post"][1], s["pre_p"]
        return f

    def ld(weighted):
        def f(k):
            b, se, _, _ = long_difference(Y[k], units, Gc, weighted)
            return b, se, np.nan
        return f

    def twfe(k):
        b, se = static_twfe(Y[k], units, G, Gc)
        return b, se, np.nan

    TVREC = [Y[c] for c in RECOVERY]
    designs = {}
    for est in ["county", "household"]:
        d = [("Main: not-yet-treated, undated counties until their rise", cs_d(est)),
             ("Undated counties as never treated", cs_d(est, ex={"Gc": Gc_listed, "upstate": ups})),
             ("Undated counties leave one year before their rise", cs_d(est, ex={"Gc": Gc_shift, "upstate": ups})),
             ("Undated counties treated from their rise", cs_d(est, Gt=G_all)),
             ("Upstate New York comparison, to 2016", cs_d(est, design="upstate", tag="upstate")),
             ("Cohorts 2012-2014, balanced event time", cs_d(est, Gt=G_bal, pre=[-2], tag=f"balanced {est}")),
             ("Comparison counties not next to dated ones",
              cs_d(est, ex={"Gc": Gc, "upstate": ups, "exclude": near})),
             ("Common support of baseline characteristics", cs_d(est, subset=cs_sub, tag="support")),
             ("Extended TWFE", et_d(est)),
             ("Extended TWFE, baseline characteristics x year", et_d(est, ZB)),
             ("Extended TWFE, baseline x year, common support", et_d(est, ZB, subset=cs_sub)),
             ("Extended TWFE, recovery controls", et_d(est, tv=TVREC)),
             ("Long difference, non-overlapping windows", ld(est == "household"))]
        if est == "county":
            d.append(("Static two-way fixed effects", twfe))
        else:
            d.insert(7, ("Without the ten largest counties", cs_d(est, subset=~big10)))
        designs[est] = d
    rows = []
    rob = {}
    for est in ["county", "household"]:
        rows.append([ELAB[est]] + [""] * (len(rob_keys) + 1))
        for lab, f in designs[est]:
            r, pres = ["\\quad " + lab], []
            for k in rob_keys:
                b, se, p = f(k)
                rob[(est, lab, k)] = (b, se, p)
                rec(f"robustness_{est}", lab, k, b, se, p)
                r.append(fmt(b, se, dec(k)))
                if k in ("s0", MK):
                    pres.append(fp(p))
            r.append(" / ".join(pres))
            rows.append(r)
            log(f"{ELAB[est][:9]} {lab:56s} " + " | ".join(f"{SHORT[k]}: {x}" for k, x in zip(rob_keys, r[1:-1])) +
                f" | pre-trend p (zero / veh): {r[-1]}")
        if est == "county":
            rows.append("HLINE")
    log("Driver supply pre-trend p by design (comparison groups not selected on driver supply are the 'never treated' and upstate rows):")
    for est in ["county", "household"]:
        for lab, _ in designs[est]:
            bb, se, p = rob[(est, lab, "drivers")]
            log(f"  {ELAB[est][:9]} {lab:56s} driver supply {fmt(bb, se, 2)} | pre-trend p {fp(p)}")
    log("Sensitivity to the detection threshold (main design, average 0-2):")
    for cthr in [0.75, 1.5]:
        Gc_t, _ = comparison_dates(units, Y, cthr)
        out = []
        for est in ["county", "household"]:
            for k in ["s0", "s2", MK]:
                s = summarize(cs_event(Y[k], G, WTS[est], "notyet", {"Gc": Gc_t, "upstate": ups}))
                out.append(f"{ELAB[est][:9]} {SHORT[k]}: {fmt(*s['post'], dec(k))}")
                rec("threshold_sensitivity", f"jump_{cthr}_{est}", k, *s["post"], s["pre_p"])
        log(f"  threshold {cthr}: no-rise counties {int((Gc_t == 0).sum())} | " + " | ".join(out))
    tex_table(os.path.join(tabdir, "A1_robustness.tex"),
              "Robustness of the average effect in the entry year and the two following years", "tab:robust",
              [""] + [HDR[k] for k in rob_keys] + ["Pre-trend $p$"], rows, "l|" + "|".join(["c"] * (len(rob_keys) + 1)),
              "Callaway and Sant'Anna estimates averaged over 0 to 2 years after entry with MSA block bootstrap "
              "standard errors, unless stated otherwise. Undated counties: counties without a recorded entry year. "
              "In the main design they serve as comparisons until their taxi and limousine nonemployers rise (Table~\\ref{tab:detect}), "
              "and upstate New York until 2017. Treated from their rise: undated counties join the treated cohorts "
              "at their detected year. Upstate New York: the 24 counties where ride-hailing was not allowed until "
              "June 2017 as the only comparison, years up to 2016. Balanced event time: cohorts observed from 2 "
              "years before to 3 years after entry, pre-trend test on 2 years before. Not next to dated ones: undated "
              "comparison counties with a dated county among their five nearest counties are dropped. Common "
              "support: treated counties whose 2006-2010 characteristics all lie within the range of undated "
              "counties. Extended TWFE: county and year effects estimated from untreated county-years "
              "\\citep{Wooldridge2025}, MSA-clustered standard errors, pre-trend test from the same model with "
              "leads. Baseline characteristics x year: 2006-2010 log mean household income, log population density, "
              "poverty rate, transit commute share and log households, each interacted with year. Recovery "
              "controls: time-varying log population, log mean household income, labor force participation and "
              "poverty rate. Long difference: change between the ACS windows 2008-2012 and 2013-2017 for counties "
              "entered in 2013-2015 against counties without ride-hailing before 2017, with baseline covariates, "
              "MSA-clustered standard errors. Static two-way fixed effects: one post-entry indicator with county and "
              "year fixed effects. Pre-trend $p$: zero vehicles / " + ("vehicles per household. " if HAS_VEH else "capped vehicle index. ") +
              "* $p<0.10$; ** $p<0.05$; *** $p<0.01$.")

    # ------------------------------------------------------------------ placebo outcomes and commuting
    log.section("PLACEBO OUTCOMES AND COMMUTING (AVERAGE 0-2, MAIN DESIGN)")
    rows = [["Commuting (\\% of workers)"] + [""] * 4]
    for k in ["c_car", "c_cp", "c_pt"]:
        r = ["\\quad " + SHORT[k]]
        for est in ["county", "household"]:
            s = main_r[est][k]
            r += [fmt(*s["post"]), f"{s['pre_p']:.2f}"]
        rows.append(r)
        log(f"{SHORT[k]:40s} county {r[1]} (pre p {r[2]}) | household {r[3]} (pre p {r[4]})")
    rows += ["HLINE", ["County characteristics"] + [""] * 4]
    for k, lab in TV.items():
        r = ["\\quad " + lab]
        for est in ["county", "household"]:
            s = summarize(cs_event(Y[k], G, WTS[est], "notyet", extra))
            d = 4 if k.startswith("ln_") else 3
            r += [fmt(*s["post"], d), f"{s['pre_p']:.2f}"]
            rec(f"placebo_{est}", "avg_0_2", k, *s["post"], s["pre_p"])
        rows.append(r)
        log(f"{lab:40s} county {r[1]} (pre p {r[2]}) | household {r[3]} (pre p {r[4]})")
    rows += ["HLINE", ["County characteristics, upstate New York comparison (to 2016)"] + [""] * 4]
    for k in ["ln_pop", "ln_inc", "pp_lf"]:
        r = ["\\quad " + TV[k]]
        for est in ["county", "household"]:
            s = summarize(cs_event(Y[k], G, WTS[est], "upstate", extra))
            d = 4 if k.startswith("ln_") else 3
            r += [fmt(*s["post"], d), f"{s['pre_p']:.2f}"]
            rec(f"placebo_upstate_{est}", "avg_0_2", k, *s["post"], s["pre_p"])
        rows.append(r)
        log(f"upstate comparison: {TV[k]:28s} county {r[1]} (pre p {r[2]}) | household {r[3]} (pre p {r[4]})")
    tex_table(os.path.join(tabdir, "A2_placebo_outcomes.tex"),
              "Entry effects on commuting and on county characteristics", "tab:placebo",
              ["", "County average", "Pre-trend $p$", "Household-weighted", "Pre-trend $p$"], rows, "l|c|c|c|c",
              "Same estimator and comparison as the main table, average over 0 to 2 years after entry. Commuting "
              "shares in percentage points of workers aged 16 and over. Log outcomes in log points. Ride-hailing "
              "should move the county characteristics little, so differences in them indicate differences in "
              "growth between counties entered earlier and later. Last panel: the 24 upstate New York counties as "
              "the only comparison, years up to 2016. * $p<0.10$; ** $p<0.05$; *** $p<0.01$.")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, k in zip(axs, ["c_car", "c_cp", "c_pt"]):
        event_panel2(ax, [(main_r["county"][k], "", "county"), (main_r["household"][k], "", "household")],
                     SHORT[k], "pp of workers", (-1.2, 1.2))
        ax.set_xlabel("Years relative to entry", fontsize=8.5, color=INK2)
    legend_two(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, os.path.join(figdir, "FA1_commuting.png"))

    # ------------------------------------------------------------------ heterogeneity
    log.section("HETEROGENEITY: EXTENDED TWFE WITH TREATMENT x CHARACTERISTIC AND CHARACTERISTIC x YEAR (COUNTY AVERAGE)")
    het_keys = TK + [MK, "drivers"]
    Zs = (ZB - ZB[treated].mean(0)) / ZB[treated].std(0, ddof=1)
    sds = ZB[treated].std(0, ddof=1)
    order = [2, 0, 4, 1, 3]                                  # poverty first
    rows = [["One characteristic at a time"] + [""] * len(het_keys)]
    het_uni, het_joint = {}, {}
    for j in order:
        r = ["\\quad " + ZB_names[j] + f" (SD {sds[j]:.2f})"]
        for k in het_keys:
            o = etwfe(Y[k], units, G, Gc, None, ZB[:, [j]])
            het_uni[(j, k)] = o["mod"][0]
            r.append(fmt(*o["mod"][0], dec(k)))
            rec("heterogeneity_univariate", ZB_names[j], k, *o["mod"][0])
        rows.append(r)
        log(f"univariate {ZB_names[j]:32s} " + " | ".join(f"{SHORT[k]}: {x}" for k, x in zip(het_keys, r[1:])))
    rows += ["HLINE", ["All five together"] + [""] * len(het_keys)]
    for k in het_keys:
        o = etwfe(Y[k], units, G, Gc, None, ZB)
        for j in range(5):
            het_joint[(j, k)] = o["mod"][j]
            rec("heterogeneity_joint", ZB_names[j], k, *o["mod"][j])
    for j in order:
        r = ["\\quad " + ZB_names[j]] + [fmt(*het_joint[(j, k)], dec(k)) for k in het_keys]
        rows.append(r)
        log(f"joint      {ZB_names[j]:32s} " + " | ".join(f"{SHORT[k]}: {x}" for k, x in zip(het_keys, r[1:])))
    log.section("HETEROGENEITY, FOR COMPARISON: SLOPES OF COUNTY-LEVEL EFFECTS (NO CHARACTERISTIC x YEAR)")
    taus = {k: county_tau(Y[k], G, Wb, extra) for k in het_keys + ["s2p"]}
    log("(log only) slopes of each treated county's effect on one characteristic, comparing treated counties among themselves:")
    for j in order:
        out, r = [], ["\\quad " + ZB_names[j]]
        for k in het_keys:
            sl = wslope(taus[k], Zs[:, j], Wb, treated)
            out.append(f"{SHORT[k]}: {fmt(sl[0], np.nanstd(sl[1:], ddof=1), dec(k))}")
            rec("heterogeneity_county_tau_slope", ZB_names[j], k, sl[0], np.nanstd(sl[1:], ddof=1))
        log(f"univariate {ZB_names[j]:32s} " + " | ".join(out))
    tex_table(os.path.join(tabdir, "T3_heterogeneity.tex"),
              "Variation of the post-entry effect with baseline county characteristics", "tab:hetero",
              ["Change in effect per +1 SD of"] + [HDR[k] for k in het_keys], rows,
              "l|" + "|".join(["c"] * len(het_keys)),
              "Extended two-way fixed effects \\citep{Wooldridge2025} with each post-entry cohort-year effect "
              "interacted with 2006-2010 characteristics, averaged over 0 to 2 years after entry, county average. "
              "Each characteristic is also interacted with year for all counties, so the estimates compare treated "
              "and comparison counties with the same characteristic. Characteristics scaled to one standard "
              "deviation (SD) among treated counties. One at a time: one characteristic per model. All five "
              "together: one model with all five. MSA-clustered standard errors. "
              "* $p<0.10$; ** $p<0.05$; *** $p<0.01$.")

    log("Dose response: ownership effect vs driver-supply effect across treated counties (no characteristic x year)")
    for k in ["s0", "s2p", MK]:
        sl = []
        for b in range(Wb.shape[0]):
            x = taus["drivers"][b] / np.nanstd(taus["drivers"][0][treated], ddof=1)
            sl.append(wslope(taus[k][b:b + 1], x, Wb[b:b + 1], treated)[0])
        sl = np.array(sl)
        se = np.nanstd(sl[1:], ddof=1)
        rec("dose_response", "per_sd_driver_effect", k, sl[0], se)
        log(f"  {SHORT[k]:24s} slope per +1 SD of the county driver-supply effect: {fmt(sl[0], se, dec(k))}")

    log.section("HETEROGENEITY: TERCILES (CALLAWAY-SANT'ANNA WITHIN TERCILE, COHORTS 2012-2014, YEARS TO 2016, COUNTY AVERAGE, AVERAGE 0-2)")
    fig_mods = [(2, "Poverty rate"), (4, "County size"), (3, "Transit commute share")]
    fig_keys = ["s0", "s2", "s3", MK]
    terc_res = {}
    for j, mlab in fig_mods:
        x = ZB[:, j]
        cuts = np.quantile(x, [1 / 3, 2 / 3])
        terc = np.digitize(x, cuts)
        log(f"{mlab}: cut points {np.round(cuts, 3)}; treated per tercile "
            f"{[int(((terc == q) & treated).sum()) for q in range(3)]}; comparison counties in 2016 / 2017 per tercile "
            f"{[(int((((Gc == 0) | (Gc > 2016)) & (terc == q)).sum()), int((((Gc == 0) | (Gc > 2017)) & (terc == q)).sum())) for q in range(3)]}")
        for k in fig_keys:
            for q in range(3):
                r = cs_event(Y[k], G_bal, Wb, "notyet16", extra, subset=(terc == q))
                s = summarize(r)
                if r["_lost"] > 0:
                    log(f"    [{mlab} T{q + 1} {k}] bootstrap draws losing a cell: {100 * r['_lost']:.1f}%")
                terc_res[(j, k, q)] = s["post"]
                rec("heterogeneity_tercile", f"{mlab}_T{q + 1}", k, *s["post"], s["pre_p"])
            log(f"  {SHORT[k]:24s} " + " | ".join(f"T{q + 1}: {fmt(*terc_res[(j, k, q)], dec(k))}" for q in range(3)))
    fig, axs = plt.subplots(len(fig_mods), len(fig_keys), figsize=(11, 7.2), sharey="col")
    for i, (j, mlab) in enumerate(fig_mods):
        for jj, k in enumerate(fig_keys):
            ax = axs[i, jj]
            b = np.array([terc_res[(j, k, q)][0] for q in range(3)])
            se = np.array([terc_res[(j, k, q)][1] for q in range(3)])
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.errorbar(range(3), b, yerr=1.96 * se, fmt="o", color=BLUE, ms=5, elinewidth=1.4)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["Low", "Middle", "High"])
            ax.set_xlim(-0.5, 2.5)
            style_axes(ax)
            if i == 0:
                ax.set_title(SHORT[k] + ("" if k in ("veh", "vmean") else " (pp)"), fontsize=9.5, color=INK,
                             fontweight="bold", loc="left")
            if jj == 0:
                ax.set_ylabel(mlab + "\ntercile", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, os.path.join(figdir, "F2_heterogeneity_terciles.png"))

    # ------------------------------------------------------------------ benchmarks
    log.section("MAGNITUDES (LOG ONLY): EFFECTS ON " + SHORT[MK].upper() + " AS A PERCENTAGE OF THE 2006-2010 LEVEL")
    log("Observed: average effect on the ACS five-year estimates 0-2 years after entry. Approximate annual effect: "
        "rescaling that assumes a constant effect in each year with ride-hailing and that a five-year estimate "
        "averages its five survey years (Equation 3 of version 3, now reported only here).")
    for est in ["county", "household"]:
        w = units.b_hh.values[treated] if est == "household" else np.ones(treated.sum())
        base = np.average(Y[MK][treated, YEARS.index(2010)], weights=w)
        cands = [("main design", main_r[est][MK]),
                 ("extended TWFE, baseline characteristics x year", etc[(est, True, (), None, MK)])]
        for lab, s_ in cands:
            for nm, (tb, tse) in [("observed average 0-2", s_["post"]), ("approximate annual effect", s_["tau"])]:
                lo_, hi_ = tb - 1.96 * tse, tb + 1.96 * tse
                log(f"{ELAB[est]}, {lab}, {nm}: baseline {base:.3f}; {tb:.4f} (95% CI {lo_:.4f} to {hi_:.4f}) = "
                    f"{100 * tb / base:.2f}% (95% CI {100 * lo_ / base:.2f}% to {100 * hi_ / base:.2f}%)")
                rec("magnitude", f"{est}_{lab}_{nm}", MK, 100 * tb / base, 100 * tse / base)

    # ------------------------------------------------------------------ documentation: parallel analysis specifications
    log.section("DOCUMENTATION D2: PARALLEL ANALYSIS SPECIFICATIONS (UNDATED COUNTIES AS NEVER TREATED)")
    mA, uA, YA = load_panel(args.data, "all", args.veh)
    ZBA = np.column_stack([np.log(uA.b_inc), np.log(uA.b_dens), uA.b_pov, uA.b_pt, np.log(uA.b_hh)])
    gem_ctrl = ["ln_dens", "income_Gini", "ln_inc", "price_gas"]
    tiers = TK + [MK]
    keepA = uA.G.isin([0] + COHORTS + [2017]).values
    post_p = lambda s: (*s["post"], s["pre_p"])
    alt = [
        ("Pooled event-study TWFE, all counties, unweighted, time-varying controls",
         lambda k: pooled_twfe_event(YA[k][keepA], uA[keepA], {c: YA[c][keepA] for c in gem_ctrl}, gem_ctrl)),
        ("Sun and Abraham, all counties, household-weighted",
         lambda k: post_p(sa_dummies(YA[k], uA, YA, uA.b_hh.values))),
        ("\\quad + time-varying controls",
         lambda k: post_p(sa_dummies(YA[k], uA, YA, uA.b_hh.values, gem_ctrl))),
        ("\\quad + baseline characteristics x year",
         lambda k: post_p(sa_dummies(YA[k], uA, YA, uA.b_hh.values, gem_ctrl, ZBA))),
        ("Sun and Abraham, metro counties, household-weighted",
         lambda k: post_p(sa_dummies(Y[k], units, Y, units.b_hh.values))),
        ("Sun and Abraham, metro counties, household-weighted, baseline x year",
         lambda k: post_p(sa_dummies(Y[k], units, Y, units.b_hh.values, (), ZB))),
        ("Sun and Abraham, metro counties, county average, baseline x year",
         lambda k: post_p(sa_dummies(Y[k], units, Y, None, (), ZB))),
    ]
    rows = [["Panel A: specifications"] + [""] * (len(tiers) + 1)]
    for lab, f in alt:
        r, pres = ["\\quad " + lab if not lab.startswith("\\quad") else "\\quad" + lab], []
        for k in tiers:
            b, se, p = f(k)
            rec("documentation_parallel", lab, k, b, se, p)
            r.append(fmt(b, se, dec(k)))
            if k in ("s0", MK):
                pres.append(fp(p))
        r.append(" / ".join(pres))
        rows.append(r)
        log(f"{lab:72s} " + " | ".join(f"{SHORT[k]}: {x}" for k, x in zip(tiers, r[1:-1])) + f" | pre p (zero/veh) {r[-1]}")
    for mvar, mname, panel in [("b_pov", "poverty rate", "B"), ("b_pt", "transit commute share", "C")]:
        med = np.median(uA[mvar].values)
        log(f"Median {mname} across all counties (2006-2010): {med:.2f}%")
        rows.append("HLINE")
        rows.append([f"Panel {panel}: split at the median {mname} ({med:.1f}\\%), household-weighted, metro counties"] +
                    [""] * (len(tiers) + 1))
        for grp, mask in [("below median", units[mvar].values <= med), ("above median", units[mvar].values > med)]:
            for elab, fn in [("Sun and Abraham, never-treated", None), ("main design", "cs")]:
                r, pres = [f"\\quad {mname} {grp}, {elab} (treated {int((mask & treated).sum())})"], []
                for k in tiers:
                    if fn is None:
                        sub = units[mask]
                        s = sa_dummies(Y[k][mask], sub, {c: Y[c][mask] for c in Y}, sub.b_hh.values)
                        b, se, p = s["post"][0], s["post"][1], s["pre_p"]
                    else:
                        s = summarize(cs_event(Y[k], G, WTS["household"], "notyet", extra, subset=mask))
                        b, se, p = s["post"][0], s["post"][1], s["pre_p"]
                    rec("documentation_median_split", f"{mvar}_{grp}_{elab}", k, b, se, p)
                    r.append(fmt(b, se, dec(k)))
                    if k in ("s0", MK):
                        pres.append(fp(p))
                r.append(" / ".join(pres))
                rows.append(r)
                log(f"{mname} {grp:13s} {elab:32s} " + " | ".join(f"{SHORT[k]}: {x}" for k, x in zip(tiers, r[1:-1])) +
                    f" | pre p (zero/veh) {r[-1]}")
    log("Check of the version-3 fix: Sun and Abraham (metro, household-weighted) against Callaway-Sant'Anna with "
        "the same never-treated comparison (undated counties), average 0-2; the two must coincide")
    for k in tiers:
        sa_ = sa_dummies(Y[k], units, Y, units.b_hh.values)
        cs_ = summarize(cs_event(Y[k], G, WTS["household"], "never", {"Gc": Gc_listed, "upstate": ups}))
        log(f"  {SHORT[k]:24s} Sun-Abraham {sa_['post'][0]:.5f} | Callaway-Sant'Anna never-treated {cs_['post'][0]:.5f}")
    tex_table(os.path.join(tabdir, "D2_parallel_analysis_specifications.tex"),
              "Documentation: specifications of the parallel analysis, average 0 to 2 years after entry",
              "tab:docparallel", [""] + [HDR[k] for k in tiers] + ["Pre-trend $p$"], rows,
              "l|" + "|".join(["c"] * (len(tiers) + 1)),
              "Not part of the paper. These specifications treat undated counties as never treated, so their "
              "comparison group includes counties that received ride-hailing. All counties: every U.S. county, "
              "non-metro counties as their own clusters. Pooled event-study TWFE: relative-time indicators pooled "
              "across cohorts, county-clustered. Sun and Abraham: cohort-specific event-time indicators, "
              "never-treated comparison, MSA-clustered, cohorts weighted by households. Main design: as in the "
              "main table. Medians over all U.S. counties. * $p<0.10$; ** $p<0.05$; *** $p<0.01$.")

    # ------------------------------------------------------------------ documentation: earlier design
    log.section("DOCUMENTATION D1: EARLIER DESIGN (TREATED ONLY, MSA DUMMIES) IS NOT IDENTIFIED")
    old, ncol, rank = old_design_normalization(m)
    log(f"Design columns {ncol}, rank {rank}")
    rows = []
    for k in ["s0", "s2"]:
        for lab in ["R default", "Alternative"]:
            b = old[(lab, k)]
            rows.append([f"{SHORT[k]}, {lab}"] + [f"{b[e]:.3f}" for e in [-3, -2, 0, 1, 2, 3]])
            log(f"{SHORT[k]:15s} {lab:12s} " + " ".join(f"e{e:+d}: {b[e]:.3f}" for e in [-3, -2, 0, 1, 2, 3]))
    tex_table(os.path.join(tabdir, "D1_old_design_normalization.tex"),
              "Documentation: event-time coefficients of the earlier design under two equally valid normalizations",
              "tab:olddesign", ["", "3 before", "2 before", "Entry", "1 after", "2 after", "3 after"], rows,
              "l|c|c|c|c|c|c",
              "Not part of the paper. Treated counties only, MSA and year dummies, event-time dummies and "
              "time-varying controls. The event-time dummies are collinear with the MSA and year dummies, so a "
              "second reference period must be dropped. R drops the dummy for 5 years after entry (default); the "
              "alternative drops 7 years before entry.")

    pd.DataFrame(records).to_csv(os.path.join(tabdir, "all_estimates.csv"), index=False)
    log(f"  wrote {os.path.join(tabdir, 'all_estimates.csv')}")
    log.section("DONE")
    log(f"Elapsed: {time.time() - t0:.1f} s")
    log.write_tex(os.path.join(args.out, "run_log.tex"))
    print("Log written to", os.path.join(args.out, "run_log.tex"))


if __name__ == "__main__":
    main()
