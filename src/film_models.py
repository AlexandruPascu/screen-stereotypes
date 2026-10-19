"""Fixed-effects regressions shared by the film studies (ratings_films.py, reception.py)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fe_ols(df, y, x, fes, cluster, controls=()):
    """OLS of y on x (and controls) with the fixed effects in `fes` swept out by alternating demeaning; None if too
    few films."""
    df = df.dropna(subset=[y, x] + fes + list(controls))
    for _ in range(3):  # singleton groups carry no information
        for fe in fes:
            df = df[df.groupby(fe)[y].transform("size") > 1]
    df = df[df.groupby(fes[0])[x].transform("std") > 0]  # only groups where the measure varies
    if len(df) < 500 or df[cluster].nunique() < 50:
        return None
    Z = df[[y, x] + list(controls)].astype(float)
    keys = [df[fe].to_numpy() for fe in fes]
    for _ in range(200):
        before = Z.copy()
        for k in keys:
            Z = Z - Z.groupby(k).transform("mean")
        if (Z - before).abs().to_numpy().max() < 1e-9:
            break
    R = Z[[y, x]].to_numpy()
    if controls:  # partial the controls out by least squares, which copes with collinear dummies (Frisch-Waugh)
        C = Z[list(controls)].to_numpy()
        R = R - C @ np.linalg.lstsq(C, R, rcond=None)[0]
    res = sm.OLS(R[:, 0], R[:, 1]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]})
    lo, hi = res.conf_int()[0]
    return dict(estimate=res.params[0], lo=lo, hi=hi, p=res.pvalues[0], films=len(df), clusters=df[cluster].nunique(),
                measure_mean=df[x].mean())
