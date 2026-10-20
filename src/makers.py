"""Study 13: who makes stereotyped films and series, and where.

Outcomes: the label-trained classifiers' probability that a summary portrays a stereotype (study 9: any, a gender
stereotype, a group stereotype; study 11: one of the six groups portrayed through a stereotype), for every rated
feature film (score_films.py) and for each scored series' first season (score_series.py).
Who: the director's and the writing team's expected share of women (imdb_people.py) for films, and the share of women
among a series' creators (TMDb) for series; compared as in study 4 (same genre, decade and language; for the writers
also a director's own films), with the summary's length (and for films its source) held fixed; for gender
stereotypes also with the women's share of the credited cast held fixed (series: of the scored episodes).
Where: rates by production country, studio, budget (2025 dollars) and streaming origin for films, and by network for
series, each with genre and decade held fixed (and summary length): the residual of the probability after those,
added back to the overall rate, averaged per group.
Outputs: results/makers/{who,where}.csv.
"""
import json

import numpy as np
import pandas as pd

from film_models import fe_ols
from paths import DATA, DERIVED, RESULTS
from series_table import first_seasons

OUT = RESULTS / "makers"
OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_parquet(DERIVED / "film_table.parquet")
cols = json.load(open(DERIVED / "film_table_columns.json"))
GENRES, LENGTH = cols["genres"], cols["length"]
fs = pd.read_parquet(DERIVED / "film_classifier_scores.parquet").rename(columns={"title": "tconst"})
f = f.drop(columns=["stereotyped_portrayal", "social_presence", "social_portrayal"]).merge(
    fs[["tconst", "any_stereotype", "gender_any", "group_stereotyped", "social_any_group", "social_any_stereotyped"]],
    on="tconst", how="left")
tm = pd.read_parquet(DERIVED / "tmdb_titles.parquet")
films_tm = tm[tm.kind == "film"].drop_duplicates("imdb").rename(columns={"imdb": "tconst"})
f = f.merge(films_tm[["tconst", "country", "companies", "budget", "release_date"]], on="tconst", how="left")
f["genre_decade"] = f.genres.fillna("none") + "|" + f.decade.astype(str)
cpi = pd.read_csv(DATA / "cpi/CPIAUCNS.csv", parse_dates=["observation_date"])
cpi = cpi.groupby(cpi.observation_date.dt.year).CPIAUCNS.mean()
f["budget_2025"] = (f.budget * (cpi.loc[2025] / f.year.map(cpi))).where(f.budget >= 1e4)
OUTCOMES = {"any_stereotype": "any stereotyped portrayal", "gender_any": "a gender stereotype",
            "group_stereotyped": "a group stereotype", "social_any_stereotyped": "one of the six groups stereotyped"}

# --- who: women behind the camera ------------------------------------------------------------------------------------
DESIGNS = [("same genre and language", ["genre_decade_language"], "director_cluster", []),
           ("same director", ["director", "year"], "director", GENRES),
           ("same franchise", ["collection", "year"], "collection", GENRES)]
rows = []
for y in OUTCOMES:
    for x in ["director_women", "writers_women"]:
        for design, fes, cluster, controls in DESIGNS:
            r = fe_ols(f, y, x, fes, cluster, controls + LENGTH)
            if r is not None:
                rows.append(dict(medium="film", outcome=y, measure=x, design=design, outcome_mean=f[y].mean(), **r))
                print(f"film {y:24s} {x:16s} {design:24s} {r['estimate'] * 100:+.2f} points [{r['lo'] * 100:+.2f}, "
                      f"{r['hi'] * 100:+.2f}] films {r['films']:,}", flush=True)
            if y == "gender_any" and design != "same franchise":
                # films by women more often centre on women, which gives more occasions for a gender stereotype; the
                # same comparison with the share of women in the credited cast held fixed
                r = fe_ols(f, y, x, fes, cluster, controls + LENGTH + ["cast_women"])
                if r is not None:
                    rows.append(dict(medium="film", outcome=y, measure=x, design=design + ", cast held fixed",
                                     outcome_mean=f[y].mean(), **r))
                    print(f"film {y:24s} {x:16s} {design + ', cast held fixed':24s} {r['estimate'] * 100:+.2f} points "
                          f"[{r['lo'] * 100:+.2f}, {r['hi'] * 100:+.2f}] films {r['films']:,}", flush=True)


def adjusted(d, y, group, fes, controls):
    """Mean of y per group after the fixed effects and controls, put back on the scale of y's overall mean."""
    d = d.dropna(subset=[y] + fes + controls)
    Z = d[[y] + controls].astype(float)
    for _ in range(50):
        before = Z.copy()
        for fe in fes:
            Z = Z - Z.groupby(d[fe].to_numpy()).transform("mean")
        if (Z - before).abs().to_numpy().max() < 1e-9:
            break
    r = Z[y].to_numpy()
    if controls:
        C = Z[controls].to_numpy()
        r = r - C @ np.linalg.lstsq(C, r, rcond=None)[0]
    res = pd.DataFrame({"group": d[group].to_numpy(), "r": r}).groupby("group").r.agg(["mean", "std", "size"])
    return pd.DataFrame({"rate": d[y].mean() + res["mean"], "lo": d[y].mean() + res["mean"] - 1.96 * res["std"] /
                         np.sqrt(res["size"]), "hi": d[y].mean() + res["mean"] + 1.96 * res["std"] / np.sqrt(res["size"]),
                         "titles": res["size"]})


# --- where: country, studio, streaming, budget ----------------------------------------------------------------------
STREAMERS = r"(?i)\b(netflix|amazon studios|amazon mgm|amazon original|apple studios|apple original|hulu|disney\+)"
f["streaming_original"] = np.where(f.companies.fillna("").str.contains(STREAMERS, regex=True), "streaming original",
                                   "other")
f["studio"] = f.companies.fillna("").str.split("|").str[0].replace("", np.nan)
top_studios = f[f.year >= 1970].studio.value_counts().head(25).index
f["studio_top"] = f.studio.where(f.studio.isin(top_studios))
top_countries = f.country.value_counts().head(20).index
f["country_top"] = f.country.where(f.country.isin(top_countries))
f["budget_band"] = pd.cut(f.budget_2025, [0, 1e6, 1e7, 5e7, 1e8, 1e12],
                          labels=["under $1m", "$1–10m", "$10–50m", "$50–100m", "over $100m"]).astype("string")
where = []
for group, fes, scope in [("country_top", ["genre_decade"], f), ("studio_top", ["genre_decade"], f[f.year >= 1970]),
                          ("streaming_original", ["genre_decade"], f[f.year >= 2015]),
                          ("budget_band", ["genre_decade"], f)]:
    for y in ["any_stereotype", "social_any_stereotyped"]:
        a = adjusted(scope.dropna(subset=[group]), y, group, fes, LENGTH)
        a = a[a.titles >= 100].reset_index().rename(columns={"group": "value"})
        where.append(a.assign(medium="film", grouping=group, outcome=y))
where = pd.concat(where, ignore_index=True)

# --- series: creators and networks (needs score_series.py) ---------------------------------------------------------
if (DERIVED / "series_classifier_scores.parquet").exists():
    ss = first_seasons(cast=True)
    for y in OUTCOMES:
        variants = [("same genre and language", ["log_words"])]
        if y == "gender_any":  # as for films: series created by women may centre more on women
            variants.append(("same genre and language, cast held fixed", ["log_words", "cast_women"]))
        for design, controls in variants:
            r = fe_ols(ss, y, "creators_women_share", ["genre_decade_language"], "series_cluster", controls)
            if r is not None:
                rows.append(dict(medium="series", outcome=y, measure="creators_women_share", design=design,
                                 outcome_mean=ss[y].mean(), **r))
                print(f"series {y:24s} creators_women_share {design:24s} {r['estimate'] * 100:+.2f} points "
                      f"[{r['lo'] * 100:+.2f}, {r['hi'] * 100:+.2f}] series {r['films']:,}", flush=True)
    top_networks = ss.network.value_counts().head(25).index
    ss["network_top"] = ss.network.where(ss.network.isin(top_networks))
    for y in ["any_stereotype", "social_any_stereotyped"]:
        a = adjusted(ss.dropna(subset=["network_top"]), y, "network_top", ["genre_decade"], ["log_words"])
        a = a[a.titles >= 50].reset_index().rename(columns={"group": "value"})
        where = pd.concat([where, a.assign(medium="series", grouping="network_top", outcome=y)], ignore_index=True)

pd.DataFrame(rows).rename(columns={"films": "titles"}).to_csv(OUT / "who.csv", index=False)
where.to_csv(OUT / "where.csv", index=False)
pd.set_option("display.width", 200)
for (medium, grouping, outcome), d in where.groupby(["medium", "grouping", "outcome"]):
    if outcome == "any_stereotype":
        print(f"== {medium} by {grouping} (any stereotyped portrayal, %):")
        print("   " + "; ".join(f"{v} {r * 100:.1f} (n={n:,})" for v, r, n in
                                d.sort_values("rate")[["value", "rate", "titles"]].itertuples(index=False)))
