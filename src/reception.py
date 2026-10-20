"""Study 12: money, critics and awards. Do box office, critics and the Academy treat stereotyped films and women on
screen the way IMDb voters do (study 4)?

Films: the film table of ratings_films.py (rated feature films, 1950–2026, with every measure and design variable).
Outcomes:
  box office  worldwide box office and budget from TMDb, as entered there (films with both of at least $10,000), in
              2025 dollars (US CPI-U, BLS, via FRED): log box office with log budget held fixed, and log return
              (box office over budget)
  critics     Rotten Tomatoes Tomatometer and average rating, and the Metascore: the films' own pages
              (reception_fetch.py) where fetched, else Wikidata's latest score (wikidata_reception.py)
  audiences   Rotten Tomatoes audience score and Metacritic user score (fetched pages), and the IMDb rating
  awards      the Academy Awards (Wikidata): a nomination of any kind for the film, and an acting nomination for a
              performance in it; Golden Globe and BAFTA nominations (Wikipedia's category pages, wiki_awards.py)
Measures: as in study 4 (women on screen, the two label-trained classifiers, a story about prejudice), and for the
"Oscar bait" question the six-groups classifier's probability that a disabled person or a person with a mental illness
appears.
Designs: as in study 4 (same genre, decade and language; a director's own films with release-year effects and genre
dummies; one franchise); measures from summaries with the summary's length and source held fixed.
Critics against audiences: the same stereotyped-portrayal model for each rating, in standard deviations of that
rating, on the films that have all the ratings of one site, so critics and audiences are compared on the same films.
Outputs: results/reception/{models,critics_vs_audiences,coverage}.csv.
"""
import json

import numpy as np
import pandas as pd

from film_models import fe_ols
from paths import DATA, DERIVED, RESULTS

OUT = RESULTS / "reception"
OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_parquet(DERIVED / "film_table.parquet")
cols = json.load(open(DERIVED / "film_table_columns.json"))
GENRES, LENGTH = cols["genres"], cols["length"]

# --- box office in 2025 dollars ---------------------------------------------------------------------------------------
cpi = pd.read_csv(DATA / "cpi/CPIAUCNS.csv", parse_dates=["observation_date"])
cpi = cpi.groupby(cpi.observation_date.dt.year).CPIAUCNS.mean()
tm = pd.read_parquet(DERIVED / "tmdb_titles.parquet", columns=["kind", "imdb", "budget", "revenue", "release_date"])
tm = tm[(tm.kind == "film") & (tm.budget >= 1e4) & (tm.revenue >= 1e4)].drop_duplicates("imdb")
tm["money_year"] = pd.to_datetime(tm.release_date, errors="coerce").dt.year
f = f.merge(tm[["imdb", "budget", "revenue", "money_year"]].rename(columns={"imdb": "tconst"}), on="tconst", how="left")
deflator = cpi.loc[2025] / f.money_year.fillna(f.year).map(cpi)
f["log_budget"] = np.log(f.budget * deflator)
f["log_box_office"] = np.log(f.revenue * deflator)
f["log_return"] = np.log(f.revenue / f.budget)

# --- critics and audiences ------------------------------------------------------------------------------------------
wd = pd.read_parquet(DERIVED / "wikidata/scores.parquet")
pick = {("rotten_tomatoes", "Tomatometer score"): "tomatometer",
        ("rotten_tomatoes", "Rotten Tomatoes average of rated reviews"): "rt_average", ("metacritic", "Metascore"): "metascore"}
wd["name"] = [pick.get((c, m)) for c, m in zip(wd.critic, wd.method)]
wd = wd.dropna(subset=["name"]).pivot_table(index="imdb", columns="name", values="value", aggfunc="last")
fetched = {}
for site, renames in [("rt", {"critic_score": "tomatometer", "critic_average": "rt_average",
                              "audience_score": "rt_audience"}),
                      ("mc", {"critic_score": "metascore", "audience_score": "mc_user"})]:
    path = DERIVED / f"reception/{site}_scores.parquet"
    if path.exists():
        d = pd.read_parquet(path)
        d = d[d.status == 200].drop_duplicates("imdb").set_index("imdb")
        fetched[site] = d[[c for c in renames if c in d.columns]].rename(columns=renames)
scores = wd.copy()
for d in fetched.values():  # a fetched page is fresher than Wikidata's copy
    scores = d.combine_first(scores) if len(scores) else d
scores = scores.reset_index().rename(columns={"index": "tconst", "imdb": "tconst"})
f = f.merge(scores, on="tconst", how="left")
f.loc[f.rt_average > 10, "rt_average"] = np.nan  # a handful of averages entered on another scale

# --- the Academy Awards ---------------------------------------------------------------------------------------------
aw = pd.read_parquet(DERIVED / "wikidata/awards.parquet")
aw = aw[(aw.family == "Academy Awards") & aw.film.notna()]
acting = aw.category.str.contains("Actor|Actress", case=False)
f["oscar_nominated"] = f.tconst.isin(aw.film).astype(float)
f["oscar_acting"] = f.tconst.isin(aw[acting].film).astype(float)
f["oscar_won"] = f.tconst.isin(aw[aw.outcome == "won"].film).astype(float)
wa = pd.read_parquet(DERIVED / "wikipedia_awards.parquet").dropna(subset=["imdb"])
f["globe_nominated"] = f.tconst.isin(wa[wa.award == "Golden Globes (film)"].imdb).astype(float)
f["bafta_nominated"] = f.tconst.isin(wa[wa.award == "BAFTA (film)"].imdb).astype(float)
for c in ["oscar_nominated", "oscar_acting", "oscar_won", "globe_nominated", "bafta_nominated"]:
    f.loc[f.year > 2025, c] = np.nan  # 2026 releases have not had their ceremony

fs = pd.read_parquet(DERIVED / "film_classifier_scores.parquet",
                     columns=["title", "social_m_disability", "social_m_mental_illness"]).rename(columns={"title": "tconst"})
f = f.merge(fs, on="tconst", how="left")

OUTCOMES = {"log_box_office": "box office (log, budget held fixed)", "log_return": "return on budget (log)",
            "tomatometer": "Tomatometer (0–100)", "rt_average": "critics' average rating (0–10)",
            "metascore": "Metascore (0–100)", "rt_audience": "Rotten Tomatoes audience score (0–100)",
            "mc_user": "Metacritic user score (0–10)", "rating": "IMDb rating (1–10)",
            "oscar_nominated": "any Oscar nomination", "oscar_acting": "an Oscar acting nomination",
            "oscar_won": "any Oscar win", "globe_nominated": "a Golden Globe nomination",
            "bafta_nominated": "a BAFTA nomination"}
MEASURES = ["cast_women", "first_woman", "stereotyped_portrayal", "social_presence", "social_portrayal",
            "prejudice_story", "social_m_disability", "social_m_mental_illness"]
FROM_SUMMARY = {"stereotyped_portrayal", "social_presence", "social_portrayal", "prejudice_story",
                "social_m_disability", "social_m_mental_illness"}
EXTRA = {"social_portrayal": ["social_presence"]}
DESIGNS = [("same genre and language", ["genre_decade_language"], "director_cluster", []),
           ("same director", ["director", "year"], "director", GENRES),
           ("same franchise", ["collection", "year"], "collection", GENRES)]

coverage = pd.DataFrame({"films": [f[y].notna().sum() for y in OUTCOMES]}, index=list(OUTCOMES))
coverage.to_csv(OUT / "coverage.csv")
print(coverage.T.to_string(), flush=True)

rows = []
for y in OUTCOMES:
    for x in MEASURES:
        for design, fes, cluster, controls in DESIGNS:
            c = controls + (LENGTH if x in FROM_SUMMARY else []) + EXTRA.get(x, []) + \
                (["log_budget"] if y == "log_box_office" else [])
            r = fe_ols(f, y, x, fes, cluster, c)
            if r is not None:
                rows.append(dict(outcome=y, measure=x, design=design, **r))
                print(f"{y:16s} {x:24s} {design:24s} {r['estimate']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                      f"films {r['films']:,}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "models.csv", index=False)

# critics against audiences, on the same films, in standard deviations of each rating
comparison = []
for site, ratings in [("Rotten Tomatoes", ["rating", "rt_audience", "tomatometer", "rt_average"]),
                      ("Metacritic", ["rating", "mc_user", "metascore"])]:
    ratings = [y for y in ratings if f[y].notna().sum() >= 1000]  # a site's audience score once enough are fetched
    common = f.dropna(subset=ratings)
    for y in ratings:
        z = common.assign(**{y: (common[y] - common[y].mean()) / common[y].std()})
        for design, fes, cluster, controls in DESIGNS[:2]:
            r = fe_ols(z, y, "stereotyped_portrayal", fes, cluster, controls + LENGTH)
            if r is not None:
                comparison.append(dict(site=site, rating=y, design=design, **r))
comparison = pd.DataFrame(comparison, columns=["site", "rating", "design", "estimate", "lo", "hi", "p", "films", "clusters",
                                               "measure_mean"])
comparison.to_csv(OUT / "critics_vs_audiences.csv", index=False)
pd.set_option("display.width", 200)
print(comparison[["site", "rating", "design", "estimate", "lo", "hi", "films"]].round(3).to_string(index=False)
      if len(comparison) else "critics against audiences: no films with audience scores yet")
