"""Study 12 for series: are series that open with stereotyped portrayals renewed less often, and do they draw fewer
Emmy and Golden Globe nominations?

Series: the finished scripted series scored by score_series.py (first season; first aired 1990–2022; at least 500
IMDb votes). Outcomes: renewed beyond the first season, and log number of seasons (TMDb); a Primetime Emmy or a
Golden Globe nomination for the series or a performance in it (wiki_awards.py). Measures: the first season's mean
classifier probabilities (study 9: any stereotyped portrayal; study 11: one of the six groups appears, and one is
stereotyped with their presence held fixed), with the summaries' length held fixed, and the share of women among the
creators (TMDb).
Designs: series on the same network that started the same year (network and year fixed effects); series of the same
genre combination, start decade and original language. Standard errors clustered by network (a series without one is
its own cluster).
Output: results/reception/series_models.csv.
"""
import numpy as np
import pandas as pd

from film_models import fe_ols
from paths import DERIVED, RESULTS
from series_table import first_seasons

OUT = RESULTS / "reception"
ss = first_seasons()
ss["renewed"] = (ss.seasons >= 2).astype(float).where(ss.seasons.notna())
ss["log_seasons"] = np.log(ss.seasons.where(ss.seasons >= 1))
wa = pd.read_parquet(DERIVED / "wikipedia_awards.parquet").dropna(subset=["imdb"])
ss["emmy_nominated"] = ss.series.isin(wa[wa.award == "Primetime Emmys"].imdb).astype(float)
ss["globe_nominated"] = ss.series.isin(wa[wa.award == "Golden Globes (TV)"].imdb).astype(float)
print(f"series: {len(ss):,}; renewed {ss.renewed.mean():.1%}; Emmy-nominated {ss.emmy_nominated.mean():.1%}; "
      f"Golden Globe-nominated {ss.globe_nominated.mean():.1%}; creators' gender known "
      f"{ss.creators_women_share.notna().sum():,}", flush=True)

OUTCOMES = ["renewed", "log_seasons", "emmy_nominated", "globe_nominated"]
MEASURES = {"any_stereotype": ["log_words"], "social_any_group": ["log_words"],
            "social_any_stereotyped": ["social_any_group", "log_words"], "creators_women_share": []}
DESIGNS = [("same network and start year", ["network", "start"]),
           ("same genre and language", ["genre_decade_language"])]
rows = []
for y in OUTCOMES:
    for x, controls in MEASURES.items():
        for design, fes in DESIGNS:
            r = fe_ols(ss, y, x, fes, "series_cluster", controls)
            if r is not None:
                rows.append(dict(outcome=y, measure=x, design=design, outcome_mean=ss[y].mean(), **r))
                print(f"{y:16s} {x:24s} {design:28s} {r['estimate']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] "
                      f"series {r['films']:,}", flush=True)
pd.DataFrame(rows).rename(columns={"films": "series"}).to_csv(OUT / "series_models.csv", index=False)
