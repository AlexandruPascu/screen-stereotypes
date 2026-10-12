"""Does who writes (or directs) an episode change who is cast in it?

Unit: an episode of a scripted series on TMDb, with every credited writer's gender known and at least two guest stars of
known gender. Outcome: women's share of the guest cast (guests are cast per episode; series regulars are not counted).
Comparison: episodes of the same series written by women only, by mixed teams, or by men only (series fixed effects,
plus air-year fixed effects; standard errors clustered by series). Second outcome: among guest roles with a job in the
character name, whether a woman got a job that screen convention gives men (or a man a job it gives women).

Writes aggregate tables and figures to results/writers/.
"""
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from occupations import occupation
from paths import DERIVED, RESULTS

OUT = RESULTS / "writers"
OUT.mkdir(parents=True, exist_ok=True)
ep = pd.read_parquet(DERIVED / "tmdb_episodes.parquet")
ep = ep[ep.tv_type.isin(["Scripted", "Miniseries"]) & ep.year.between(1950, 2026) & (ep.season > 0)]
ep = ep[(ep.writers_0 == 0) & (ep.writers_3 == 0) & (ep.writers_1 + ep.writers_2 > 0)]
ep["guests_known"] = ep.guests_1 + ep.guests_2
ep = ep[ep.guests_known >= 2].copy()
ep["female_guests"] = ep.guests_1 / ep.guests_known
ep["team"] = np.select([ep.writers_2 == 0, ep.writers_1 == 0], ["women only", "men only"], "mixed")
ep["director"] = np.select([(ep.directors_1 > 0) & (ep.directors_2 == 0), (ep.directors_2 > 0) & (ep.directors_1 == 0)],
                           ["woman", "man"], "unknown or mixed")
ep["english"] = ep.language == "en"
print(f"episodes in the comparison: {len(ep):,} from {ep.series.nunique():,} series; writing teams: "
      + ", ".join(f"{k} {v:,}" for k, v in ep.team.value_counts().items()))


def within_series(d, y, x_cols, label):
    """OLS of y on x with series and year fixed effects (two-way demeaning), SE clustered by series."""
    d = d.copy()
    X = pd.get_dummies(d[x_cols], drop_first=False).astype(float)
    X = X[[c for c in X.columns if c not in ("team_men only", "director_man")]]
    X = X.join(pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float))
    Z = X.join(d[[y]])
    for _ in range(10):  # alternating projections: remove series means and year means
        Z = Z - Z.groupby(d.series).transform("mean")
    res = sm.OLS(Z[y], Z.drop(columns=[y])).fit(cov_type="cluster", cov_kwds={"groups": d.series})
    keep = [c for c in X.columns if not c.startswith("y_")]
    out = pd.DataFrame({"estimate": res.params[keep], "lo": res.conf_int()[0][keep], "hi": res.conf_int()[1][keep],
                        "p": res.pvalues[keep]})
    out["model"], out["episodes"], out["series"] = label, len(d), d.series.nunique()
    return out


# Only series with both women-only/mixed and men-only episodes carry information on the writer comparison.
var = ep.groupby("series").team.nunique()
mixed_series = var[var > 1].index
print(f"series with episodes by more than one kind of writing team: {len(mixed_series):,} "
      f"({ep.series.isin(mixed_series).sum():,} episodes)")
models = pd.concat([
    within_series(ep, "female_guests", ["team"], "all scripted series"),
    within_series(ep[ep.english], "female_guests", ["team"], "English-language series"),
    within_series(ep, "female_guests", ["team", "director"], "all, with director gender"),
    within_series(ep[ep.year >= 2010], "female_guests", ["team"], "episodes aired 2010+"),
])
models.to_csv(OUT / "guest_share_models.csv")
print(models.round(4).to_string())

desc = ep.assign(decade=(ep.year // 10) * 10).groupby(["decade", "team"]).agg(
    episodes=("female_guests", "size"), female_guests=("female_guests", "mean")).reset_index()
desc.to_csv(OUT / "guest_share_by_decade_and_team.csv", index=False)
creators = ep.assign(creator=ep.creator_genders.fillna("").map(
    lambda s: "none listed" if not s else "women only" if set(s.split("|")) == {"1"} else
    "men only" if set(s.split("|")) == {"2"} else "mixed or unknown"))
cr = creators.groupby("creator").agg(episodes=("female_guests", "size"), series=("series", "nunique"),
                                      female_guests=("female_guests", "mean")).reset_index()
cr.to_csv(OUT / "guest_share_by_creator.csv", index=False)
print(cr.to_string(index=False))

# Second outcome: counter-stereotypical job casting among guest roles.
skew = pd.read_csv(RESULTS / "occupations/occupation_by_era_english.csv")
skew = skew[skew.era == "2010-26"].set_index("occupation")["skew"]
g = pd.read_parquet(DERIVED / "tmdb_guest_roles.parquet")
g = g[g.gender.isin([1, 2]) & (g.character != "")]
g["name"] = (g.character.str.lower().str.replace(r"\([^)]*\)", " ", regex=True)
             .str.replace(r"#\s*\d+|\b\d+\b", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
uniq = pd.Series(g.name.unique())
jobs = dict(zip(uniq, uniq.map(occupation)))
g["occupation"] = g.name.map(jobs)
g = g.dropna(subset=["occupation"])
g["typed"] = g.occupation.map(skew)
g = g[g.typed.abs() >= 0.5]  # clearly gender-typed jobs only
g["counter"] = ((g.typed < 0) & (g.gender == 1)) | ((g.typed > 0) & (g.gender == 2))
g = g.merge(ep[["series", "season", "episode", "team", "director", "year", "english"]], on=["series", "season", "episode"])
print(f"guest roles with a gender-typed job, in comparison episodes: {len(g):,}; counter-stereotypical {g.counter.mean():.1%}")
cm = pd.concat([
    within_series(g.assign(counter=g.counter.astype(float)), "counter", ["team"], "counter-stereotypical job casting"),
    within_series(g[g.english].assign(counter=g.counter.astype(float)), "counter", ["team"], "same, English-language"),
])
cm.to_csv(OUT / "counter_stereotype_models.csv")
print(cm.round(4).to_string())
g.groupby(["team"]).agg(roles=("counter", "size"), counter=("counter", "mean")).to_csv(OUT / "counter_stereotype_by_team.csv")

fig, ax = plt.subplots(figsize=(9, 5))
for team, colour in [("women only", "#c0392b"), ("mixed", "#8e44ad"), ("men only", "#2c3e50")]:
    d = desc[(desc.team == team) & (desc.episodes >= 200)]
    ax.plot(d.decade, d.female_guests, marker="o", color=colour, label=f"written by {team}")
ax.set_ylabel("women's share of the episode's guest cast")
ax.set_title("Guest casting by the gender of the episode's writers (TMDb, scripted series)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "guest_share_by_writers.png", dpi=150)
