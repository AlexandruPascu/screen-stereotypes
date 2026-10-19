"""Do audiences rate episodes differently when they cast or write women and groups differently? (2022 question, redone)

Unit: a rated IMDb episode (2026 ratings). Each measure is compared within its own series (series and air-year fixed
effects, standard errors clustered by series), so the estimate says: among episodes of the same show, is an episode
with more of X rated higher or lower? Outcomes: average rating (1-10), log(1 + number of votes), and for episodes
whose Wikipedia episode table gives one, log viewers (millions; Nielsen for US series, BARB for British ones).

Measures (each model uses the episodes where that measure exists):
  cast_women_share        women's share of the episode's credited cast (IMDb principals, actor/actress)
  guest_women_share       women's share of the episode's TMDb guest stars (aligned series only)
  summary_women_active    women's share of the active subjects in the plot summary
  summary_women_looks     the summary describes a woman by her appearance (0/1)
  summary_women_victim    the summary has a woman as the victim of violence (0/1)
  group_stereotype        the summary mentions a group together with that group's stereotype words (0/1)
  offensive_stereotype    the same with only the pejorative stereotype words (0/1)
  slur                    the summary contains a slur (0/1)
  social_named            the summary names one of study 11's six groups (immigrants, disability, mental illness,
                          older people, poverty, homelessness) (0/1)
  social_stereotype       it names one of them together with that group's stereotype words (0/1)
  social_pejorative       the same with only the pejorative stereotype words (0/1)
  social_label            it calls one of them by an insulting label (0/1)
The summary measures come from each episode's longest summary, and popular episodes have longer, more detailed ones, so
their models also hold the summary's length (log words) and source fixed.
Writes results/ratings/models.csv. Correlational: the 2022 null result used a measure that did not measure stereotypes.
"""
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from paths import DERIVED, IMDB, RESULTS

OUT = RESULTS / "ratings"
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit = '8GB'")


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{{cols}}})"


ep = con.execute(f"""
    WITH e AS (SELECT tconst, parentTconst AS series, TRY_CAST(seasonNumber AS INT) AS season,
                      TRY_CAST(episodeNumber AS INT) AS episode
               FROM {tsv('title.episode.tsv.gz', ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber'])}),
    r AS (SELECT tconst, CAST(averageRating AS DOUBLE) AS rating, CAST(numVotes AS INT) AS votes
          FROM {tsv('title.ratings.tsv.gz', ['tconst', 'averageRating', 'numVotes'])}),
    b AS (SELECT tconst, TRY_CAST(startYear AS INT) AS year FROM {tsv('title.basics.tsv.gz', ['tconst', 'titleType',
          'primaryTitle', 'originalTitle', 'isAdult', 'startYear', 'endYear', 'runtimeMinutes', 'genres'])}),
    cast_ AS (SELECT tconst, AVG((category = 'actress')::INT) AS cast_women_share, COUNT(*) AS cast_n
              FROM {tsv('title.principals.tsv.gz', ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters'])}
              WHERE category IN ('actor', 'actress') GROUP BY tconst)
    SELECT e.*, r.rating, r.votes, b.year, c.cast_women_share, c.cast_n
    FROM e JOIN r USING (tconst) JOIN b USING (tconst) LEFT JOIN cast_ c USING (tconst)
    WHERE b.year BETWEEN 1950 AND 2026""").df()
print(f"rated episodes: {len(ep):,} in {ep.series.nunique():,} series", flush=True)

# TMDb guest casting, attached only through series whose numbering lines up with IMDb's
aligned = pd.read_parquet(DERIVED / "series_alignment.parquet")
aligned = set(aligned[aligned.aligned].imdb)
te = pd.read_parquet(DERIVED / "tmdb_episodes.parquet", columns=["imdb", "season", "episode", "guests_1", "guests_2"])
te = te[te.imdb.isin(aligned) & ((te.guests_1 + te.guests_2) >= 2)]
te["guest_women_share"] = te.guests_1 / (te.guests_1 + te.guests_2)
ep = ep.merge(te[["imdb", "season", "episode", "guest_women_share"]].drop_duplicates(["imdb", "season", "episode"]),
              left_on=["series", "season", "episode"], right_on=["imdb", "season", "episode"], how="left")

tg = pd.read_parquet(DERIVED / "title_gender_text.parquet")
tg["summary_women_active"] = tg.women_active / (tg.women_active + tg.men_active).replace(0, np.nan)
tg["summary_women_looks"] = (tg.women_appearance > 0).astype(float).where(tg.women_mentions > 0)
tg["summary_women_victim"] = (tg.women_victim > 0).astype(float).where(tg.women_mentions > 0)
ep = ep.merge(tg[["title", "summary_women_active", "summary_women_looks", "summary_women_victim"]],
              left_on="tconst", right_on="title", how="left")
gm = pd.read_parquet(DERIVED / "title_groups.parquet")
gm["group_stereotype"] = ((gm.mention_mask & gm.stereotype_mask) != 0).astype(float)
gm["slur"] = (gm.slur_mask != 0).astype(float)
ep = ep.merge(gm[["title", "group_stereotype", "slur"]].rename(columns={"title": "t2"}), left_on="tconst",
              right_on="t2", how="left")
off = pd.read_parquet(DERIVED / "title_groups_offensive.parquet")
off["offensive_stereotype"] = ((off.mention_mask & off.stereotype_mask) != 0).astype(float)
ep = ep.merge(off[["title", "offensive_stereotype"]].rename(columns={"title": "t3"}), left_on="tconst", right_on="t3",
              how="left")
sg = pd.read_parquet(DERIVED / "title_groups_social.parquet")
sg["social_named"] = (sg.mention_mask != 0).astype(float)
sg["social_stereotype"] = ((sg.mention_mask & sg.stereotype_mask) != 0).astype(float)
sg["social_label"] = (sg.slur_mask != 0).astype(float)
so = pd.read_parquet(DERIVED / "title_groups_social_offensive.parquet")
so["social_pejorative"] = ((so.mention_mask & so.stereotype_mask) != 0).astype(float)
sg = sg.merge(so[["title", "social_pejorative"]], on="title", how="left")
ep = ep.merge(sg[["title", "social_named", "social_stereotype", "social_pejorative", "social_label"]].rename(
    columns={"title": "t4"}), left_on="tconst", right_on="t4", how="left")
ep["log_votes"] = np.log1p(ep.votes)
viewers = DERIVED / "wiki_viewers.parquet"
if viewers.exists():
    wv = pd.read_parquet(viewers, columns=["tconst", "viewers"]).drop_duplicates("tconst")
    ep = ep.merge(wv, on="tconst", how="left")
    ep["log_viewers"] = np.log(ep.viewers)
tl = con.execute(f"""SELECT title AS tconst, max(words) AS summary_words,
    arg_max(source, (words, text, source)) AS summary_source  -- one text can come from several sources
    FROM '{DERIVED / "texts.parquet"}' WHERE words >= 5 GROUP BY title""").df()
ep = ep.merge(tl, on="tconst", how="left")
ep["log_summary_words"] = np.log1p(ep.summary_words)
SOURCES = pd.get_dummies(ep.summary_source, prefix="source", drop_first=True, dtype=float)
ep = pd.concat([ep, SOURCES], axis=1)
LENGTH = ["log_summary_words"] + list(SOURCES.columns)
FROM_SUMMARY = {"summary_women_active", "summary_women_looks", "summary_women_victim", "group_stereotype",
                "offensive_stereotype", "slur", "social_named", "social_stereotype", "social_pejorative", "social_label"}


def within(d, y, x, controls=()):
    """y on x within series, with air-year dummies and controls partialled out by least squares (Frisch-Waugh), which
    copes with dummies that are collinear in a small sample; None with fewer than 50 series."""
    d = d.dropna(subset=[y, x, "year"] + list(controls))
    d = d[d.groupby("series")[x].transform("std") > 0]  # only series where the measure varies
    if d.series.nunique() < 50:
        return None
    Z = pd.get_dummies(d.year.astype(int), prefix="y", drop_first=True).astype(float)
    for c in controls:
        Z[c] = d[c].to_numpy()
    Z[x], Z[y] = d[x].to_numpy(), d[y].to_numpy()
    Z = Z - Z.groupby(d.series.to_numpy()).transform("mean")
    R = Z[[y, x]].to_numpy()
    C = Z.drop(columns=[y, x]).to_numpy()
    R = R - C @ np.linalg.lstsq(C, R, rcond=None)[0]
    res = sm.OLS(R[:, 0], R[:, 1]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.series)[0]})
    lo, hi = res.conf_int()[0]
    return dict(measure=x, outcome=y, estimate=res.params[0], lo=lo, hi=hi, p=res.pvalues[0], episodes=len(d),
                series=d.series.nunique(), measure_mean=d[x].mean())


rows = []
for x in ["cast_women_share", "guest_women_share", "summary_women_active", "summary_women_looks",
          "summary_women_victim", "group_stereotype", "offensive_stereotype", "slur", "social_named",
          "social_stereotype", "social_pejorative", "social_label"]:
    for y in ["rating", "log_votes"] + (["log_viewers"] if "log_viewers" in ep.columns else []):
        if ep[[x, y]].notna().all(axis=1).sum() > 1000:  # enough episodes with both
            r = within(ep, y, x, LENGTH if x in FROM_SUMMARY else ())
            if r is not None:
                rows.append(r)
                print(r, flush=True)
models = pd.DataFrame(rows)
models.to_csv(OUT / "models.csv", index=False)
pd.set_option("display.width", 200)
print(models.round(4).to_string(index=False))
