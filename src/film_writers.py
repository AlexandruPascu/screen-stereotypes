"""Study 2 for films (and a TV replication), from IMDb's writer and director credits.

Writing team: the expected share of women among a title's credited writers (mean p_female from imdb_people.py; titles
need at least 80% of writers with a known first name). Because p_female is the expected gender given the name, using it
as the regressor keeps the slope unbiased (Berkson-type error); the coefficient reads as "all-women team vs all-men team".
Outcomes (exact gender from actor/actress credits): women's share of the credited cast, of the top-3 listed, and whether
the first-listed role is a woman's.
Designs:
  films A   same genre combination and decade (fixed effects), plus the director's expected gender, SE by director
  films B   same director (director fixed effects) plus decade: films by one director with different writing teams
  TV        same series and air year (fixed effects), SE by series — the IMDb version of study 2's TMDb result
Fiction only; adult titles left out. Output: results/behind_camera/writer_models.csv.
"""
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from paths import DERIVED, IMDB, RESULTS

OUT = RESULTS / "behind_camera"
con = duckdb.connect()
con.execute("SET memory_limit = '8GB'; SET preserve_insertion_order = false")


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{{cols}}})"


d = con.execute(f"""
    WITH b AS (SELECT tconst, titleType, isAdult = '1' AS adult, TRY_CAST(startYear AS INT) AS year, genres
               FROM {tsv('title.basics.tsv.gz', ['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
                                                 'startYear', 'endYear', 'runtimeMinutes', 'genres'])}),
    e AS (SELECT tconst, parentTconst FROM {tsv('title.episode.tsv.gz', ['tconst', 'parentTconst', 'seasonNumber',
                                                                         'episodeNumber'])}),
    t AS (SELECT b.tconst, COALESCE(e.parentTconst, b.tconst) AS work, COALESCE(b.year, w.year) AS year,
                 CASE WHEN b.titleType = 'movie' THEN 'film' WHEN b.titleType = 'tvEpisode' THEN 'tv' END AS medium,
                 COALESCE(w.genres, b.genres) AS genres, b.adult OR COALESCE(w.adult, false) AS adult
          FROM b LEFT JOIN e USING (tconst) LEFT JOIN b w ON w.tconst = e.parentTconst),
    p AS (SELECT nconst, p_female FROM '{DERIVED / "people.parquet"}'),
    crew AS (SELECT tconst, directors, writers FROM {tsv('title.crew.tsv.gz', ['tconst', 'directors', 'writers'])}
             WHERE writers IS NOT NULL),
    w AS (SELECT c.tconst, COUNT(*) AS writers, COUNT(p.p_female) AS writers_known, AVG(p.p_female) AS writers_women
          FROM (SELECT tconst, unnest(string_split(writers, ',')) AS nconst FROM crew) c LEFT JOIN p USING (nconst)
          GROUP BY c.tconst),
    dir AS (SELECT c.tconst, split_part(c.directors, ',', 1) AS director, p.p_female AS director_women
            FROM crew c LEFT JOIN p ON p.nconst = split_part(c.directors, ',', 1) WHERE c.directors IS NOT NULL),
    cast_ AS (SELECT tconst, COUNT(*) AS cast_n, AVG((category = 'actress')::INT) AS cast_women,
                     AVG((category = 'actress')::INT) FILTER (WHERE rank <= 3) AS top3_women,
                     MAX(((category = 'actress') AND rank = 1)::INT) AS first_woman
              FROM (SELECT tconst, category, ROW_NUMBER() OVER (PARTITION BY tconst ORDER BY TRY_CAST(ordering AS INT))
                           AS rank
                    FROM {tsv('title.principals.tsv.gz', ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters'])}
                    WHERE category IN ('actor', 'actress'))
              GROUP BY tconst)
    SELECT t.tconst, t.work, t.year, t.medium, t.genres, w.writers, w.writers_known, w.writers_women, dir.director,
           dir.director_women, c.cast_n, c.cast_women, c.top3_women, c.first_woman
    FROM t JOIN w USING (tconst) JOIN cast_ c USING (tconst) LEFT JOIN dir USING (tconst)
    WHERE t.medium IS NOT NULL AND NOT t.adult AND t.year BETWEEN 1950 AND 2026 AND c.cast_n >= 2
      AND NOT regexp_matches(COALESCE(t.genres, ''), 'Documentary|Reality-TV|Talk-Show|Game-Show|News')
      AND w.writers_known >= 0.8 * w.writers""").df()
d["decade"] = (d.year // 10 * 10).astype(int)
print(d.groupby("medium").agg(titles=("tconst", "size"), writers_women=("writers_women", "mean"),
                              cast_women=("cast_women", "mean")).round(3).to_string(), flush=True)


def fe_ols(df, y, x, fes, cluster):
    """OLS of y on x with the given fixed effects absorbed by alternating demeaning; SE clustered."""
    df = df.dropna(subset=[y] + x + fes + [cluster]).copy()
    for fe in fes:
        df = df[df.groupby(fe)[y].transform("size") > 1]
    Z = df[[y] + x].astype(float)
    for _ in range(12):
        for fe in fes:
            Z = Z - Z.groupby(df[fe].to_numpy()).transform("mean")
    res = sm.OLS(Z[y], Z[x]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]})
    return res, len(df), df[cluster].nunique()


rows = []
films = d[d.medium == "film"].copy()
films["genre_decade"] = films.genres.fillna("none") + "|" + films.decade.astype(str)
films["director"] = films.director.fillna("none")
tv = d[d.medium == "tv"].copy()
for outcome in ["cast_women", "top3_women", "first_woman"]:
    for label, df, x, fes, cluster in [
        ("films: same genre and decade", films, ["writers_women", "director_women"], ["genre_decade"], "director"),
        ("films: same director", films[films.director != "none"], ["writers_women"], ["director", "decade"], "director"),
        ("TV episodes: same series and year", tv, ["writers_women"], ["work", "year"], "work"),
    ]:
        res, n, groups = fe_ols(df, outcome, x, fes, cluster)
        for var in x:
            lo, hi = res.conf_int().loc[var]
            rows.append((label, outcome, var, res.params[var], lo, hi, res.pvalues[var], n, groups))
        print(f"{label} / {outcome}: " + ", ".join(f"{v} {res.params[v]:+.4f} [{res.conf_int().loc[v, 0]:+.4f}, "
                                                    f"{res.conf_int().loc[v, 1]:+.4f}]" for v in x) + f" (n={n:,})",
              flush=True)
m = pd.DataFrame(rows, columns=["design", "outcome", "variable", "estimate", "lo", "hi", "p", "titles", "clusters"])
m.to_csv(OUT / "writer_models.csv", index=False)
