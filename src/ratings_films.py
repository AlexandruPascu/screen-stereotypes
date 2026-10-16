"""Study 4 for films: are films rated differently when they cast, write or describe women and groups differently?

Unit: a rated feature film (IMDb, at least MIN_VOTES votes, 1950–2026, documentaries and reality formats left out).
Films have no "other episodes of the same show", so each measure is compared in three ways:
  same genre and language films with the same IMDb genre combination, decade and original language (TMDb),
                          clustered by director
  same director           a director's own films, with release-year effects and IMDb genre dummies (clustered by
                          director)
  same franchise          films of one TMDb collection (sequels, series of films), with release-year effects and
                          genre dummies (clustered by collection)
Measures (each model uses the films where that measure exists):
  cast_women              women's share of the credited cast (IMDb principals, actor/actress)
  top3_women              women's share of the first three listed roles
  first_woman             a woman in the first-listed role (0/1)
  writers_women           the writing team's expected share of women (imdb_people.py; teams with >= 80% known)
  director_women          the (first) director's probability of being a woman
  summary_women_active    women's share of the active subjects in the plot summary
  summary_women_looks     the summary describes a woman by her appearance (0/1)
  summary_women_victim    the summary has a woman as the victim of violence (0/1)
  group_stereotype        the summary mentions a group together with that group's stereotype words (0/1)
  offensive_stereotype    the same with only the pejorative stereotype words (0/1)
  slur                    the summary contains a slur (0/1)
  prejudice_story         study 7's classifier puts the summary in its top 2% (mostly stories about prejudice)
  stereotyped_portrayal   study 9's label-trained classifier: probability of a stereotyped portrayal (scored sample)
  social_named, social_stereotype, social_pejorative, social_label
                          study 11's six groups: named; named with their stereotype words; with pejorative ones; an
                          insulting label (0/1)
  social_portrayal        study 11's label-trained classifier: probability that one of the six groups is portrayed
                          through a stereotype (scored sample)
The summary measures come from each film's longest summary, and well-known films have long, detailed ones, so their
models also hold the summary's length (log words) and source fixed; otherwise "the summary mentions X" partly
measures how much has been written about the film.
Outcomes: average rating (1–10) and log(1 + number of votes). Writes results/ratings/film_models.csv. Correlational.
"""
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from paths import DERIVED, IMDB, RESULTS

OUT = RESULTS / "ratings"
OUT.mkdir(parents=True, exist_ok=True)
MIN_VOTES = 20
con = duckdb.connect()
con.execute("SET memory_limit = '8GB'; SET preserve_insertion_order = false")


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{{cols}}})"


f = con.execute(f"""
    WITH b AS (SELECT tconst, TRY_CAST(startYear AS INT) AS year, genres
               FROM {tsv('title.basics.tsv.gz', ['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
                                                 'startYear', 'endYear', 'runtimeMinutes', 'genres'])}
               WHERE titleType = 'movie' AND isAdult = '0'
                 AND NOT regexp_matches(COALESCE(genres, ''), 'Documentary|Reality-TV|Talk-Show|Game-Show|News')),
    r AS (SELECT tconst, CAST(averageRating AS DOUBLE) AS rating, CAST(numVotes AS INT) AS votes
          FROM {tsv('title.ratings.tsv.gz', ['tconst', 'averageRating', 'numVotes'])}),
    p AS (SELECT nconst, p_female FROM '{DERIVED / "people.parquet"}'),
    crew AS (SELECT c.tconst, c.directors, c.writers FROM {tsv('title.crew.tsv.gz', ['tconst', 'directors', 'writers'])} c
             SEMI JOIN b USING (tconst)),
    w AS (SELECT c.tconst, COUNT(*) AS writers, COUNT(p.p_female) AS writers_known, AVG(p.p_female) AS writers_women
          FROM (SELECT tconst, unnest(string_split(writers, ',')) AS nconst FROM crew WHERE writers IS NOT NULL) c
          LEFT JOIN p USING (nconst) GROUP BY c.tconst),
    d AS (SELECT c.tconst, c.director, p.p_female AS director_women
          FROM (SELECT tconst, split_part(directors, ',', 1) AS director FROM crew WHERE directors IS NOT NULL) c
          LEFT JOIN p ON p.nconst = c.director),
    cast_ AS (SELECT tconst, AVG(woman) AS cast_women, AVG(woman) FILTER (WHERE rank <= 3) AS top3_women,
                     MAX(woman * (rank = 1)::INT) AS first_woman, COUNT(*) AS cast_n
              FROM (SELECT tconst, (category = 'actress')::INT AS woman,
                           ROW_NUMBER() OVER (PARTITION BY tconst ORDER BY TRY_CAST(ordering AS INT)) AS rank
                    FROM {tsv('title.principals.tsv.gz', ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters'])}
                    SEMI JOIN b USING (tconst) WHERE category IN ('actor', 'actress'))
              GROUP BY tconst)
    SELECT b.tconst, b.year, b.genres, r.rating, r.votes, d.director, d.director_women,
           CASE WHEN w.writers_known >= 0.8 * w.writers THEN w.writers_women END AS writers_women,
           c.cast_women, c.top3_women, c.first_woman, c.cast_n
    FROM b JOIN r USING (tconst) LEFT JOIN d USING (tconst) LEFT JOIN w USING (tconst) LEFT JOIN cast_ c USING (tconst)
    WHERE b.year BETWEEN 1950 AND 2026 AND r.votes >= {MIN_VOTES}""").df()
f.loc[f.cast_n < 2, ["cast_women", "top3_women", "first_woman"]] = np.nan  # one credited actor says little
f["decade"] = f.year // 10 * 10
f["log_votes"] = np.log1p(f.votes)

tmdb = pd.read_parquet(DERIVED / "tmdb_titles.parquet", columns=["kind", "imdb", "language", "collection"])
tmdb = tmdb[tmdb.kind == "film"].drop_duplicates("imdb")
f = f.merge(tmdb[["imdb", "language", "collection"]].rename(columns={"imdb": "tconst"}), on="tconst", how="left")
f["genre_decade_language"] = f.genres.fillna("none") + "|" + f.decade.astype(str) + "|" + f.language

tg = pd.read_parquet(DERIVED / "title_gender_text.parquet")
tg["summary_women_active"] = tg.women_active / (tg.women_active + tg.men_active).replace(0, np.nan)
tg["summary_women_looks"] = (tg.women_appearance > 0).astype(float).where(tg.women_mentions > 0)
tg["summary_women_victim"] = (tg.women_victim > 0).astype(float).where(tg.women_mentions > 0)
gm = pd.read_parquet(DERIVED / "title_groups.parquet")
gm["group_stereotype"] = ((gm.mention_mask & gm.stereotype_mask) != 0).astype(float)
gm["slur"] = (gm.slur_mask != 0).astype(float)
off = pd.read_parquet(DERIVED / "title_groups_offensive.parquet")
off["offensive_stereotype"] = ((off.mention_mask & off.stereotype_mask) != 0).astype(float)
tc = pd.read_parquet(DERIVED / "title_classifier_scores.parquet", columns=["title", "stereotype_prob"])
tc["prejudice_story"] = (tc.stereotype_prob >= tc.stereotype_prob.quantile(0.98)).astype(float)
ls = pd.read_parquet(DERIVED / "label_classifier_sample_scores.parquet", columns=["title", "any_stereotype"])
sg = pd.read_parquet(DERIVED / "title_groups_social.parquet")
sg["social_named"] = (sg.mention_mask != 0).astype(float)
sg["social_stereotype"] = ((sg.mention_mask & sg.stereotype_mask) != 0).astype(float)
sg["social_label"] = (sg.slur_mask != 0).astype(float)
so = pd.read_parquet(DERIVED / "title_groups_social_offensive.parquet")
so["social_pejorative"] = ((so.mention_mask & so.stereotype_mask) != 0).astype(float)
sg = sg.merge(so[["title", "social_pejorative"]], on="title", how="left")
ss = pd.read_parquet(DERIVED / "label_classifier_social_sample_scores.parquet", columns=["title", "any_stereotyped"])
for table, cols in [(tg, ["summary_women_active", "summary_women_looks", "summary_women_victim"]),
                    (gm, ["group_stereotype", "slur"]), (off, ["offensive_stereotype"]), (tc, ["prejudice_story"]),
                    (ls.rename(columns={"any_stereotype": "stereotyped_portrayal"}), ["stereotyped_portrayal"]),
                    (sg, ["social_named", "social_stereotype", "social_pejorative", "social_label"]),
                    (ss.rename(columns={"any_stereotyped": "social_portrayal"}), ["social_portrayal"])]:
    f = f.merge(table[["title"] + cols].drop_duplicates("title").rename(columns={"title": "tconst"}), on="tconst",
                how="left")
tl = con.execute(f"""SELECT title AS tconst, max(words) AS summary_words,
    arg_max(source, (words, text)) AS summary_source
    FROM '{DERIVED / "texts.parquet"}' WHERE words >= 5 GROUP BY title""").df()
f = f.merge(tl, on="tconst", how="left")
f["log_summary_words"] = np.log1p(f.summary_words)
SOURCES = pd.get_dummies(f.summary_source, prefix="source", drop_first=True, dtype=float)
f = pd.concat([f, SOURCES], axis=1)
LENGTH = ["log_summary_words"] + list(SOURCES.columns)
f["director_cluster"] = f.director.fillna(f.tconst)  # films without a known director stand alone
GENRES = f.genres.fillna("").str.get_dummies(sep=",").add_prefix("genre_").astype(float)
f = pd.concat([f, GENRES], axis=1)
GENRES = list(GENRES.columns)
print(f"rated feature films with >= {MIN_VOTES} votes, 1950-2026: {len(f):,}; with a director {f.director.notna().sum():,}; "
      f"with a TMDb language {f.language.notna().sum():,}; in a TMDb collection {f.collection.notna().sum():,}", flush=True)


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


DESIGNS = [("same genre and language", ["genre_decade_language"], "director_cluster", []),
           ("same director", ["director", "year"], "director", GENRES),
           ("same franchise", ["collection", "year"], "collection", GENRES)]
MEASURES = ["cast_women", "top3_women", "first_woman", "writers_women", "director_women", "summary_women_active",
            "summary_women_looks", "summary_women_victim", "group_stereotype", "offensive_stereotype", "slur",
            "prejudice_story", "stereotyped_portrayal", "social_named", "social_stereotype", "social_pejorative",
            "social_label", "social_portrayal"]
FROM_SUMMARY = {"summary_women_active", "summary_women_looks", "summary_women_victim", "group_stereotype",
                "offensive_stereotype", "slur", "prejudice_story", "stereotyped_portrayal", "social_named",
                "social_stereotype", "social_pejorative", "social_label", "social_portrayal"}
rows = []
for x in MEASURES:
    for y in ["rating", "log_votes"]:
        for design, fes, cluster, controls in DESIGNS:
            r = fe_ols(f, y, x, fes, cluster, controls + (LENGTH if x in FROM_SUMMARY else []))
            if r is None:
                continue
            rows.append(dict(measure=x, outcome=y, design=design, **r))
            print(f"{x:22s} {y:9s} {design:22s} {r['estimate']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                  f"films {r['films']:,}, clusters {r['clusters']:,}", flush=True)
models = pd.DataFrame(rows)
models.to_csv(OUT / "film_models.csv", index=False)
