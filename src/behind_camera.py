"""Women behind the camera, in leading roles, and the ages of leads — from IMDb data the earlier studies left unused.

1. Jobs: directors and writers (title.crew plus title.principals) and producers, editors, composers, cinematographers,
   casting directors and production designers (title.principals). Gender is a probability from imdb_people.py (exact
   for people with acting credits, else from the first name); a group's women's share is the mean probability, which
   is unbiased where hard labels are not. Unit: a person's job on a film or series (counted once per series).
2. Leads: women's share of the 1st-listed and top-3 acting credits against the rest of the credited cast (exact gender:
   the actor/actress credit). IMDb lists casts in credits order, usually billing order but sometimes order of
   appearance, so "top 3" is used alongside "1st".
3. Ages: films with at least 100 votes; age at release of the top-3 cast (birth year known for 683k people, mostly the
   better known); age gap between the first-listed man and woman among the top 3.
Fiction only (documentary, reality, talk, game and news titles left out); adult titles left out.
Outputs: results/behind_camera/*.csv (figures drawn by figures.py).
"""
import duckdb

from paths import DERIVED, IMDB, RESULTS

OUT = RESULTS / "behind_camera"
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit = '8GB'; SET preserve_insertion_order = false")


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{{cols}}})"


con.execute(f"""CREATE TEMP TABLE basics AS SELECT tconst, titleType, isAdult = '1' AS adult,
    TRY_CAST(startYear AS INT) AS year, genres FROM {tsv('title.basics.tsv.gz', ['tconst', 'titleType', 'primaryTitle',
    'originalTitle', 'isAdult', 'startYear', 'endYear', 'runtimeMinutes', 'genres'])}""")
con.execute(f"""CREATE TEMP TABLE episode AS SELECT tconst, parentTconst FROM {tsv('title.episode.tsv.gz',
    ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber'])}""")
con.execute(f"""CREATE TEMP TABLE titles AS
    SELECT b.tconst, COALESCE(e.parentTconst, b.tconst) AS work, COALESCE(b.year, w.year) AS year,
           CASE WHEN COALESCE(w.titleType, b.titleType) = 'movie' THEN 'film'
                WHEN COALESCE(w.titleType, b.titleType) IN ('tvSeries', 'tvMiniSeries') THEN 'tv' END AS medium,
           COALESCE(w.genres, b.genres) AS genres, b.adult OR COALESCE(w.adult, false) AS adult,
           t.language = 'en' AS english
    FROM basics b LEFT JOIN episode e USING (tconst) LEFT JOIN basics w ON w.tconst = e.parentTconst
    LEFT JOIN '{DERIVED / "tmdb_titles.parquet"}' t ON t.imdb = COALESCE(e.parentTconst, b.tconst)""")
FICTION = ("t.medium IS NOT NULL AND NOT t.adult AND t.year BETWEEN 1920 AND 2026 AND "
           "NOT regexp_matches(COALESCE(t.genres, ''), 'Documentary|Reality-TV|Talk-Show|Game-Show|News')")
con.execute(f"""CREATE TEMP TABLE people AS SELECT nconst, p_female, birth_year FROM '{DERIVED / "people.parquet"}'""")
con.execute(f"""CREATE TEMP TABLE principals AS SELECT tconst, TRY_CAST(ordering AS INT) AS ordering, nconst, category
    FROM {tsv('title.principals.tsv.gz', ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters'])}""")
print("tables loaded", flush=True)

# 1. jobs behind the camera
con.execute(f"""CREATE TEMP TABLE jobs AS
    WITH crew AS (
        SELECT tconst, unnest(string_split(directors, ',')) AS nconst, 'director' AS job
        FROM {tsv('title.crew.tsv.gz', ['tconst', 'directors', 'writers'])} WHERE directors IS NOT NULL
        UNION SELECT tconst, unnest(string_split(writers, ',')), 'writer'
        FROM {tsv('title.crew.tsv.gz', ['tconst', 'directors', 'writers'])} WHERE writers IS NOT NULL
        UNION SELECT tconst, nconst, category FROM principals
        WHERE category IN ('director', 'writer', 'producer', 'editor', 'composer', 'cinematographer',
                           'casting_director', 'production_designer'))
    SELECT t.work, c.nconst, c.job, any_value(t.medium) AS medium, MIN(t.year) AS year, any_value(t.english) AS english
    FROM crew c JOIN titles t USING (tconst) WHERE {FICTION} GROUP BY t.work, c.nconst, c.job""")
jobs = con.execute("""SELECT j.job, j.medium, CASE WHEN j.year < 1950 THEN 1940 ELSE j.year // 10 * 10 END AS decade,
        COALESCE(j.english, false) AS english, COUNT(*) AS credits, COUNT(p.p_female) AS with_gender,
        AVG(p.p_female) AS women_share
    FROM jobs j LEFT JOIN people p USING (nconst) GROUP BY ALL ORDER BY ALL""").df()
jobs.to_csv(OUT / "jobs_by_decade.csv", index=False)
print(jobs.groupby("job").apply(lambda d: f"credits {d.credits.sum():,}, gender known {d.with_gender.sum() / d.credits.sum():.0%}",
                                include_groups=False).to_string(), flush=True)

# 2. leads: credits order among the acting credits of each film or series
con.execute(f"""CREATE TEMP TABLE cast_ranked AS
    SELECT p.tconst, t.medium, t.year, COALESCE(t.english, false) AS english, p.category = 'actress' AS female, p.nconst,
           ROW_NUMBER() OVER (PARTITION BY p.tconst ORDER BY p.ordering) AS rank
    FROM principals p JOIN titles t USING (tconst)
    WHERE p.category IN ('actor', 'actress') AND {FICTION} AND t.tconst = t.work""")
leads = con.execute("""SELECT medium, CASE WHEN year < 1950 THEN 1940 ELSE year // 10 * 10 END AS decade, english,
        CASE WHEN rank = 1 THEN '1st listed' WHEN rank <= 3 THEN '2nd-3rd listed' ELSE '4th or later' END AS position,
        COUNT(*) AS credits, AVG(female::INT) AS women_share
    FROM cast_ranked GROUP BY ALL ORDER BY ALL""").df()
leads.to_csv(OUT / "leads_by_decade.csv", index=False)

# 3. ages of the top-3 cast in films with at least 100 votes
ages = con.execute(f"""
    WITH r AS (SELECT tconst FROM {tsv('title.ratings.tsv.gz', ['tconst', 'averageRating', 'numVotes'])}
               WHERE CAST(numVotes AS INT) >= 100),
    top AS (SELECT c.*, c.year - p.birth_year AS age FROM cast_ranked c JOIN r USING (tconst)
            LEFT JOIN people p USING (nconst) WHERE c.medium = 'film' AND c.rank <= 3 AND c.year >= 1930)
    SELECT CASE WHEN year < 1950 THEN 1940 ELSE year // 10 * 10 END AS decade, english, female, COUNT(*) AS credits,
           COUNT(age) FILTER (WHERE age BETWEEN 5 AND 100) AS with_age,
           AVG(age) FILTER (WHERE age BETWEEN 5 AND 100) AS mean_age,
           MEDIAN(age) FILTER (WHERE age BETWEEN 5 AND 100) AS median_age,
           AVG((age >= 40)::INT) FILTER (WHERE age BETWEEN 5 AND 100) AS share_40_plus
    FROM top GROUP BY ALL ORDER BY ALL""").df()
ages.to_csv(OUT / "ages_by_decade.csv", index=False)
gap = con.execute(f"""
    WITH r AS (SELECT tconst FROM {tsv('title.ratings.tsv.gz', ['tconst', 'averageRating', 'numVotes'])}
               WHERE CAST(numVotes AS INT) >= 100),
    top AS (SELECT c.*, c.year - p.birth_year AS age FROM cast_ranked c JOIN r USING (tconst)
            LEFT JOIN people p USING (nconst) WHERE c.medium = 'film' AND c.rank <= 3 AND c.year >= 1930),
    firsts AS (SELECT tconst, any_value(year) AS year, any_value(english) AS english,
                      arg_min(age, rank) FILTER (WHERE female) AS woman_age,
                      arg_min(age, rank) FILTER (WHERE NOT female) AS man_age
               FROM top GROUP BY tconst)
    SELECT CASE WHEN year < 1950 THEN 1940 ELSE year // 10 * 10 END AS decade, english, COUNT(*) AS films,
           AVG(man_age - woman_age) AS mean_gap, MEDIAN(man_age - woman_age) AS median_gap,
           AVG((man_age - woman_age >= 10)::INT) AS share_gap_10_plus
    FROM firsts WHERE woman_age BETWEEN 5 AND 100 AND man_age BETWEEN 5 AND 100 GROUP BY ALL ORDER BY ALL""").df()
gap.to_csv(OUT / "lead_age_gap_by_decade.csv", index=False)

import pandas as pd
pd.set_option("display.width", 200)
j = jobs.groupby(["job", "medium", "decade"]).apply(
    lambda d: pd.Series({"share": (d.women_share * d.with_gender).sum() / d.with_gender.sum()}), include_groups=False)
print(j.reset_index().pivot_table(index=["job", "medium"], columns="decade", values="share").round(3).to_string())
l = leads.groupby(["medium", "decade", "position"]).apply(
    lambda d: pd.Series({"share": (d.women_share * d.credits).sum() / d.credits.sum()}), include_groups=False)
print(l.reset_index().pivot_table(index=["medium", "position"], columns="decade", values="share").round(3).to_string())
a = ages.groupby(["decade", "female"]).apply(
    lambda d: pd.Series({"mean_age": (d.mean_age * d.with_age).sum() / d.with_age.sum(),
                         "coverage": d.with_age.sum() / d.credits.sum()}), include_groups=False)
print(a.reset_index().pivot_table(index="female", columns="decade", values=["mean_age", "coverage"]).round(2).to_string())
g = gap.groupby("decade").apply(lambda d: pd.Series({"films": d.films.sum(),
                                                     "mean_gap": (d.mean_gap * d.films).sum() / d.films.sum()}),
                                include_groups=False)
print(g.round(2).T.to_string())
