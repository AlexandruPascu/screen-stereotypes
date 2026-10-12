"""Every named acting role in IMDb's 2026 files, one row per (film or series, performer, character).

A role in a series is counted once however many episodes it spans, so long-running leads do not swamp the counts.
Gender is IMDb's actor/actress category. Adult titles are left out. Output: DERIVED/roles.parquet (private).
"""
import time

import duckdb

from paths import DERIVED, IMDB

DERIVED.mkdir(exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit = '7GB'; SET preserve_insertion_order = false")
started = time.time()


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return (f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', "
            f"columns={{{cols}}})")


con.execute(f"""CREATE TEMP TABLE basics AS SELECT tconst, titleType, isAdult = '1' AS adult,
    TRY_CAST(startYear AS INTEGER) AS year, genres FROM {tsv('title.basics.tsv.gz', ['tconst', 'titleType',
    'primaryTitle', 'originalTitle', 'isAdult', 'startYear', 'endYear', 'runtimeMinutes', 'genres'])}""")
con.execute(f"""CREATE TEMP TABLE episode AS SELECT tconst, parentTconst FROM {tsv('title.episode.tsv.gz',
    ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber'])}""")
print(f"basics and episodes loaded ({time.time() - started:.0f}s)", flush=True)

con.execute(f"""CREATE TEMP TABLE credits AS
    SELECT p.tconst, p.nconst, p.category = 'actress' AS female,
           UNNEST(TRY_CAST(json(p.characters) AS VARCHAR[])) AS character
    FROM {tsv('title.principals.tsv.gz', ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters'])} p
    WHERE p.category IN ('actor', 'actress') AND p.characters IS NOT NULL""")
print(f"credits: {con.execute('SELECT COUNT(*) FROM credits').fetchone()[0]:,} ({time.time() - started:.0f}s)", flush=True)

con.execute("""CREATE TEMP TABLE named AS
    SELECT c.tconst, c.nconst, c.female, c.character,
           trim(regexp_replace(regexp_replace(regexp_replace(lower(strip_accents(c.character)),
                '\\([^)]*\\)', ' ', 'g'), '#\\s*\\d+|\\b\\d+\\b', ' ', 'g'), '\\s+', ' ', 'g')) AS name,
           COALESCE(e.parentTconst, c.tconst) AS work, e.parentTconst IS NOT NULL AS is_episode,
           b.year AS credit_year, b.adult
    FROM credits c JOIN basics b USING (tconst) LEFT JOIN episode e USING (tconst)""")

con.execute(f"""COPY (
    SELECT n.work, n.nconst, n.name, any_value(n.female) AS female, any_value(n.character) AS character,
           COALESCE(MIN(n.credit_year), any_value(w.year)) AS year, COUNT(*) AS credits,
           bool_or(n.is_episode) AS from_episodes, any_value(w.titleType) AS work_type, any_value(w.genres) AS genres,
           CASE WHEN any_value(w.titleType) IN ('tvSeries', 'tvMiniSeries', 'tvMovie', 'tvSpecial', 'tvShort') THEN 'tv'
                WHEN any_value(w.titleType) = 'movie' THEN 'film' ELSE 'other' END AS medium
    FROM named n JOIN basics w ON w.tconst = n.work
    WHERE NOT n.adult AND NOT w.adult AND n.name <> ''
    GROUP BY n.work, n.nconst, n.name
) TO '{DERIVED / 'roles.parquet'}' (FORMAT parquet, COMPRESSION zstd)""")
n = con.execute(f"SELECT COUNT(*), SUM(female::INT) FROM '{DERIVED / 'roles.parquet'}'").fetchone()
print(f"roles: {n[0]:,}, women {n[1] / n[0]:.1%} ({time.time() - started:.0f}s)", flush=True)
