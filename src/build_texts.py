"""One table of English plot text per title, from every source we hold. Private: holds IMDb and TMDb text.

Sources, keyed by IMDb title id (tconst) where possible:
  tmdb    TMDb overviews: films and series by their IMDb id; episodes through (series, season, episode), but only for
          series whose numbering is shown to line up with IMDb's (titles agree, or air years agree when titles are
          generic). Episodes of series TMDb cannot tie to IMDb are kept under a TMDb key (tmdb:<series>:<s>:<e>).
  imdb22  the 2022 IMDb scrape (plot, outline, synopsis, user keywords).
  plot17  the 2017 IMDb plot.list, matched by (series title, year, season, episode) or (film title, year).

Each source is written to its own part file as soon as it is built (the whole set does not fit in memory as Python
objects), then DuckDB merges them. Output: DERIVED/texts.parquet and DERIVED/series_alignment.parquet.
"""
import gc
import json
import re
import sqlite3
import unicodedata
import zlib

import duckdb
import numpy as np
import pandas as pd

from paths import DATA, DERIVED, IMDB, TMDB_DB

pd.options.future.infer_string = True  # Arrow-backed strings: a fraction of the memory of Python objects
PARTS = DERIVED / "texts_parts"
PARTS.mkdir(exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")


def norm(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{{cols}}})"


def save(name, title, kind, text, keywords=None, source=None):
    df = pd.DataFrame({"title": title, "source": source, "kind": kind, "text": text,
                       "keywords": keywords if keywords is not None else ""})
    df.to_parquet(PARTS / f"{name}.parquet", index=False)
    print(f"  wrote {name}: {len(df):,} rows", flush=True)


con.execute(f"""CREATE TEMP TABLE basics AS SELECT tconst, titleType, primaryTitle, originalTitle,
    TRY_CAST(startYear AS INT) AS year FROM {tsv('title.basics.tsv.gz', ['tconst', 'titleType', 'primaryTitle',
    'originalTitle', 'isAdult', 'startYear', 'endYear', 'runtimeMinutes', 'genres'])}""")
con.execute(f"""CREATE TEMP TABLE episode AS SELECT e.tconst, e.parentTconst, TRY_CAST(e.seasonNumber AS INT) AS season,
    TRY_CAST(e.episodeNumber AS INT) AS episode, b.primaryTitle, b.year
    FROM {tsv('title.episode.tsv.gz', ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber'])} e
    JOIN basics b USING (tconst) WHERE e.seasonNumber IS NOT NULL AND e.episodeNumber IS NOT NULL""")

# --- TMDb films and series --------------------------------------------------------------------------------------
db = sqlite3.connect(f"file:{TMDB_DB}?mode=ro", uri=True)
keys, kinds, texts, kws = [], [], [], []
for path, body in db.execute("SELECT path, body FROM responses WHERE status = 200 AND (path LIKE '/movie/%' OR "
                             "path LIKE '/tv/%?append%')"):
    d = json.loads(zlib.decompress(body))
    film = path.startswith("/movie/")
    imdb = (d.get("external_ids") or {}).get("imdb_id") or d.get("imdb_id")
    k = d.get("keywords") or {}
    keys.append(imdb or (f"tmdb:film:{d['id']}" if film else f"tmdb:tv:{d['id']}"))
    kinds.append("film" if film else "series")
    texts.append(d.get("overview") or "")
    kws.append("|".join(x["name"] for x in k.get("keywords") or k.get("results") or []))
save("tmdb_titles", keys, kinds, texts, kws, "tmdb")
del keys, kinds, texts, kws
gc.collect()

# --- TMDb episodes with the per-series alignment check -------------------------------------------------------------
link = pd.read_parquet(DERIVED / "tmdb_titles.parquet", columns=["kind", "tmdb_id", "imdb"])
link = link[link.kind == "series"][["tmdb_id", "imdb"]]
cols = {"series": [], "season": [], "episode": [], "name": [], "overview": [], "air_year": []}
for path, body in db.execute("SELECT path, body FROM responses WHERE status = 200 AND path LIKE '/tv/%/season/%'"):
    sid = int(path.split("/")[2])
    for e in json.loads(zlib.decompress(body)).get("episodes") or []:
        if e.get("season_number") is None or e.get("episode_number") is None:
            continue
        air = e.get("air_date") or ""
        for c, v in zip(cols, (sid, e["season_number"], e["episode_number"], e.get("name") or "", e.get("overview") or "",
                               int(air[:4]) if air[:4].isdigit() else None)):
            cols[c].append(v)
tm = pd.DataFrame(cols).merge(link, left_on="series", right_on="tmdb_id", how="left").drop(columns="tmdb_id")
del cols
gc.collect()
print(f"TMDb episodes {len(tm):,}; in series linked to IMDb {tm.imdb.notna().sum():,}", flush=True)

con.register("tm_keys", tm[["series", "season", "episode", "imdb", "name", "air_year"]].dropna(subset=["imdb"]))
j = con.execute("""SELECT t.series, t.season, t.episode, t.imdb, t.name, t.air_year, e.tconst, e.primaryTitle, e.year
    FROM tm_keys t JOIN episode e ON e.parentTconst = t.imdb AND e.season = t.season AND e.episode = t.episode""").df()
con.unregister("tm_keys")
generic = re.compile(r"^(episode|folge|episodio|capitulo|aflevering|avsnitt|odcinek|bolum|jakso|part)\s*\d+$|^\d+$")
a, t = j.primaryTitle.map(norm), j.name.map(norm)
j["comparable"] = ~a.str.match(generic) & ~t.str.match(generic) & (a != "") & (t != "")
j["name_ok"] = j.comparable & ((a == t) | np.array([x in y or y in x for x, y in zip(a, t)]))
j["year_known"] = j.year.notna() & j.air_year.notna()
j["year_ok"] = j.year_known & ((j.year - j.air_year).abs() <= 1)
al = j.groupby("imdb").agg(joined=("tconst", "size"), comparable=("comparable", "sum"), name_ok=("name_ok", "sum"),
                           year_known=("year_known", "sum"), year_ok=("year_ok", "sum")).reset_index()
al["name_rate"] = al.name_ok / al.comparable.replace(0, np.nan)
al["year_rate"] = al.year_ok / al.year_known.replace(0, np.nan)
al["aligned"] = np.where(al.comparable >= 5, al.name_rate >= 0.7, al.year_rate.fillna(0) >= 0.9)
al.to_parquet(DERIVED / "series_alignment.parquet", index=False)
good = j[j.imdb.isin(set(al[al.aligned].imdb))][["series", "season", "episode", "tconst"]]
print(f"series checked {len(al):,}: aligned {al.aligned.sum():,} ({al[al.aligned].joined.sum():,} joined episodes), "
      f"rejected {(~al.aligned).sum():,} ({al[~al.aligned].joined.sum():,} episodes)", flush=True)
del j, a, t
gc.collect()
tm = tm.merge(good, on=["series", "season", "episode"], how="left")
tm["key"] = tm.tconst.fillna("tmdb:" + tm.series.astype(str) + ":" + tm.season.astype(str) + ":" + tm.episode.astype(str))
tm = tm[tm.overview != ""]
save("tmdb_episodes", tm.key, "episode", tm.overview, None, "tmdb")
del tm, good
gc.collect()

# --- 2022 scrape ---------------------------------------------------------------------------------------------
big = pd.read_parquet(DATA / "gcp-2022/bigquery/bigdata.parquet",
                      columns=["titleCodes", "plot", "plotOutline", "synopsis", "keywords"])
for c in ["plot", "plotOutline", "synopsis", "keywords"]:
    big[c] = big[c].where(~big[c].isin(["NULL", "", "[]"])).fillna("")
text = (big["plot"] + " " + big.plotOutline + " " + big.synopsis).str.strip()
save("imdb22", big.titleCodes, "episode", text, big.keywords, "imdb22")
del big, text
gc.collect()

# --- 2017 plot.list ------------------------------------------------------------------------------------------
title_re = re.compile(r'^(?P<q>")?(?P<title>.+?)(?(q)") \((?P<year>\d{4})(?:/[IVXL]+)?\)(?P<rest>.*)$')
ep_re = re.compile(r"\{(?P<ename>.*?)(?:\s*\(#(?P<s>\d+)\.(?P<e>\d+)\))?\}")
pl, cur, buf, plots = [], None, [], []


def close():
    if not cur:
        return
    m = title_re.match(cur)
    text = " ".join(plots + ([" ".join(buf)] if buf else []))
    if not m or not text.strip() or "(VG)" in m["rest"] or "(V)" in m["rest"]:
        return
    em = ep_re.search(m["rest"]) if m["q"] else None
    pl.append(("episode" if em else "series" if m["q"] else "film", norm(m["title"]), int(m["year"]),
               int(em["s"]) if em and em["s"] else None, int(em["e"]) if em and em["e"] else None, text))


with open(DATA / "gcp-2022/buckets/plot.list", encoding="latin-1") as f:
    started = False
    for line in f:
        if not started:
            started = line.startswith("PLOT SUMMARIES LIST")
            continue
        if line.startswith("MV: "):
            close()
            cur, buf, plots = line[4:].strip(), [], []
        elif line.startswith("PL: "):
            buf.append(line[4:].strip())
        elif buf and (line.startswith("BY: ") or line.startswith("---")):
            plots.append(" ".join(buf))
            buf = []
    close()
pl = pd.DataFrame(pl, columns=["kind", "k", "year", "season", "episode", "text"])


def unique_keys(types):
    b = con.execute(f"SELECT tconst, primaryTitle, originalTitle, year FROM basics WHERE titleType IN "
                    f"({', '.join(repr(x) for x in types)})").df()
    k = pd.concat([b.assign(k=b.primaryTitle.map(norm)), b.assign(k=b.originalTitle.map(norm))])[["tconst", "k", "year"]]
    k = k.drop_duplicates()
    n = k.groupby(["k", "year"]).tconst.transform("nunique")
    return k[n == 1].drop_duplicates(["k", "year"])


sk = unique_keys(["tvSeries", "tvMiniSeries"])
ep_keys = con.execute("SELECT tconst, parentTconst, season, episode FROM episode").df()
pe = pl[(pl.kind == "episode") & pl.season.notna()].merge(sk, on=["k", "year"]).merge(
    ep_keys, left_on=["tconst", "season", "episode"], right_on=["parentTconst", "season", "episode"],
    suffixes=("_series", ""))
del ep_keys
ps = pl[pl.kind == "series"].merge(sk, on=["k", "year"])
pf = pl[pl.kind == "film"].merge(unique_keys(["movie", "tvMovie"]), on=["k", "year"])
print(f"plot.list matched: episodes {len(pe):,}, series {len(ps):,}, films {len(pf):,} (of {len(pl):,} entries)", flush=True)
save("plot17", pd.concat([pe.tconst, ps.tconst, pf.tconst]), ["episode"] * len(pe) + ["series"] * len(ps) +
     ["film"] * len(pf), pd.concat([pe.text, ps.text, pf.text]), None, "plot17")
del pl, pe, ps, pf, sk
gc.collect()

# --- merge ---------------------------------------------------------------------------------------------------
con.execute(f"""COPY (
    SELECT title, source, any_value(kind) AS kind, any_value(trim(text)) AS text, any_value(keywords) AS keywords,
           len(string_split(trim(any_value(text)), ' ')) * (trim(any_value(text)) <> '')::INT AS words
    FROM '{PARTS}/*.parquet' WHERE trim(text) <> '' OR keywords <> ''
    GROUP BY title, source
) TO '{DERIVED / "texts.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
print(con.execute(f"""SELECT source, kind, COUNT(*) AS titles, SUM(words) AS words FROM '{DERIVED / "texts.parquet"}'
    GROUP BY ALL ORDER BY 1, 2""").df().to_string(index=False))
print(con.execute(f"""SELECT COUNT(DISTINCT title) FILTER (WHERE words > 0) AS titles_with_text, SUM(words) AS words,
    COUNT(DISTINCT title) FILTER (WHERE words > 0 AND title LIKE 'tt%') AS imdb_titles_with_text
    FROM '{DERIVED / "texts.parquet"}'""").df().to_string(index=False))
