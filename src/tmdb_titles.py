"""TMDb facts that IMDb lacks, per IMDb title: original language, origin country, TMDb genres and type, and the
franchise a film belongs to (TMDb collection).

Series get their IMDb id from external_ids or, failing that, from the /find lookups. Output: DERIVED/tmdb_titles.parquet.
"""
import json
import sqlite3
import zlib

import pandas as pd

from paths import DERIVED, TMDB_DB

db = sqlite3.connect(f"file:{TMDB_DB}?mode=ro", uri=True)
rows = []
for path, body in db.execute("SELECT path, body FROM responses WHERE status = 200 AND "
                             "(path LIKE '/tv/%?append%' OR path LIKE '/movie/%')"):
    d = json.loads(zlib.decompress(body))
    series = path.startswith("/tv/")
    imdb = (d.get("external_ids") or {}).get("imdb_id") or d.get("imdb_id")
    countries = d.get("origin_country") or [c["iso_3166_1"] for c in d.get("production_countries") or []]
    rows.append(("series" if series else "film", d["id"], imdb or None, d.get("original_language"),
                 countries[0] if countries else None, "|".join(countries),
                 "|".join(g["name"] for g in d.get("genres") or []), d.get("type") if series else None,
                 d.get("popularity"), d.get("vote_count"),
                 None if series else (d.get("belongs_to_collection") or {}).get("id")))
t = pd.DataFrame(rows, columns=["kind", "tmdb_id", "imdb", "language", "country", "countries", "tmdb_genres", "tv_type",
                                "popularity", "tmdb_votes", "collection"])
found = []
for path, body in db.execute("SELECT path, body FROM responses WHERE path LIKE '/find/%' AND status = 200"):
    hits = json.loads(zlib.decompress(body)).get("tv_results") or []
    if hits:
        found.append((hits[0]["id"], path.split("/")[2].split("?")[0]))
found = pd.DataFrame(found, columns=["tmdb_id", "imdb_found"])
series = t.kind == "series"
t.loc[series, "imdb"] = t[series].merge(found, on="tmdb_id", how="left").set_index(t[series].index).pipe(
    lambda m: m.imdb.fillna(m.imdb_found))
t = t.dropna(subset=["imdb"]).drop_duplicates("imdb")
t.to_parquet(DERIVED / "tmdb_titles.parquet", index=False)
print(t.groupby("kind").size().to_dict(), "titles with an IMDb id")
