"""Critic and audience scores from Rotten Tomatoes and Metacritic film pages, fetched slowly and openly, for study 12.

Wikidata (wikidata_reception.py) has Tomatometer scores for about 26,000 films and Metascores for about 10,000, most
retrieved in 2021–2026, and neither site's audience score. This fills the gaps and adds the audience side.
Which pages: the film pages whose ids Wikidata knows (Rotten Tomatoes /m/<id>, Metacritic /movie/<id>), for rated
feature films with at least MIN_VOTES (5,000) IMDb votes, most-voted first: the films critics review in numbers
(below that, Wikidata's copies remain, from a median of 9–15 reviews). Both sites' robots.txt allow these pages; search,
picture galleries and account pages, which they disallow, are never requested.
Manners: one request at a time, a random 2–4 seconds apart, under a User-Agent that names this project. HTTP 429 or
503 means wait (Retry-After, else 5 minutes, doubling) and retry; a 403 or a challenge page means stop. Nothing tries
to look like a browser or get around a block. Every response is kept (compressed, private) and the run resumes where
it stopped.
Parsed: Rotten Tomatoes Tomatometer, average rating, review count and certification, audience score and its count;
Metacritic Metascore and review count, user score and rating count.
Usage: reception_fetch.py rt|mc [max_pages | parse]   (parse: only turn the pages fetched so far into the table)
Outputs (private): DERIVED/reception/{rt,mc}.sqlite (pages) and DERIVED/reception/{rt,mc}_scores.parquet.
"""
import json
import os
import random
import re
import sqlite3
import sys
import time
import zlib

import duckdb
import pandas as pd
import requests

from paths import DERIVED, IMDB

SITE = sys.argv[1]
assert SITE in ("rt", "mc"), "usage: reception_fetch.py rt|mc [max_pages | parse]"
PARSE_ONLY = len(sys.argv) > 2 and sys.argv[2] == "parse"
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 and not PARSE_ONLY else None
MIN_VOTES = int(os.environ.get("MIN_VOTES", "5000"))
UA = "screen-stereotypes-research/1.0 (non-commercial research; https://github.com/AlexandruPascu/screen-stereotypes)"
BASE = {"rt": "https://www.rottentomatoes.com/", "mc": "https://www.metacritic.com/"}[SITE]
PREFIX = {"rt": "m/", "mc": "movie/"}[SITE]
OUT = DERIVED / "reception"
OUT.mkdir(exist_ok=True)
CHALLENGE = re.compile(r"(?i)captcha|cf-chl|just a moment\.\.\.|access denied|are you a robot")


def films():
    """Rated feature films with a page id for this site, most-voted first."""
    ids = pd.read_parquet(DERIVED / "wikidata/ids.parquet")
    col = {"rt": "rotten_tomatoes_id", "mc": "metacritic_id"}[SITE]
    ids = ids[ids[col].fillna("").str.startswith(PREFIX)][["imdb", col]].rename(columns={col: "page"})
    cols = lambda names: "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in names) + "}"
    rated = duckdb.connect().execute(f"""SELECT b.tconst AS imdb, CAST(r.numVotes AS INT) AS votes
        FROM read_csv('{IMDB / "title.basics.tsv.gz"}', delim='\\t', header=true, quote='', escape='', nullstr='\\N',
                      columns={cols(['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'startYear',
                                     'endYear', 'runtimeMinutes', 'genres'])}) b
        JOIN read_csv('{IMDB / "title.ratings.tsv.gz"}', delim='\\t', header=true, quote='', escape='', nullstr='\\N',
                      columns={cols(['tconst', 'averageRating', 'numVotes'])}) r USING (tconst)
        WHERE b.titleType = 'movie' AND CAST(r.numVotes AS INT) >= {MIN_VOTES}""").df()
    d = ids.merge(rated, on="imdb").sort_values(["votes", "imdb"], ascending=[False, True])
    allowed = d.page.str.fullmatch(PREFIX + r"[A-Za-z0-9_\-.%]+")  # a film page, nothing below it
    return d[allowed].drop_duplicates("page")


def parse_rt(html):
    out = {}
    for name, key in [("criticsScore", "critic"), ("audienceScore", "audience")]:
        m = re.search(r'"' + name + r'"\s*:\s*(\{[^{}]*\})', html)
        if m:
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            out[f"{key}_score"] = pd.to_numeric(d.get("score"), errors="coerce")
            out[f"{key}_reviews"] = pd.to_numeric(d.get("reviewCount") or d.get("ratingCount"), errors="coerce")
            if key == "critic":
                out["critic_average"] = pd.to_numeric(d.get("averageRating"), errors="coerce")
                out["certified_fresh"] = bool(d.get("certified"))
    return out


def parse_mc(html):
    out = {}
    for block in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S):
        try:
            d = json.loads(block)
        except json.JSONDecodeError:
            continue
        rating = d.get("aggregateRating") if isinstance(d, dict) else None
        if rating:
            out["critic_score"] = pd.to_numeric(rating.get("ratingValue"), errors="coerce")
            out["critic_reviews"] = pd.to_numeric(rating.get("reviewCount"), errors="coerce")
    values = re.findall(r'data-testid="global-score-value"[^>]*>(?:<[^>]+>)*\s*([0-9.]+|tbd)', html)
    user = next((v for v in values if "." in v), None)  # user scores are shown with a decimal, Metascores without
    if user:
        out["audience_score"] = float(user)
    m = re.search(r"Based on ([0-9,]+) User Rating", html)
    if m:
        out["audience_reviews"] = int(m.group(1).replace(",", ""))
    return out


db = sqlite3.connect(OUT / f"{SITE}.sqlite")
db.execute("CREATE TABLE IF NOT EXISTS pages (page TEXT PRIMARY KEY, imdb TEXT, status INTEGER, fetched_at REAL, "
           "final_url TEXT, body BLOB)")
done = {p for (p,) in db.execute("SELECT page FROM pages")}
queue = films()
todo = queue[~queue.page.isin(done)]
if PARSE_ONLY:
    todo = todo.head(0)
if MAX_PAGES:
    todo = todo.head(MAX_PAGES)
print(f"{SITE}: {len(queue):,} film pages with >= {MIN_VOTES} votes; already fetched {len(done):,}; to fetch {len(todo):,}",
      flush=True)

session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept": "text/html", "Accept-Language": "en"})
started, fetched, wait = time.time(), 0, 300
for imdb, page in zip(todo.imdb, todo.page):
    while True:
        try:
            r = session.get(BASE + page + ("/" if SITE == "mc" else ""), timeout=60)
        except requests.RequestException as e:
            print(f"  network error on {page} ({e}); waiting 60 s", flush=True)
            time.sleep(60)
            continue
        if r.status_code in (429, 503):
            pause = int(r.headers.get("Retry-After", wait)) if str(r.headers.get("Retry-After", "")).isdigit() else wait
            print(f"  {r.status_code} on {page}: waiting {pause // 60} min as asked", flush=True)
            time.sleep(pause)
            wait = min(wait * 2, 3600)
            continue
        break
    challenge = r.status_code == 200 and len(r.content) < 60000 and CHALLENGE.search(r.text) and "aggregateRating" not in r.text
    if r.status_code == 403 or challenge:  # film pages run to hundreds of kilobytes; challenge pages are small
        print(f"  {r.status_code} / challenge on {page}: the site is refusing us, so the run stops here", flush=True)
        break
    wait = 300
    db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?)",
               (page, imdb, r.status_code, time.time(), r.url, zlib.compress(r.content) if r.status_code == 200 else None))
    db.commit()
    fetched += 1
    if fetched % 50 == 0:
        rate = fetched / (time.time() - started) * 3600
        print(f"  {fetched:,}/{len(todo):,} fetched ({rate:.0f} pages/hour; about {(len(todo) - fetched) / rate:.1f} h left)",
              flush=True)
    time.sleep(random.uniform(2, 4))

# parse everything fetched so far into one table
parse = parse_rt if SITE == "rt" else parse_mc
rows = []
for page, imdb, status, at, body in db.execute("SELECT page, imdb, status, fetched_at, body FROM pages"):
    row = {"imdb": imdb, "page": page, "status": status, "fetched_at": pd.to_datetime(at, unit="s")}
    if status == 200 and body:
        row.update(parse(zlib.decompress(body).decode("utf-8", errors="replace")))
    rows.append(row)
scores = pd.DataFrame(rows)
scores.to_parquet(OUT / f"{SITE}_scores.parquet", index=False)
ok = scores[scores.status == 200]
print(f"{SITE}: {len(scores):,} pages kept ({len(ok):,} ok); with a critic score {ok.critic_score.notna().sum() if 'critic_score' in ok else 0:,}, "
      f"with an audience score {ok.audience_score.notna().sum() if 'audience_score' in ok else 0:,}", flush=True)
