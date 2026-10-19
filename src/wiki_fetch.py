"""Wikipedia episode summaries (CC BY-SA 4.0), matched to IMDb episodes.

1. Fetches the wikitext of every article that uses {{Episode list}} (episode lists and season pages), 50 per request,
   paced, with a descriptive User-Agent; keeps the raw responses (DATA/wikipedia/pages.jsonl.gz) so reruns are offline.
2. Traces each page to its series' IMDb id: the page's Wikidata item (or its "part of the series"/"part of" parent)
   with an IMDb id (P345); failing that, the series article named in the page title.
3. Parses each {{Episode list}} entry: title, season (from the page title or the section heading), episode number
   within the season, and ShortSummary.
4. Joins to IMDb episodes on (series, season, episode) and keeps only series whose episode titles agree with IMDb's
   for at least 70% of comparable episodes (the same check used for TMDb).
5. The same for each entry's viewer count ("Viewers", in millions: Nielsen for US series, BARB for British ones), kept
   whether or not the entry has a summary.
The Wikidata and Wikipedia lookups of step 2 are cached (DATA/wikipedia/lookups.json.gz), so reruns are offline.
Usage: wiki_fetch.py [--viewers-only]   (--viewers-only leaves the summary table as it was built)
Outputs: DERIVED/wiki_episodes.parquet (tconst, series, season, episode, title, summary) and
DERIVED/wiki_viewers.parquet (tconst, series, season, episode, viewers).
"""
import gzip
import json
import re
import sys
import time
import unicodedata

import duckdb
import mwparserfromhell
import numpy as np
import pandas as pd
import requests

from paths import DATA, DERIVED, IMDB

API = "https://en.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"
UA = "screen-stereotypes research (non-commercial; https://github.com/AlexandruPascu/screen-stereotypes)"
W = DATA / "wikipedia"
s = requests.Session()
s.headers["User-Agent"] = UA


def get(url, **params):
    params.update(format="json", formatversion=2)
    if url == API:
        params["maxlag"] = 5
    for attempt in range(8):
        try:
            r = s.get(url, params=params, timeout=90)
            if r.status_code == 200 and "error" not in r.json():
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"API kept failing for {params.get('action')}")


def norm(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# 1. fetch (resumable: titles already in pages.jsonl.gz are skipped)
titles = (W / "episode_list_pages.txt").read_text().split("\n")
store = W / "pages.jsonl.gz"
done = set()
if store.exists():
    with gzip.open(store, "rt") as f:
        done = {json.loads(line)["title"] for line in f}
todo = [t for t in titles if t not in done]
print(f"pages: {len(titles):,}; to fetch {len(todo):,}", flush=True)
with gzip.open(store, "at") as out:
    for i in range(0, len(todo), 50):
        d = get(API, action="query", prop="revisions|pageprops", rvprop="content|ids", rvslots="main",
                ppprop="wikibase_item", titles="|".join(todo[i:i + 50]), redirects=1)
        for p in d["query"]["pages"]:
            if p.get("revisions"):
                out.write(json.dumps({"title": p["title"], "item": (p.get("pageprops") or {}).get("wikibase_item"),
                                      "revid": p["revisions"][0]["revid"],
                                      "text": p["revisions"][0]["slots"]["main"]["content"]}) + "\n")
        if i % 2000 == 0:
            print(f"  fetched {i + 50:,}/{len(todo):,}", flush=True)
        time.sleep(0.5)
with gzip.open(store, "rt") as f:
    pages = [json.loads(line) for line in f]
pages = list({p["title"]: p for p in pages}.values())

# 2. page -> series IMDb id
SEASON_TITLE = re.compile(r"^(?P<series>.+?) \((?:season|series) (?P<n>\d+)\)$|^(?P<series2>.+?) (?:season|series) (?P<n2>\d+)$",
                          re.I)


def series_title(title):
    t = re.sub(r"^List of ", "", title)
    t = re.sub(r" episodes( \(.*\))?$", "", t)
    m = SEASON_TITLE.match(t)
    return (m.group("series") or m.group("series2")) if m else t


LOOKUPS = W / "lookups.json.gz"
cached = json.load(gzip.open(LOOKUPS, "rt")) if LOOKUPS.exists() else {"claims": {}, "article_item": {}}
claims = cached["claims"]


def fetch_claims(ids):
    ids = [q for q in dict.fromkeys(ids) if q and q not in claims]
    for i in range(0, len(ids), 50):
        d = get(WD, action="wbgetentities", ids="|".join(ids[i:i + 50]), props="claims")
        for q, e in d.get("entities", {}).items():
            claims[q] = e.get("claims", {})
        time.sleep(0.5)


def values(q, prop):
    out = []
    for snak in claims.get(q, {}).get(prop, []):
        v = snak.get("mainsnak", {}).get("datavalue", {}).get("value")
        out.append(v["id"] if isinstance(v, dict) and "id" in v else v)
    return out


fetch_claims([p["item"] for p in pages])
fetch_claims([x for p in pages for x in values(p["item"], "P179") + values(p["item"], "P361")])


def imdb_of(q):
    ids = [v for v in values(q, "P345") if str(v).startswith("tt")]
    for parent in values(q, "P179") + values(q, "P361"):
        ids += [v for v in values(parent, "P345") if str(v).startswith("tt")]
    return ids[0] if ids else None


page_imdb = {p["title"]: imdb_of(p["item"]) for p in pages}
article_item = cached["article_item"]
missing = sorted({series_title(t) for t, v in page_imdb.items() if not v} - set(article_item) - set(cached.get("no_item", [])))
for i in range(0, len(missing), 50):
    d = get(API, action="query", prop="pageprops", ppprop="wikibase_item", titles="|".join(missing[i:i + 50]), redirects=1)
    back = {r["to"]: r["from"] for r in d["query"].get("redirects", [])}
    for p in d["query"]["pages"]:
        q = (p.get("pageprops") or {}).get("wikibase_item")
        if q:
            article_item[back.get(p["title"], p["title"])] = q
    time.sleep(0.5)
fetch_claims(list(article_item.values()))
cached["no_item"] = sorted(set(cached.get("no_item", [])) | (set(missing) - set(article_item)))
json.dump(cached, gzip.open(LOOKUPS, "wt"))
for t, v in page_imdb.items():
    if not v:
        page_imdb[t] = imdb_of(article_item.get(series_title(t)))
print(f"pages traced to a series IMDb id: {sum(bool(v) for v in page_imdb.values()):,}/{len(pages):,}", flush=True)

# 3. parse entries with their season
HEADING = re.compile(r"(?:season|series|book|volume|chapter|part|cycle)\s+(\d+)", re.I)
VIEWERS = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")
rows, viewer_rows = [], []
for p in pages:
    imdb = page_imdb.get(p["title"])
    if not imdb:
        continue
    m = SEASON_TITLE.match(re.sub(r"^List of | episodes$", "", p["title"]))
    page_season = int(m.group("n") or m.group("n2")) if m else None
    code = mwparserfromhell.parse(p["text"])
    for section in code.get_sections(levels=[2, 3], include_lead=True, flat=True):
        heads = section.filter_headings()
        hm = HEADING.search(heads[0].title.strip_code()) if heads else None
        season = page_season or (int(hm.group(1)) if hm else None)
        for t in section.filter_templates(recursive=True):  # entries sit inside {{Episode table}} on most pages
            if not str(t.name).strip().lower().startswith("episode list"):
                continue
            val = lambda k: mwparserfromhell.parse(str(t.get(k).value)).strip_code().strip() if t.has(k) else ""
            num2, num = re.match(r"\d+", val("EpisodeNumber2")), re.match(r"\d+", val("EpisodeNumber"))
            episode = int(num2.group()) if num2 else (int(num.group()) if num and page_season else None)
            summary = val("ShortSummary")
            if season is None or episode is None:
                continue
            v = VIEWERS.match(val("Viewers"))
            if v and 0 < float(v.group(1)) < 100:  # millions; a stray year or rating share falls outside
                viewer_rows.append((imdb, season, episode, val("Title").strip('"'), float(v.group(1))))
            if len(summary.split()) < 5:
                continue
            rows.append((imdb, season, episode, val("Title").strip('"'), summary, p["title"]))
wk = pd.DataFrame(rows, columns=["series", "season", "episode", "title", "summary", "page"])
wk["words"] = wk.summary.str.split().str.len()
wk = wk.sort_values("words", ascending=False).drop_duplicates(["series", "season", "episode"])
print(f"episode summaries with series, season and number: {len(wk):,} in {wk.series.nunique():,} series", flush=True)

# 4. join to IMDb and keep series whose titles agree
con = duckdb.connect()
cols = lambda names: "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in names) + "}"
ep = con.execute(f"""SELECT e.tconst, e.parentTconst AS series, TRY_CAST(e.seasonNumber AS INT) AS season,
        TRY_CAST(e.episodeNumber AS INT) AS episode, b.primaryTitle
    FROM read_csv('{IMDB / "title.episode.tsv.gz"}', delim='\\t', header=true, quote='', escape='', nullstr='\\N',
                  columns={cols(['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber'])}) e
    JOIN read_csv('{IMDB / "title.basics.tsv.gz"}', delim='\\t', header=true, quote='', escape='', nullstr='\\N',
                  columns={cols(['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'startYear',
                                 'endYear', 'runtimeMinutes', 'genres'])}) b USING (tconst)""").df()
generic = re.compile(r"^(episode|part|chapter)\s*\d+$|^\d+$")


def aligned(d):
    """d joined to IMDb episodes, and the part in series whose episode titles agree with IMDb's."""
    j = d.merge(ep, on=["series", "season", "episode"])
    a, t = j.primaryTitle.map(norm), j.title.map(norm)
    j["comparable"] = ~a.str.match(generic) & ~t.str.match(generic) & (a != "") & (t != "")
    j["ok"] = j.comparable & ((a == t) | np.array([x in y or y in x for x, y in zip(a, t)]))
    agree = j.groupby("series").agg(n=("comparable", "sum"), ok=("ok", "sum"))
    good = set(agree[(agree.n >= 3) & (agree.ok / agree.n.clip(lower=1) >= 0.7)].index)
    return j, j[j.series.isin(good)]


j, out = aligned(wk)
out = out[["tconst", "series", "season", "episode", "title", "summary", "words"]]
if "--viewers-only" not in sys.argv:
    out.to_parquet(DERIVED / "wiki_episodes.parquet", index=False)
print(f"joined to IMDb episodes: {len(j):,}; in series whose titles agree: {len(out):,} ({out.series.nunique():,} series); "
      f"median {out.words.median():.0f} words", flush=True)

# 5. viewer counts, through the same alignment
wv = pd.DataFrame(viewer_rows, columns=["series", "season", "episode", "title", "viewers"]).drop_duplicates(
    ["series", "season", "episode"])
_, vout = aligned(wv)
vout[["tconst", "series", "season", "episode", "viewers"]].to_parquet(DERIVED / "wiki_viewers.parquet", index=False)
print(f"episodes with a viewer count: {len(wv):,}; matched to IMDb in series whose titles agree: {len(vout):,} "
      f"({vout.series.nunique():,} series)", flush=True)
