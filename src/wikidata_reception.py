"""Critics and awards from Wikidata (CC0), for study 12: Rotten Tomatoes and Metacritic scores, the two sites' page ids,
and nominations and wins at the Academy Awards, BAFTA Film Awards, Golden Globes and Primetime Emmys.

Scores: Wikidata's "review score" statements (P444) by Rotten Tomatoes or Metacritic, with how they were determined
(Tomatometer, average rating, Metascore...), when, and from how many reviews; the latest per title and method is
kept. Page ids: Rotten Tomatoes (P1258) and Metacritic (P1712), used by reception_fetch.py to fill gaps. Awards: every
category of the four ceremonies, one query per category; for people, the work they were nominated for (P1686).
Everything is matched to IMDb ids (P345). Queries go one at a time with a pause, as the query service asks.
Outputs (private): DERIVED/wikidata/{scores,ids,awards}.parquet.
"""
import io
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from paths import DERIVED

OUT = DERIVED / "wikidata"
OUT.mkdir(exist_ok=True)
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "screen-stereotypes-research/1.0 (https://github.com/AlexandruPascu/screen-stereotypes)"
CRITICS = {"Q105584": "rotten_tomatoes", "Q150248": "metacritic"}
FAMILIES = {"Q19020": "Academy Awards", "Q732997": "BAFTA Film Awards", "Q1011547": "Golden Globes",
            "Q1044427": "Primetime Emmys"}


def sparql(query, tries=4):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/csv"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = r.read().decode("utf-8")
            time.sleep(1.5)
            return pd.read_csv(io.StringIO(text), dtype=str).apply(
                lambda c: c.str.replace("http://www.wikidata.org/entity/", "", regex=False))
        except Exception as e:  # timeouts and 429s: wait longer each time
            print(f"  query failed ({e}); retrying", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Wikidata query kept failing")


def number(value):
    """'93%' -> 93; '6.8/10' -> 6.8; '75/100' -> 75; anything else -> None."""
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:%|/\s*(?:10|100|5))?\s*$", str(value))
    return float(m.group(1)) if m else None


# --- scores ---------------------------------------------------------------------------------------------------------
rows = []
for qid, critic in CRITICS.items():
    d = sparql(f"""SELECT ?imdb ?score ?method ?methodLabel ?when ?reviews WHERE {{
        ?film p:P444 ?st ; wdt:P345 ?imdb . ?st ps:P444 ?score ; pq:P447 wd:{qid} .
        OPTIONAL {{ ?st pq:P459 ?method . }} OPTIONAL {{ ?st pq:P585 ?when . }} OPTIONAL {{ ?st pq:P7887 ?reviews . }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}""")
    rows.append(d.assign(critic=critic))
    print(f"{critic}: {len(d):,} score statements", flush=True)
scores = pd.concat(rows, ignore_index=True)
scores["value"] = scores.score.map(number)
scores["when"] = pd.to_datetime(scores["when"], errors="coerce", utc=True)
scores["reviews"] = pd.to_numeric(scores.reviews, errors="coerce")
scores["method"] = scores.methodLabel.fillna("unspecified")
scores = (scores.dropna(subset=["value"]).sort_values("when", na_position="first")
          .drop_duplicates(["imdb", "critic", "method"], keep="last"))
scores[["imdb", "critic", "method", "value", "reviews", "when"]].to_parquet(OUT / "scores.parquet", index=False)
print(scores.groupby(["critic", "method"]).size().to_string(), flush=True)

# --- page ids -------------------------------------------------------------------------------------------------------
ids = []
for prop, name in [("P1258", "rotten_tomatoes_id"), ("P1712", "metacritic_id")]:
    d = sparql(f"SELECT ?imdb ?id WHERE {{ ?item wdt:{prop} ?id ; wdt:P345 ?imdb . FILTER(STRSTARTS(?imdb, 'tt')) }}")
    ids.append(d.drop_duplicates("imdb").set_index("imdb").rename(columns={"id": name}))
    print(f"{name}: {len(d):,}", flush=True)
pd.concat(ids, axis=1).reset_index().rename(columns={"index": "imdb"}).to_parquet(OUT / "ids.parquet", index=False)

# --- awards ---------------------------------------------------------------------------------------------------------
awards = []
for fam, family in FAMILIES.items():
    cats = sparql(f"""SELECT ?cat ?catLabel WHERE {{ ?cat wdt:P31 wd:{fam} .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}""")
    for cat, label in zip(cats.cat, cats.catLabel):
        for prop, outcome in [("P1411", "nominated"), ("P166", "won")]:
            d = sparql(f"""SELECT ?subject ?work ?when WHERE {{
                ?item p:{prop} ?st ; wdt:P345 ?subject . ?st ps:{prop} wd:{cat} .
                OPTIONAL {{ ?st pq:P1686 ?w . ?w wdt:P345 ?work . }} OPTIONAL {{ ?st pq:P585 ?when . }} }}""")
            if len(d):
                awards.append(d.assign(family=family, category=label, outcome=outcome))
    print(f"{family}: {len(cats)} categories", flush=True)
awards = pd.concat(awards, ignore_index=True)
awards["film"] = awards.work.where(awards.work.notna(), awards.subject.where(awards.subject.str.startswith("tt")))
awards["person"] = awards.subject.where(awards.subject.str.startswith("nm"))
awards["year"] = pd.to_datetime(awards["when"], errors="coerce", utc=True).dt.year
awards = awards[["family", "category", "outcome", "film", "person", "year"]].drop_duplicates()
awards.to_parquet(OUT / "awards.parquet", index=False)
print(awards.groupby(["family", "outcome"]).size().to_string(), flush=True)
