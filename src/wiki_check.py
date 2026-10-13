"""How much episode text would Wikipedia add? Measures it before anything is downloaded in bulk.

1. Lists every article that uses the {{Episode list}} template (episode lists and season pages).
2. Reads a random sample and counts episodes with a ShortSummary, and the summaries' length.
3. Checks whether each sampled page leads, through Wikidata, to the series' IMDb id (needed to join our tables).
Writes DATA/wikipedia/episode_list_pages.txt (all page titles) and prints the estimate.
"""
import json
import random
import re
import time

import mwparserfromhell
import requests

from paths import DATA

API = "https://en.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"
UA = "screen-stereotypes research (non-commercial; https://github.com/AlexandruPascu/screen-stereotypes)"
OUT = DATA / "wikipedia"
OUT.mkdir(exist_ok=True)
s = requests.Session()
s.headers["User-Agent"] = UA


def get(url, **params):
    params.update(format="json", formatversion=2)
    if url == API:
        params["maxlag"] = 5  # Wikipedia; Wikidata reads here are few and paced, and its query-service lag blocks them
    for attempt in range(6):
        try:
            r = s.get(url, params=params, timeout=60)
            if r.status_code == 200 and "error" not in r.json():
                return r.json()
        except (requests.RequestException, ValueError):
            pass  # dropped connection or truncated reply: wait and retry
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Wikipedia API kept failing for {params.get('action')}")


pages, cont = [], {}
while True:
    d = get(API, action="query", list="embeddedin", eititle="Template:Episode list", einamespace=0, eilimit=500, **cont)
    pages += [p["title"] for p in d["query"]["embeddedin"]]
    if "continue" not in d:
        break
    cont = d["continue"]
(OUT / "episode_list_pages.txt").write_text("\n".join(pages))
print(f"articles using {{{{Episode list}}}}: {len(pages):,}", flush=True)

random.seed(1)
sample = random.sample(pages, 300)
entries = with_summary = 0
words, page_items, per_page = [], {}, []
for i in range(0, len(sample), 50):
    batch = sample[i:i + 50]
    d = get(API, action="query", prop="revisions|pageprops", rvprop="content", rvslots="main", ppprop="wikibase_item",
            titles="|".join(batch))
    for p in d["query"]["pages"]:
        text = p["revisions"][0]["slots"]["main"]["content"] if p.get("revisions") else ""
        page_items[p["title"]] = (p.get("pageprops") or {}).get("wikibase_item")
        n = k = 0
        for t in mwparserfromhell.parse(text).filter_templates():
            name = str(t.name).strip().lower()
            if not name.startswith("episode list"):
                continue
            n += 1
            if t.has("ShortSummary"):
                summary = mwparserfromhell.parse(str(t.get("ShortSummary").value)).strip_code().strip()
                if len(summary.split()) >= 5:
                    k += 1
                    words.append(len(summary.split()))
        entries += n
        with_summary += k
        per_page.append((p["title"], n, k))
    time.sleep(1)
print(f"sample of {len(sample)} pages: {entries:,} episode entries, {with_summary:,} with a summary of 5+ words "
      f"({with_summary / max(1, entries):.0%}); summary words median {sorted(words)[len(words) // 2] if words else 0}",
      flush=True)
scale = len(pages) / len(sample)
print(f"estimate for all pages: ~{entries * scale:,.0f} episode entries, ~{with_summary * scale:,.0f} with summaries",
      flush=True)

# Can a page be traced to the series' IMDb id? page item -> (P179 part of the series | P361 part of | itself) -> P345
items = [q for q in page_items.values() if q]
claims = {}
for i in range(0, len(items), 50):
    d = get(WD, action="wbgetentities", ids="|".join(items[i:i + 50]), props="claims")
    for q, e in d.get("entities", {}).items():
        claims[q] = e.get("claims", {})
    time.sleep(1)


def values(c, prop):
    out = []
    for snak in c.get(prop, []):
        v = snak.get("mainsnak", {}).get("datavalue", {}).get("value")
        out.append(v["id"] if isinstance(v, dict) and "id" in v else v)
    return out


parents = {q: values(c, "P179") + values(c, "P361") for q, c in claims.items()}
need = sorted({p for ps in parents.values() for p in ps if p})
for i in range(0, len(need), 50):
    d = get(WD, action="wbgetentities", ids="|".join(need[i:i + 50]), props="claims")
    for q, e in d.get("entities", {}).items():
        claims.setdefault(q, e.get("claims", {}))
    time.sleep(1)
linked = 0
for title, q in page_items.items():
    imdb = [v for v in values(claims.get(q, {}), "P345") if str(v).startswith("tt")] if q else []
    for parent in parents.get(q, []):
        imdb += [v for v in values(claims.get(parent, {}), "P345") if str(v).startswith("tt")]
    linked += bool(imdb)
print(f"sampled pages traced to a series IMDb id through Wikidata: {linked}/{len(page_items)} "
      f"({linked / max(1, len(page_items)):.0%})", flush=True)
json.dump(per_page, open(OUT / "sample_counts.json", "w"))
