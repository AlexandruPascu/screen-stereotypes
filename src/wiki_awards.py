"""Golden Globe, BAFTA and Primetime Emmy nominations from Wikipedia's award-category pages (CC BY-SA 4.0), for
study 12; Wikidata holds only a few hundred of them.

Each category page lists every year's nominees in tables, the winner first and highlighted; the film or series is the
italic link in each row (also in acting categories, where it is the work the performance was in). Pages come from
the Wikipedia API (parsed HTML), paced, with a descriptive User-Agent; titles are traced to IMDb ids through their
Wikidata items. Categories: Golden Globes for films and for TV series, BAFTA film awards, and the Primetime Emmys for
drama and comedy series and their actors.
Output (private): DERIVED/wikipedia_awards.parquet (award, category, year, title, won, imdb).
"""
import re
import time

import lxml.html
import pandas as pd
import requests

from paths import DERIVED

API = "https://en.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"
UA = "screen-stereotypes research (non-commercial; https://github.com/AlexandruPascu/screen-stereotypes)"
PAGES = {
    "Golden Globes (film)": [
        "Golden Globe Award for Best Motion Picture – Drama", "Golden Globe Award for Best Motion Picture – Musical or Comedy",
        "Golden Globe Award for Best Director", "Golden Globe Award for Best Screenplay",
        "Golden Globe Award for Best Actor – Motion Picture Drama", "Golden Globe Award for Best Actress in a Motion Picture – Drama",
        "Golden Globe Award for Best Actor – Motion Picture Musical or Comedy",
        "Golden Globe Award for Best Actress – Motion Picture Comedy or Musical",
        "Golden Globe Award for Best Supporting Actor – Motion Picture",
        "Golden Globe Award for Best Supporting Actress – Motion Picture"],
    "Golden Globes (TV)": [
        "Golden Globe Award for Best Television Series – Drama", "Golden Globe Award for Best Television Series – Musical or Comedy",
        "Golden Globe Award for Best Actor – Television Series Drama", "Golden Globe Award for Best Actress – Television Series Drama",
        "Golden Globe Award for Best Actor – Television Series Musical or Comedy",
        "Golden Globe Award for Best Actress – Television Series Musical or Comedy"],
    "BAFTA (film)": [
        "BAFTA Award for Best Film", "BAFTA Award for Best Direction", "BAFTA Award for Best Actor in a Leading Role",
        "BAFTA Award for Best Actress in a Leading Role", "BAFTA Award for Best Actor in a Supporting Role",
        "BAFTA Award for Best Actress in a Supporting Role", "BAFTA Award for Best Original Screenplay",
        "BAFTA Award for Best Adapted Screenplay"],
    "Primetime Emmys": [
        "Primetime Emmy Award for Outstanding Drama Series", "Primetime Emmy Award for Outstanding Comedy Series",
        "Primetime Emmy Award for Outstanding Lead Actor in a Drama Series",
        "Primetime Emmy Award for Outstanding Lead Actress in a Drama Series",
        "Primetime Emmy Award for Outstanding Lead Actor in a Comedy Series",
        "Primetime Emmy Award for Outstanding Lead Actress in a Comedy Series",
        "Primetime Emmy Award for Outstanding Supporting Actor in a Drama Series",
        "Primetime Emmy Award for Outstanding Supporting Actress in a Drama Series",
        "Primetime Emmy Award for Outstanding Supporting Actor in a Comedy Series",
        "Primetime Emmy Award for Outstanding Supporting Actress in a Comedy Series"],
}
s = requests.Session()
s.headers["User-Agent"] = UA


def get(url, **params):
    params.update(format="json", formatversion=2)
    for attempt in range(5):
        try:
            r = s.get(url, params=params, timeout=90)
            if r.status_code == 200 and "error" not in r.json():
                time.sleep(0.5)
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    return None


YEAR = re.compile(r"\b(19[3-9]\d|20[0-2]\d)\b")
rows = []
for award, pages in PAGES.items():
    for page in pages:
        d = get(API, action="parse", page=page, prop="text", redirects=1)
        if d is None:
            print(f"  missing page: {page}", flush=True)
            continue
        doc = lxml.html.fromstring(d["parse"]["text"])
        # a work is linked only the first time a page names it; later rows give the bare italic title
        linked = {}
        for a in doc.xpath('//table[contains(@class, "wikitable")]//td//i//a[@href]'):
            href = a.get("href")
            if href.startswith("/wiki/") and ":" not in href[6:]:
                linked.setdefault(a.text_content().strip(), a.get("title") or requests.utils.unquote(href[6:]).replace("_", " "))
        found = 0
        for table in doc.xpath('//table[contains(@class, "wikitable")]'):
            year = None
            for tr in table.xpath(".//tr"):
                cells = tr.xpath("./th|./td")
                # the year cell opens each year's rows (a header cell on some pages, a spanning data cell on others);
                # a cell holding an italic title is a work, never a year (a film called "1917")
                if cells and not cells[0].xpath(".//i") and (cells[0].tag == "th" or cells[0].get("rowspan")
                                                              or len(cells) > 2):
                    m = YEAR.search(cells[0].text_content())
                    if m:
                        year = int(m.group(1))
                italics = tr.xpath("./td//i")
                if year is None or not italics:
                    continue
                links = italics[0].xpath(".//a[@href]")
                href = links[0].get("href") if links else ""
                if href.startswith("/wiki/") and ":" not in href[6:]:
                    title = links[0].get("title") or requests.utils.unquote(href[6:]).replace("_", " ")
                else:
                    title = linked.get(italics[0].text_content().strip())
                if not title:
                    continue
                won = ("background" in (tr.get("style") or "")
                       or any("background" in (td.get("style") or "") for td in tr.xpath("./td"))
                       or bool(tr.xpath("./td//b//i | ./td//i//b")))
                rows.append((award, page, year, title, won))
                found += 1
        print(f"{award}: {page.split(' for ')[-1]}: {found} nominations", flush=True)
aw = pd.DataFrame(rows, columns=["award", "category", "year", "title", "won"]).drop_duplicates()

# Wikipedia title -> Wikidata item -> IMDb id
titles = sorted(set(aw.title))
item = {}
for i in range(0, len(titles), 50):
    d = get(API, action="query", prop="pageprops", ppprop="wikibase_item", titles="|".join(titles[i:i + 50]), redirects=1)
    if not d:
        continue
    back = {r["to"]: r["from"] for r in d["query"].get("redirects", [])}
    norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
    for p in d["query"]["pages"]:
        q = (p.get("pageprops") or {}).get("wikibase_item")
        if q:
            t = back.get(p["title"], p["title"])
            item[norm.get(t, t)] = q
qids = sorted(set(item.values()))
imdb = {}
for i in range(0, len(qids), 50):
    d = get(WD, action="wbgetentities", ids="|".join(qids[i:i + 50]), props="claims")
    for q, e in (d or {}).get("entities", {}).items():
        for snak in e.get("claims", {}).get("P345", []):
            v = snak.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(v, str) and v.startswith("tt"):
                imdb[q] = v
                break
aw["imdb"] = aw.title.map(item).map(imdb)
aw.to_parquet(DERIVED / "wikipedia_awards.parquet", index=False)
print(f"nominations: {len(aw):,}; traced to IMDb: {aw.imdb.notna().sum():,}; winners flagged: {aw.won.sum():,}", flush=True)
print(aw.groupby("award").agg(nominations=("title", "size"), imdb=("imdb", "count"), won=("won", "sum"),
                              years=("year", lambda y: f"{y.min()}–{y.max()}")).to_string())
