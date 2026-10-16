"""The sample of summaries for the social-groups labelling pass (labelling/codebook_social.md).

Summaries of 15–350 words, text only, as in the first pass (label_sample.py):
  random     1,400: the first pass's random summaries, labelled again for the six social groups — true prevalence,
             recall of each measure, and labels that combine with the first pass's
  targeted   what each social measure flags — their precision:
               mention: <group>                    100 per group, summaries the mention lists match
               group + pejorative words: <group>   50 per group, a mention together with a pejorative stereotype word
               pejorative label: <group>           35 per group, an insulting label for the group
               viewer-tagged social theme          150, titles viewers tagged with one of the six themes
               first pass: other discrimination    every first-pass summary noted "other discrimination"
Each targeted stratum records how many summaries were eligible, so estimates can be weighted. About 10% of items, drawn
across all strata, are marked for a second labelling pass.
Outputs (private: they hold text): DERIVED/labelling_social/items.jsonl and key.parquet.
"""
import json
import re

import duckdb
import numpy as np
import pandas as pd

from paths import DERIVED, IMDB

FIRST = DERIVED / "labelling"
OUT = DERIVED / "labelling_social"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2027)
con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")
t = con.execute(f"""
    WITH t AS (SELECT title, arg_max(text, words) AS text, max(words) AS words, arg_max(source, words) AS source,
                      arg_max(kind, words) AS kind,
                      string_agg(lower(keywords), '|') FILTER (WHERE keywords <> '') AS kw
               FROM '{DERIVED / "texts.parquet"}' GROUP BY title),
    y AS (SELECT tconst AS title, TRY_CAST(startYear AS INT) AS year FROM read_csv('{IMDB / "title.basics.tsv.gz"}',
          delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','titleType':'VARCHAR',
          'primaryTitle':'VARCHAR','originalTitle':'VARCHAR','isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR',
          'runtimeMinutes':'VARCHAR','genres':'VARCHAR'}})
          UNION ALL SELECT 'tmdb:' || series || ':' || season || ':' || episode, year FROM '{DERIVED / "tmdb_episodes.parquet"}')
    SELECT t.*, y.year FROM t LEFT JOIN y USING (title) WHERE t.words BETWEEN 15 AND 350""").df()
t["medium"] = np.where(t.kind == "film", "film", "tv")
t["era"] = pd.cut(t.year, [0, 1969, 1999, 2100], labels=["before 1970", "1970-99", "2000+"]).astype("string")
groups = json.load(open(DERIVED / "title_groups_bits_social.json"))
bit = {g: 1 << i for i, g in enumerate(groups)}
full = pd.read_parquet(DERIVED / "title_groups_social.parquet")
off = pd.read_parquet(DERIVED / "title_groups_social_offensive.parquet", columns=["title", "stereotype_mask"]).rename(
    columns={"stereotype_mask": "pejorative_word_mask"})
t = t.merge(full, on="title", how="left").merge(off, on="title", how="left")
for c in ["mention_mask", "stereotype_mask", "slur_mask", "pejorative_word_mask"]:
    t[c] = t[c].fillna(0).astype("int64")
TAGS = (r"immigra|refugee|asylum seeker|deportation|disabilit|wheelchair|blindness|deafness|paraplegi|amputee|autism|"
        r"down syndrome|mental illness|schizophreni|bipolar|psychiatric|mental hospital|mental patient|insanity|"
        r"old age|elderly|retirement home|nursing home|ageism|senior citizen|poverty|working class|slum|homeless")
t["tagged"] = t.kw.fillna("").str.contains(TAGS, regex=True)
print(f"eligible summaries (15–350 words): {len(t):,}", flush=True)

chosen, rows = set(), []


def draw(stratum, pool, n):
    pool = pool[~pool.title.isin(chosen)]
    take = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1e9)))
    chosen.update(take.title)
    rows.append(take.assign(stratum=stratum, eligible=len(pool)))
    print(f"  {stratum}: {len(take):,} of {len(pool):,} eligible", flush=True)


# the first pass's random summaries and its "other discrimination" notes, with their texts as labelled then
key = pd.read_parquet(FIRST / "key.parquet", columns=["id", "title", "stratum", "eligible"])
first_text = {r["id"]: r["text"] for r in map(json.loads, open(FIRST / "items.jsonl"))}
noted = {r["id"] for r in map(json.loads, open(FIRST / "labels.jsonl"))
         if re.search(r"other discrimination", r["notes"], re.I)}
for stratum, ids in [("random", key[key.stratum.str.startswith("random")]),
                     ("first pass: other discrimination", key[key.id.isin(noted)])]:
    for first_stratum, d in ids.groupby("stratum", sort=True) if stratum == "random" else [(stratum, ids)]:
        pool = t[t.title.isin(d.title)].drop(columns="text").merge(d[["title", "id"]].rename(columns={"id": "first_id"}),
                                                                  on="title")
        pool["text"] = pool.first_id.map(first_text)
        draw(first_stratum, pool, len(pool))
        if stratum == "random":  # the population the first pass drew these from, for weighting
            rows[-1]["eligible"] = d.eligible.iloc[0]
for g in groups:
    draw(f"mention: {g}", t[(t.mention_mask & bit[g]) != 0], 100)
for g in groups:
    draw(f"group + pejorative words: {g}", t[(t.mention_mask & t.pejorative_word_mask & bit[g]) != 0], 50)
for g in groups:
    draw(f"pejorative label: {g}", t[(t.slur_mask & bit[g]) != 0], 35)
draw("viewer-tagged social theme", t[t.tagged], 150)

s = pd.concat(rows).sample(frac=1, random_state=8).reset_index(drop=True)
s["id"] = [f"S{i:05d}" for i in range(1, len(s) + 1)]
s["double"] = False
s.loc[s.sample(round(len(s) * 0.1), random_state=12).index, "double"] = True
with open(OUT / "items.jsonl", "w") as f:
    for i, text, d in zip(s.id, s.text, s.double):
        f.write(json.dumps({"id": i, "text": text, "double": bool(d)}, ensure_ascii=False) + "\n")
s.drop(columns=["text", "kw"]).to_parquet(OUT / "key.parquet", index=False)
print(f"items: {len(s):,} ({s.words.sum():,} words, ~{s.words.sum() * 1.4 / 1e6:.1f}M tokens of text); "
      f"double-labelled: {s.double.sum()}", flush=True)
