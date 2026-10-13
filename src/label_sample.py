"""Step 3: the sample of summaries for the labelling model (codebook in labelling/).

3,000 summaries of 15–350 words, text only (no title names, so the labeller cannot lean on what it knows):
  random    1,400: 6 equal cells (film/TV × before 1970 / 1970–99 / 2000+) — true prevalence, and recall of each measure
  targeted  1,600: what each measure flags (classifier top 2%, viewer-tagged, group + pejorative words, group without,
            slurs, women described by looks, women as victims, most and least gender-coded wording) — their precision
Each targeted stratum records how many summaries were eligible, so estimates can be weighted. 300 items, drawn across
all strata, are marked for a second labelling pass (label reliability).
Outputs (private: they hold text): DERIVED/labelling/items.jsonl and DERIVED/labelling/key.parquet.
"""
import json

import duckdb
import numpy as np
import pandas as pd

from paths import DERIVED, IMDB

OUT = DERIVED / "labelling"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")
t = con.execute(f"""
    WITH t AS (SELECT title, arg_max(text, words) AS text, max(words) AS words, arg_max(source, words) AS source,
                      arg_max(kind, words) AS kind FROM '{DERIVED / "texts.parquet"}' GROUP BY title),
    y AS (SELECT tconst AS title, TRY_CAST(startYear AS INT) AS year FROM read_csv('{IMDB / "title.basics.tsv.gz"}',
          delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','titleType':'VARCHAR',
          'primaryTitle':'VARCHAR','originalTitle':'VARCHAR','isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR',
          'runtimeMinutes':'VARCHAR','genres':'VARCHAR'}})
          UNION ALL SELECT 'tmdb:' || series || ':' || season || ':' || episode, year FROM '{DERIVED / "tmdb_episodes.parquet"}')
    SELECT t.*, y.year FROM t LEFT JOIN y USING (title) WHERE t.words BETWEEN 15 AND 350""").df()
t["medium"] = np.where(t.kind == "film", "film", "tv")
t["era"] = pd.cut(t.year, [0, 1969, 1999, 2100], labels=["before 1970", "1970-99", "2000+"]).astype("string")
clf = pd.read_parquet(DERIVED / "title_classifier_scores.parquet", columns=["title", "stereotype_prob", "y"])
off = pd.read_parquet(DERIVED / "title_groups_offensive.parquet")
full = pd.read_parquet(DERIVED / "title_groups.parquet", columns=["title", "mention_mask", "stereotype_mask"]).rename(
    columns={"stereotype_mask": "any_stereotype_mask"})
gt = pd.read_parquet(DERIVED / "title_gender_text.parquet", columns=["title", "women_mentions", "women_appearance",
                                                                       "women_victim"])
emb = pd.read_parquet(DERIVED / "title_embedding_scores.parquet", columns=["title", "gender_congruence"])
t = (t.merge(clf, on="title", how="left").merge(off, on="title", how="left")
      .merge(full.drop(columns="mention_mask"), on="title", how="left").merge(gt, on="title", how="left")
      .merge(emb, on="title", how="left"))
for c in ["mention_mask", "stereotype_mask", "slur_mask", "any_stereotype_mask"]:
    t[c] = t[c].fillna(0).astype("int64")
t["group_offensive"] = (t.mention_mask & t.stereotype_mask) != 0
t["group_mention_only"] = (t.mention_mask != 0) & ((t.mention_mask & t.any_stereotype_mask) == 0)
print(f"eligible summaries (15–350 words): {len(t):,}", flush=True)

chosen, rows = set(), []


def draw(stratum, pool, n):
    pool = pool[~pool.title.isin(chosen)]
    take = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1e9)))
    chosen.update(take.title)
    rows.append(take.assign(stratum=stratum, eligible=len(pool)))
    print(f"  {stratum}: {len(take):,} of {len(pool):,} eligible", flush=True)


known = t.era.notna()
for medium in ["film", "tv"]:
    for era in ["before 1970", "1970-99", "2000+"]:
        draw(f"random {medium} {era}", t[known & (t.medium == medium) & (t.era == era)], 1400 // 6 + (1 if era == "2000+" else 0))
top = t.stereotype_prob.quantile(0.98)
draw("classifier top 2%", t[t.stereotype_prob >= top], 250)
draw("viewer-tagged stereotype", t[t.y == 1], 200)
draw("group + pejorative words", t[t.group_offensive], 250)
draw("group, no stereotype words", t[t.group_mention_only], 150)
draw("slur detected", t[t.slur_mask != 0], 150)
draw("woman described by looks", t[t.women_appearance > 0], 200)
draw("woman as victim", t[t.women_victim > 0], 150)
hi, lo = t.gender_congruence.quantile(0.95), t.gender_congruence.quantile(0.05)
draw("most gender-coded wording (top 5%)", t[t.gender_congruence >= hi], 150)
draw("least gender-coded wording (bottom 5%)", t[t.gender_congruence <= lo], 100)

s = pd.concat(rows).sample(frac=1, random_state=7).reset_index(drop=True)
s["id"] = [f"L{i:05d}" for i in range(1, len(s) + 1)]
s["double"] = False
s.loc[s.sample(300, random_state=11).index, "double"] = True
with open(OUT / "items.jsonl", "w") as f:
    for i, text, d in zip(s.id, s.text, s.double):
        f.write(json.dumps({"id": i, "text": text, "double": bool(d)}, ensure_ascii=False) + "\n")
s.drop(columns=["text"]).to_parquet(OUT / "key.parquet", index=False)
print(f"items: {len(s):,} ({s.words.sum():,} words, ~{s.words.sum() * 1.4 / 1e6:.1f}M tokens of text); "
      f"double-labelled: {s.double.sum()}", flush=True)
