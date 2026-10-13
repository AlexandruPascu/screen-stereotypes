"""Step 1a: gender and ethnic bias in the language of plot summaries, by decade (Garg et al. 2018, PNAS).

Word lists are Garg et al.'s published lists (gender word pairs, occupations, Williams-Best and other adjectives,
surnames by group, religion and terrorism words; DATA/lexicons/garg2018). Embeddings are word2vec skip-gram models
trained on our own summaries. Early decades are small (3–6M words), so instead of one model per decade this uses
temporal referencing (Dubossarsky et al. 2019): one model on all text, where each *target* word (occupation,
adjective) carries its decade ("nurse_1960") and the group words (she/he, surnames, religion words) are shared anchors.
Three seeds per run; results are averaged and their spread reported.

Bias of a target word t towards group A vs group B (Garg's relative norm distance, per word):
    bias(t) = ||t - mean(B)|| - ||t - mean(A)||   (unit vectors; > 0 means t sits closer to A)
For gender A = women's words, B = men's words.
Validation as in the paper: occupation bias vs women's share of that occupation in the US census (Garg's IPUMS-derived
table), decade by decade; plus our own on-screen casting share (results/occupations).
Outputs: results/embeddings/*.csv (figures drawn by figures.py).
"""
import re
import sys

import duckdb
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from paths import DERIVED, IMDB, LEXICONS, RESULTS

OUT = RESULTS / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)
G = LEXICONS / "garg2018"
SEEDS = [1, 2, 3] if len(sys.argv) < 2 else [int(x) for x in sys.argv[1].split(",")]
words = lambda f: [w.strip().lower() for w in open(G / f) if w.strip() and not w.startswith("#")]

FEMALE, MALE = words("female_pairs.txt"), words("male_pairs.txt")
OCCUPATIONS = words("occupations1950.txt")
ADJ = {"williams-best": words("adjectives_williamsbest.txt"), "appearance": words("adjectives_appearance.txt"),
       "intelligence": words("adjectives_intelligencegeneral.txt"), "sensitive": words("adjectives_sensitive.txt"),
       "otherization": words("adjectives_otherization.txt"), "terrorism": words("words_terrorism.txt")}
NAMES = {g: words(f"names_{g}.txt") for g in ["white", "asian", "chinese", "hispanic", "russian", "black"]}
RELIGION = {"islam": words("words_islam.txt"), "christianity": words("words_christianity.txt")}
TARGETS = set(OCCUPATIONS) | {w for ws in ADJ.values() for w in ws}
ANCHORS = set(FEMALE) | set(MALE) | {w for ws in NAMES.values() for w in ws} | {w for ws in RELIGION.values() for w in ws}
TARGETS -= ANCHORS  # a word is either tagged by decade or a shared anchor, never both
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def period(year):
    return "pre1950" if year < 1950 else str(int(year // 10 * 10))


con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")
texts = con.execute(f"""
    WITH t AS (SELECT title, arg_max(text, words) AS text, arg_max(kind, words) AS kind
               FROM '{DERIVED / "texts.parquet"}' WHERE words >= 5 GROUP BY title),
    y AS (SELECT tconst AS title, TRY_CAST(startYear AS INT) AS year FROM read_csv('{IMDB / "title.basics.tsv.gz"}',
          delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','titleType':'VARCHAR',
          'primaryTitle':'VARCHAR','originalTitle':'VARCHAR','isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR',
          'runtimeMinutes':'VARCHAR','genres':'VARCHAR'}})
          UNION ALL SELECT 'tmdb:' || series || ':' || season || ':' || episode, year FROM '{DERIVED / "tmdb_episodes.parquet"}')
    SELECT t.text, y.year, t.kind FROM t JOIN y USING (title) WHERE y.year BETWEEN 1900 AND 2026""").df()
print(f"summaries with a year: {len(texts):,}", flush=True)


class Corpus:
    """Streams tokenised summaries; target words get their decade tag."""

    def __iter__(self):
        for text, year in zip(texts.text, texts.year):
            p = period(year)
            yield [f"{w}_{p}" if w in TARGETS else w for w in TOKEN.findall(text.lower())]


def unit(v):
    return v / np.linalg.norm(v)


def centroid(kv, ws):
    vs = [unit(kv[w]) for w in ws if w in kv]
    return unit(np.mean(vs, axis=0)) if vs else None, len(vs)


def bias(kv, target, a, b):
    if target not in kv:
        return np.nan
    t = unit(kv[target])
    return np.linalg.norm(t - b) - np.linalg.norm(t - a)


PERIODS = ["pre1950"] + [str(d) for d in range(1950, 2030, 10)]
rows = []
for seed in SEEDS:
    model = Word2Vec(Corpus(), vector_size=300, window=5, min_count=10, sg=1, negative=5, epochs=5, workers=12, seed=seed)
    kv = model.wv
    f, nf = centroid(kv, FEMALE)
    m, nm = centroid(kv, MALE)
    print(f"seed {seed}: vocabulary {len(kv):,}; gender anchors found {nf}/{len(FEMALE)} women, {nm}/{len(MALE)} men",
          flush=True)
    for p in PERIODS:
        for w in OCCUPATIONS:
            rows.append(("gender", "occupation", w, p, seed, bias(kv, f"{w}_{p}", f, m)))
        for name, ws in ADJ.items():
            for w in ws:
                rows.append(("gender", f"adjective:{name}", w, p, seed, bias(kv, f"{w}_{p}", f, m)))
        white, _ = centroid(kv, NAMES["white"])
        for grp in ["asian", "chinese", "hispanic", "russian", "black"]:
            c, n = centroid(kv, NAMES[grp])
            if c is None or n < 5:
                continue
            for name in ["otherization", "williams-best", "intelligence", "appearance"]:
                for w in ADJ[name]:
                    rows.append((f"{grp} vs white surnames", f"adjective:{name}", w, p, seed, bias(kv, f"{w}_{p}", c, white)))
        isl, _ = centroid(kv, RELIGION["islam"])
        chr_, _ = centroid(kv, RELIGION["christianity"])
        for w in ADJ["terrorism"]:
            rows.append(("islam vs christianity", "terrorism", w, p, seed, bias(kv, f"{w}_{p}", isl, chr_)))
    model.save(str(DERIVED / f"w2v_temporal_seed{seed}.model"))

b = pd.DataFrame(rows, columns=["contrast", "set", "word", "period", "seed", "bias"]).dropna()
per_word = b.groupby(["contrast", "set", "word", "period"]).bias.agg(["mean", "std", "count"]).reset_index()
per_word.to_csv(OUT / "word_bias_by_period.csv", index=False)
summary = per_word.groupby(["contrast", "set", "period"])["mean"].agg(["mean", "count"]).reset_index()
summary.to_csv(OUT / "set_bias_by_period.csv", index=False)
pd.set_option("display.width", 200)
print(summary.pivot_table(index=["contrast", "set"], columns="period", values="mean").round(3).to_string(), flush=True)

def corr(d, col):
    return pd.Series({"occupations": len(d), "r": np.corrcoef(d["mean"], d[col])[0, 1] if len(d) >= 5 else np.nan})


# Validation 1: occupation bias vs census women's share (Garg's table), per census decade
census = pd.read_csv(G / "occupation_percentages_gender_occ1950.csv")
census["period"] = (census["Census year"] // 10 * 10).astype(int).astype(str)
census = census.groupby(["Occupation", "period"]).Female.mean().reset_index()
census["logit_women"] = np.log(census.Female.clip(0.01, 0.99) / (1 - census.Female.clip(0.01, 0.99)))
occ = per_word[(per_word.contrast == "gender") & (per_word.set == "occupation")]
v = occ.merge(census, left_on=["word", "period"], right_on=["Occupation", "period"])
val = v.groupby("period").apply(corr, "logit_women", include_groups=False).reset_index()
val.to_csv(OUT / "validation_census.csv", index=False)
print("occupation bias vs census women's share (correlation by decade):\n", val.round(3).to_string(index=False), flush=True)

# Validation 2: occupation bias vs our on-screen casting share (study 1), by era
cast = pd.read_csv(RESULTS / "occupations/occupation_by_decade_english.csv").groupby(["occupation", "decade"]).agg(
    roles=("roles", "sum"), women=("women", "sum")).reset_index()
cast = cast[cast.roles >= 30]
cast["period"] = cast.decade.astype(int).astype(str)
cast["logit_cast"] = np.log((cast.women + 0.5) / (cast.roles - cast.women + 0.5))
v2 = occ.merge(cast, left_on=["word", "period"], right_on=["occupation", "period"])
val2 = v2.groupby("period").apply(corr, "logit_cast", include_groups=False).reset_index()
val2.to_csv(OUT / "validation_casting.csv", index=False)
print("occupation bias vs on-screen casting share (correlation by decade):\n", val2.round(3).to_string(index=False))
