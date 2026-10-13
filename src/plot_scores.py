"""Step 1b: a stereotype score for each plot summary, from pretrained word embeddings (GloVe, PDDL).

Gender: Garg et al.'s women's and men's word lists define a gender direction in GloVe; every word gets a score for how
feminine- or masculine-coded it is in general English (cosine with that direction). For each summary, the words used
for its women (what they do, what is done to them, how they are described; from parse_texts/agency) and for its men
are averaged:
    gender_congruence = mean score of words for women - mean score of words for men
> 0 means the summary describes its women in more feminine-coded and its men in more masculine-coded language than
the reverse. Explicitly gendered words (mother, king...) are left out.
Groups: for each group, the centroid of its pejorative stereotype words (lexicon.json "offensive_attributes"); a
summary's closeness to it is the cosine between the summary's average word vector and that centroid. Summaries that
mention the group are compared with summaries that do not (same decade): the gap catches stereotypes phrased with
words the lists do not contain.
Outputs: DERIVED/title_embedding_scores.parquet (private), results/embeddings/plot_*.csv and figures.
"""
import json
import re

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA, DERIVED, IMDB, LEXICONS, RESULTS

OUT = RESULTS / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)
G = LEXICONS / "garg2018"
words = lambda f: [w.strip().lower() for w in open(G / f) if w.strip()]
FEMALE, MALE = words("female_pairs.txt"), words("male_pairs.txt")
glove_path = next((DATA / "embeddings").rglob("glove-wiki-gigaword-300.gz"))
kv = KeyedVectors.load_word2vec_format(str(glove_path), binary=False)  # gensim-data ships GloVe in word2vec text format
kv.fill_norms()
unit = lambda v: v / np.linalg.norm(v)
direction = unit(np.mean([unit(kv[w]) for w in FEMALE if w in kv], axis=0) -
                 np.mean([unit(kv[w]) for w in MALE if w in kv], axis=0))
norms = kv.get_normed_vectors()
word_score = pd.Series(norms @ direction, index=kv.index_to_key)
common = word_score.iloc[:30000]  # GloVe orders words by frequency; rare tokens make a noisy sanity list
print("most feminine-coded common words:", ", ".join(common.sort_values().tail(15).index[::-1]))
print("most masculine-coded common words:", ", ".join(common.sort_values().head(15).index))

con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")
gendered = set(FEMALE) | set(MALE) | set("""wife husband mother father son daughter sister brother girlfriend boyfriend
bride groom queen king princess prince lady gentleman mom dad aunt uncle niece nephew widow widower actress actor
waitress waiter heroine hero mrs mr miss ms sir madam""".split())
con.register("scores", pd.DataFrame({"lemma": word_score.index, "score": word_score.values}))
t = con.execute(f"""SELECT g.title, any_value(g.decade) AS decade, any_value(g.medium) AS medium,
        AVG(s.score) FILTER (WHERE g.female) AS women_score, AVG(s.score) FILTER (WHERE NOT g.female) AS men_score,
        COUNT(*) FILTER (WHERE g.female) AS women_words, COUNT(*) FILTER (WHERE NOT g.female) AS men_words
    FROM '{DERIVED / "gendered_tuples.parquet"}' g JOIN scores s USING (lemma)
    WHERE g.lemma NOT IN (SELECT unnest({sorted(gendered)}))
    GROUP BY g.title""").df()
t["gender_congruence"] = (t.women_score - t.men_score).where((t.women_words >= 2) & (t.men_words >= 2))
print(f"summaries with words for both women and men: {t.gender_congruence.notna().sum():,}", flush=True)

genres = con.execute(f"""
    WITH e AS (SELECT tconst, parentTconst FROM read_csv('{IMDB / "title.episode.tsv.gz"}', delim='\\t', header=true,
               quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','parentTconst':'VARCHAR',
               'seasonNumber':'VARCHAR','episodeNumber':'VARCHAR'}})),
    b AS (SELECT tconst, genres FROM read_csv('{IMDB / "title.basics.tsv.gz"}', delim='\\t', header=true, quote='',
          escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','titleType':'VARCHAR','primaryTitle':'VARCHAR',
          'originalTitle':'VARCHAR','isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR',
          'runtimeMinutes':'VARCHAR','genres':'VARCHAR'}}))
    SELECT b.tconst AS title, COALESCE(pb.genres, b.genres) AS genres FROM b LEFT JOIN e USING (tconst)
    LEFT JOIN b pb ON pb.tconst = e.parentTconst""").df()
t = t.merge(genres, on="title", how="left")
c = t.dropna(subset=["gender_congruence"])
by_decade = c.groupby(["decade", "medium"]).agg(summaries=("gender_congruence", "size"),
                                                congruence=("gender_congruence", "mean"),
                                                women_score=("women_score", "mean"), men_score=("men_score", "mean"),
                                                se=("gender_congruence", "sem")).reset_index()
by_decade.to_csv(OUT / "plot_gender_congruence_by_decade.csv", index=False)
g = c.assign(genre=c.genres.fillna("").str.split(",")).explode("genre")
by_genre = g[g.genre != ""].groupby("genre").agg(summaries=("gender_congruence", "size"),
                                                  congruence=("gender_congruence", "mean")).reset_index()
by_genre = by_genre[by_genre.summaries >= 2000].sort_values("congruence")
by_genre.to_csv(OUT / "plot_gender_congruence_by_genre.csv", index=False)
pd.set_option("display.width", 200)
print(by_decade.pivot(index="decade", columns="medium", values="congruence").round(4).to_string())
print(by_genre.round(4).to_string(index=False), flush=True)

# groups: closeness of a summary's wording to the group's pejorative stereotype words
lex = json.load(open(DERIVED / "lexicon.json"))
groups = json.load(open(DERIVED / "title_groups_bits.json"))
tg = pd.read_parquet(DERIVED / "title_groups_offensive.parquet", columns=["title", "mention_mask"])
texts = con.execute(f"""SELECT title, arg_max(text, words) AS text FROM '{DERIVED / "texts.parquet"}' WHERE words >= 5
    GROUP BY title""").df().merge(tg, on="title").merge(t[["title", "decade"]], on="title", how="left")
TOKEN = re.compile(r"[a-z]+")
STOP = set("""the a an and or of to in on at for with his her he she they their is are was be by from as it its this
that who after when while into out up but not has have had will about him them be been being which what where there
than then also only just more most some such can could would should""".split())
mention_words = {w for v in lex.values() for m in v["mentions"] for w in m.lower().split()}
rng = np.random.default_rng(0)
baseline_idx = rng.choice(len(texts), size=min(300_000, len(texts)), replace=False)
mentioning = np.flatnonzero(texts.mention_mask.to_numpy() != 0)
idx = np.union1d(baseline_idx, mentioning)
vecs = np.zeros((len(idx), kv.vector_size), dtype=np.float32)
for k, i in enumerate(idx):
    ws = [w for w in TOKEN.findall(texts.text.iat[i].lower()) if w not in STOP and w not in mention_words and w in kv]
    if ws:
        vecs[k] = unit(np.mean([norms[kv.key_to_index[w]] for w in ws], axis=0))
sub = texts.iloc[idx][["title", "mention_mask", "decade"]].reset_index(drop=True)
rows, per_title = [], {}
for bit, grp in enumerate(groups):
    attrs = [w for w in lex[grp]["offensive_attributes"] if w in kv]
    if len(attrs) < 3:
        continue
    cvec = unit(np.mean([norms[kv.key_to_index[w]] for w in attrs], axis=0))
    sim = vecs @ cvec
    mentions = (sub.mention_mask.to_numpy() & (1 << bit)) != 0
    for title, s in zip(sub.title[mentions], sim[mentions]):
        per_title.setdefault(title, {})[grp] = float(s)
    d = pd.DataFrame({"decade": sub.decade, "mention": mentions, "sim": sim})
    a = d.groupby(["decade", "mention"]).sim.agg(["mean", "size"]).unstack()
    a.columns = [f"{x}_{'with' if y else 'without'}" for x, y in a.columns]
    a = a.reset_index()
    a["group"], a["gap"] = grp, a.mean_with - a.mean_without
    rows.append(a)
gl = pd.concat(rows)
gl.to_csv(OUT / "plot_group_stereotype_similarity_by_decade.csv", index=False)
overall = gl.groupby("group").apply(lambda d: np.average(d.gap.dropna(), weights=d.size_with[d.gap.notna()]),
                                    include_groups=False).sort_values(ascending=False)
print("gap in closeness to pejorative stereotype words, mentioning vs not (all decades):\n",
      overall.round(4).to_string(), flush=True)

t["group_similarity"] = t.title.map(lambda x: per_title.get(x))
t.drop(columns=["genres"]).to_parquet(DERIVED / "title_embedding_scores.parquet", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
for medium, colour in [("film", "#c0392b"), ("tv", "#2c3e50")]:
    d = by_decade[(by_decade.medium == medium) & (by_decade.decade >= 1950) & (by_decade.summaries >= 500)]
    ax.errorbar(d.decade, d.congruence, yerr=1.96 * d.se, marker="o", color=colour, capsize=3, label=medium)
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("gender congruence of wording\n(women's words more feminine-coded than men's → higher)")
ax.set_title("How gender-stereotyped is the wording around women and men in plot summaries?")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "plot_gender_congruence_by_decade.png", dpi=150)
