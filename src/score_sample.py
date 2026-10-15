"""Use the label-trained classifier (label_classifier.py) on a fresh random sample of summaries, for trends.

Sample: up to 2,000 summaries per decade (1930s–2020s) and medium (film/TV), 15–350 words like the labelled ones, none
of them among the 3,000 labelled items. Features: EmbeddingGemma-2 (text encoder, "Classification" prompt) plus this
project's hand-built measures, the best combination for portrayals in label_classifier.py; one encoder for all targets
(Qwen3 is 3x slower on a CPU and only ties or edges it on themes and group stereotypes).
Probabilities: a logistic regression per target trained on all 3,000 labels, then mapped to population rates with a
calibration fitted on the out-of-fold scores of the 1,400 random labelled items.
Outputs: results/label_classifier/sample_*.csv; DERIVED/label_classifier_sample_scores.parquet (private).
"""
import json
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from paths import DERIVED, IMDB, RESULTS

torch.set_num_threads(8)
OUT = RESULTS / "label_classifier"
L = DERIVED / "labelling"
PER_CELL = 2000
TARGETS = ["any_stereotype", "gender_any", "group_stereotyped", "prejudice_theme"]
con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")


def measures_of(t):
    return pd.DataFrame({
        "looks": t.women_appearance.fillna(0) > 0, "victim": t.women_victim.fillna(0) > 0,
        "coded": t.gender_congruence.fillna(0), "coded_missing": t.gender_congruence.isna(),
        "group_pejorative": t.group_offensive.fillna(False), "group_only": t.group_mention_only.fillna(False),
        "slur": t.slur_mask.fillna(0) != 0, "viewer_tag": t.y.fillna(0) > 0, "theme_classifier": t.stereotype_prob.fillna(0),
        "log_words": np.log1p(t.words)}).astype(float).to_numpy()


def attach_measures(t):
    clf = pd.read_parquet(DERIVED / "title_classifier_scores.parquet", columns=["title", "stereotype_prob", "y"])
    off = pd.read_parquet(DERIVED / "title_groups_offensive.parquet")
    full = pd.read_parquet(DERIVED / "title_groups.parquet", columns=["title", "stereotype_mask"]).rename(
        columns={"stereotype_mask": "any_stereotype_mask"})
    gt = pd.read_parquet(DERIVED / "title_gender_text.parquet", columns=["title", "women_appearance", "women_victim"])
    emb = pd.read_parquet(DERIVED / "title_embedding_scores.parquet", columns=["title", "gender_congruence"])
    t = (t.merge(clf, on="title", how="left").merge(off, on="title", how="left").merge(full, on="title", how="left")
          .merge(gt, on="title", how="left").merge(emb, on="title", how="left"))
    for c in ["mention_mask", "stereotype_mask", "slur_mask", "any_stereotype_mask"]:
        t[c] = t[c].fillna(0).astype("int64")
    t["group_offensive"] = (t.mention_mask & t.stereotype_mask) != 0
    t["group_mention_only"] = (t.mention_mask != 0) & ((t.mention_mask & t.any_stereotype_mask) == 0)
    return t


# --- the labelled 3,000: features, labels, and out-of-fold scores for calibration --------------------------------
items = pd.DataFrame([json.loads(line) for line in open(L / "items.jsonl")])
labels = [json.loads(line) for line in open(L / "labels.jsonl")]
y = pd.DataFrame({"id": [r["id"] for r in labels],
                  "gender_any": [r["gender"]["portrayal"] != "none" for r in labels],
                  "group_stereotyped": [any(g["portrayal"] == "stereotyped" for g in r["groups"]) for r in labels],
                  "prejudice_theme": [r["prejudice_theme"]["present"] for r in labels]})
y["any_stereotype"] = y.gender_any | y.group_stereotyped
lab = items.merge(y, on="id").merge(pd.read_parquet(L / "key.parquet"), on="id").sort_values("id", ignore_index=True)
lab["random"] = lab.stratum.str.startswith("random")
parents = con.execute(f"""SELECT tconst AS title, parentTconst FROM read_csv('{IMDB / "title.episode.tsv.gz"}',
    delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','parentTconst':'VARCHAR',
    'seasonNumber':'VARCHAR','episodeNumber':'VARCHAR'}})""").df()
lab = lab.merge(parents, on="title", how="left")
lab["work"] = lab.parentTconst.fillna(lab.title.str.extract(r"^(tmdb:\d+)", expand=False)).fillna(lab.title)
X_lab = np.hstack([np.load(DERIVED / "label_embeddings/gemma2.npy"), measures_of(lab)])

models, calibrators = {}, {}
for target in TARGETS:
    yv = lab[target].astype(int).to_numpy()
    best, best_auc = 1.0, -1
    from sklearn.metrics import roc_auc_score
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        aucs = []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X_lab, yv, groups=lab.work):
            sc = StandardScaler().fit(X_lab[tr])
            m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X_lab[tr]), yv[tr])
            aucs.append(roc_auc_score(yv[te], m.predict_proba(sc.transform(X_lab[te]))[:, 1]))
        if np.mean(aucs) > best_auc:
            best, best_auc = C, np.mean(aucs)
    oof = np.zeros(len(lab))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(X_lab, yv, groups=lab.work):
        sc = StandardScaler().fit(X_lab[tr])
        m = LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X_lab[tr]), yv[tr])
        oof[te] = m.decision_function(sc.transform(X_lab[te]))
    rnd = lab.random.to_numpy()
    calibrators[target] = LogisticRegression(C=1e6, max_iter=1000).fit(oof[rnd][:, None], yv[rnd])  # Platt, random items
    sc = StandardScaler().fit(X_lab)
    models[target] = (sc, LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X_lab), yv))
    print(f"{target}: C={best}, inner AUC {best_auc:.3f}", flush=True)

# --- the fresh sample ---------------------------------------------------------------------------------------------
t = con.execute(f"""
    WITH t AS (SELECT title, arg_max(text, words) AS text, max(words) AS words, arg_max(kind, words) AS kind
               FROM '{DERIVED / "texts.parquet"}' GROUP BY title),
    y AS (SELECT tconst AS title, TRY_CAST(startYear AS INT) AS year FROM read_csv('{IMDB / "title.basics.tsv.gz"}',
          delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','titleType':'VARCHAR',
          'primaryTitle':'VARCHAR','originalTitle':'VARCHAR','isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR',
          'runtimeMinutes':'VARCHAR','genres':'VARCHAR'}})
          UNION ALL SELECT 'tmdb:' || series || ':' || season || ':' || episode, year FROM '{DERIVED / "tmdb_episodes.parquet"}')
    SELECT t.*, y.year FROM t JOIN y USING (title) WHERE t.words BETWEEN 15 AND 350 AND y.year BETWEEN 1930 AND 2026
    ORDER BY t.title""").df()
t = t[~t.title.isin(lab.title)]
t["medium"] = np.where(t.kind == "film", "film", "tv")
t["decade"] = (t.year // 10 * 10).astype(int)
t["cell_size"] = t.groupby(["decade", "medium"]).title.transform("size")
s = t.sample(frac=1, random_state=2026).groupby(["decade", "medium"]).head(PER_CELL).reset_index(drop=True)
s = attach_measures(s)
print(f"sample {len(s):,} summaries; cells: {s.groupby(['decade', 'medium']).size().to_dict()}", flush=True)

cache = DERIVED / "label_embeddings/sample_gemma2.npy"
if cache.exists() and np.load(cache).shape[0] == len(s):
    E = np.load(cache)
else:
    from sentence_transformers import SentenceTransformer
    started = time.time()
    model = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
    parts = []
    for i in range(0, len(s), 2000):
        parts.append(model.encode(s.text.iloc[i:i + 2000].tolist(), prompt_name="Classification", batch_size=16,
                                  normalize_embeddings=True, show_progress_bar=False))
        print(f"  encoded {i + len(parts[-1]):,}/{len(s):,} ({(time.time() - started) / 60:.0f} min)", flush=True)
    E = np.vstack(parts)
    np.save(cache, E)
X = np.hstack([E, measures_of(s)])
for target in TARGETS:
    sc, m = models[target]
    s[target] = calibrators[target].predict_proba(m.decision_function(sc.transform(X))[:, None])[:, 1]

# weights: each cell's sample stands for all eligible summaries in that cell
s["weight"] = s.cell_size / s.groupby(["decade", "medium"]).title.transform("size")
s.drop(columns=["text"]).to_parquet(DERIVED / "label_classifier_sample_scores.parquet", index=False)
by = s.groupby(["decade", "medium"])[TARGETS].mean().reset_index()
by["summaries"] = s.groupby(["decade", "medium"]).size().values
by.to_csv(OUT / "sample_prevalence_by_decade.csv", index=False)
genres = con.execute(f"""
    WITH b AS (SELECT tconst, genres FROM read_csv('{IMDB / "title.basics.tsv.gz"}', delim='\\t', header=true, quote='',
               escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','titleType':'VARCHAR','primaryTitle':'VARCHAR',
               'originalTitle':'VARCHAR','isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR',
               'runtimeMinutes':'VARCHAR','genres':'VARCHAR'}}))
    SELECT b.tconst AS title, COALESCE(p.genres, b.genres) AS genres FROM b
    LEFT JOIN read_csv('{IMDB / "title.episode.tsv.gz"}', delim='\\t', header=true, quote='', escape='', nullstr='\\N',
        columns={{'tconst':'VARCHAR','parentTconst':'VARCHAR','seasonNumber':'VARCHAR','episodeNumber':'VARCHAR'}}) e
        ON e.tconst = b.tconst
    LEFT JOIN b p ON p.tconst = e.parentTconst""").df()
g = s.merge(genres, on="title", how="left")
g = g.assign(genre=g.genres.fillna("").str.split(",")).explode("genre")
g = g[g.genre != ""]
bg = g.groupby("genre").apply(lambda x: pd.Series({**{t: np.average(x[t], weights=x.weight) for t in TARGETS},
                                                     "summaries": len(x)}), include_groups=False).reset_index()
bg[bg.summaries >= 300].sort_values("any_stereotype").to_csv(OUT / "sample_prevalence_by_genre.csv", index=False)
pd.set_option("display.width", 200)
print(by.pivot(index="decade", columns="medium", values="any_stereotype").round(3).to_string())
print(bg[bg.summaries >= 300].sort_values("any_stereotype").round(3).to_string(index=False))
