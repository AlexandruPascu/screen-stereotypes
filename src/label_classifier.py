"""Step 1 of stage 2: a stereotype classifier trained on the 3,000 labelled summaries (labels from study 8).

Targets (from the labelling model's reading): any stereotyped portrayal, a gender stereotype, a group stereotype, and a
story about prejudice. Features compared, each with a logistic regression on top:
  tf-idf          word and word-pair counts
  glove           average GloVe word vector (2014)
  mpnet           all-mpnet-base-v2 sentence embedding (2021)
  qwen3           Qwen3-Embedding-0.6B (2025, Apache-2.0) with a task instruction
  gemma2          EmbeddingGemma-2, text encoder only (2026, Apache-2.0), with its "Classification" prompt
  measures        this project's hand-built measures (looks, victim, gender-coded wording, group words, slurs, viewer tag,
                  the prejudice-theme classifier, text length)
  best + measures the best embedding with the hand-built measures added
Evaluation: 5-fold cross-validation with every series kept whole on one side, repeated 3 times; AUC and average
precision on all items and on the 1,400 random items only (the fair test for the whole population).
Embeddings are cached in DERIVED/label_embeddings/ (encoding is the slow part on a CPU).
Outputs: results/label_classifier/cv_metrics.csv and a figure; DERIVED/label_classifier_oof.parquet (private).
"""
import json
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from paths import DATA, DERIVED, IMDB, RESULTS

torch.set_num_threads(8)  # faster than 12 on this hybrid CPU (measured)
OUT = RESULTS / "label_classifier"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = DERIVED / "label_embeddings"
CACHE.mkdir(exist_ok=True)
L = DERIVED / "labelling"
TARGETS = {"any_stereotype": "any stereotyped portrayal", "gender_any": "gender stereotype",
           "group_stereotyped": "group stereotype", "prejudice_theme": "story about prejudice"}
QWEN_TASK = ("Instruct: Decide whether this film or TV plot summary portrays a gender, ethnic, national, religious or "
             "sexual-orientation stereotype\nQuery:")

items = pd.DataFrame([json.loads(line) for line in open(L / "items.jsonl")])
labels = [json.loads(line) for line in open(L / "labels.jsonl")]
y = pd.DataFrame({
    "id": [r["id"] for r in labels],
    "gender_any": [r["gender"]["portrayal"] != "none" for r in labels],
    "group_stereotyped": [any(g["portrayal"] == "stereotyped" for g in r["groups"]) for r in labels],
    "prejudice_theme": [r["prejudice_theme"]["present"] for r in labels],
})
y["any_stereotype"] = y.gender_any | y.group_stereotyped
key = pd.read_parquet(L / "key.parquet")
d = items.merge(y, on="id").merge(key, on="id").sort_values("id", ignore_index=True)
d["random"] = d.stratum.str.startswith("random")

# series grouping: an episode belongs to its series, a TMDb-only episode to its TMDb series
con = duckdb.connect()
parents = con.execute(f"""SELECT tconst AS title, parentTconst FROM read_csv('{IMDB / "title.episode.tsv.gz"}',
    delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','parentTconst':'VARCHAR',
    'seasonNumber':'VARCHAR','episodeNumber':'VARCHAR'}})""").df()
d = d.merge(parents, on="title", how="left")
d["work"] = d.parentTconst.fillna(d.title.str.extract(r"^(tmdb:\d+)", expand=False)).fillna(d.title)
print(f"items {len(d):,} ({d.random.sum():,} random) in {d.work.nunique():,} films/series; positives: "
      + ", ".join(f"{t} {int(d[t].sum())} ({int(d[d.random][t].sum())} random)" for t in TARGETS), flush=True)


def encode(name):
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    started = time.time()
    if name == "mpnet":
        m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        X = m.encode(d.text.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    elif name == "qwen3":
        m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                processor_kwargs={"padding_side": "left"})
        X = m.encode(d.text.tolist(), prompt=QWEN_TASK, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    elif name == "gemma2":
        m = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
        X = m.encode(d.text.tolist(), prompt_name="Classification", batch_size=16, normalize_embeddings=True,
                     show_progress_bar=False)
    np.save(path, X)
    print(f"encoded with {name} in {(time.time() - started) / 60:.1f} min", flush=True)
    return X


glove = KeyedVectors.load_word2vec_format(str(next((DATA / "embeddings").rglob("glove-wiki-gigaword-300.gz"))),
                                          binary=False)
normed = glove.get_normed_vectors()
glove_X = np.zeros((len(d), glove.vector_size), dtype=np.float32)
for i, text in enumerate(d.text):
    ws = [glove.key_to_index[w] for w in re.findall(r"[a-z]+", text.lower()) if w in glove.key_to_index]
    if ws:
        glove_X[i] = normed[ws].mean(axis=0)
measures = pd.DataFrame({
    "looks": d.women_appearance.fillna(0) > 0, "victim": d.women_victim.fillna(0) > 0,
    "coded": d.gender_congruence.fillna(0), "coded_missing": d.gender_congruence.isna(),
    "group_pejorative": d.group_offensive.fillna(False), "group_only": d.group_mention_only.fillna(False),
    "slur": d.slur_mask.fillna(0) != 0, "viewer_tag": d.y.fillna(0) > 0, "theme_classifier": d.stereotype_prob.fillna(0),
    "log_words": np.log1p(d.words)}).astype(float).to_numpy()

features = {"glove": glove_X, "measures": measures}
for name in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["mpnet", "gemma2", "qwen3"]):
    features[name] = encode(name)


def cv_scores(X, target, repeat, tfidf=False):
    """Out-of-fold probabilities from 5 series-grouped folds; C chosen inside each training fold."""
    p = np.zeros(len(d))
    yv = d[target].astype(int).to_numpy()
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=repeat).split(d, yv, groups=d.work):
        if tfidf:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english")
            Xtr, Xte = vec.fit_transform(d.text.iloc[tr]), vec.transform(d.text.iloc[te])
        else:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        best, best_auc = 1.0, -1
        for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:  # inner grouped split for the penalty
            inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=99).split(tr, yv[tr], groups=d.work.iloc[tr]))
            aucs = []
            for itr, ite in inner:
                m = LogisticRegression(C=C, max_iter=3000).fit(Xtr[itr], yv[tr][itr])
                aucs.append(roc_auc_score(yv[tr][ite], m.predict_proba(Xtr[ite])[:, 1]))
            if np.mean(aucs) > best_auc:
                best, best_auc = C, np.mean(aucs)
        p[te] = LogisticRegression(C=best, max_iter=3000).fit(Xtr, yv[tr]).predict_proba(Xte)[:, 1]
    return p


rows, oof = [], {"id": d.id}
order = ["tf-idf", "glove", "mpnet", "qwen3", "gemma2", "measures"]
for target in TARGETS:
    preds = {}
    for name in order:
        if name != "tf-idf" and name not in features:
            continue
        preds[name] = [cv_scores(features.get(name), target, r, tfidf=(name == "tf-idf")) for r in range(3)]
    emb = [n for n in ["mpnet", "qwen3", "gemma2"] if n in preds]
    if emb:
        top = max(emb, key=lambda n: np.mean([roc_auc_score(d[target], p) for p in preds[n]]))
        combo = np.hstack([features[top], measures])
        preds[f"{top} + measures"] = [cv_scores(combo, target, r) for r in range(3)]
    for name, ps in preds.items():
        for scope, mask in [("all items", np.ones(len(d), bool)), ("random items", d.random.to_numpy())]:
            yv = d[target][mask]
            aucs = [roc_auc_score(yv, p[mask]) for p in ps]
            aps = [average_precision_score(yv, p[mask]) for p in ps]
            rows.append((target, name, scope, int(mask.sum()), yv.mean(), np.mean(aucs), np.std(aucs), np.mean(aps),
                         np.std(aps)))
        oof[f"{target}|{name}"] = np.mean(ps, axis=0)
    print(f"{target} done", flush=True)

m = pd.DataFrame(rows, columns=["target", "model", "scope", "items", "base_rate", "auc", "auc_sd", "average_precision",
                                "ap_sd"])
m.to_csv(OUT / "cv_metrics.csv", index=False)
pd.DataFrame(oof).to_parquet(DERIVED / "label_classifier_oof.parquet", index=False)
pd.set_option("display.width", 220)
print(m.pivot_table(index="model", columns=["target", "scope"], values="auc").round(3).to_string())
print(m[m.scope == "random items"].pivot_table(index="model", columns="target", values="average_precision").round(3)
      .to_string())
