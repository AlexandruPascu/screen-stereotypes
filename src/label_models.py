"""The label-trained classifiers of studies 9 and 11, refitted exactly as score_sample.py and social_classifier.py
fit them, for scoring titles outside their samples (score_films.py, score_series.py).

fit() refits both and checks them against the sample scores those two scripts saved; apply() turns features into
calibrated probabilities. Features: EmbeddingGemma-2 vectors plus measures9() or measures11().
"""
import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from paths import DERIVED, IMDB

con = duckdb.connect()
con.execute("SET memory_limit = '8GB'")


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{{cols}}})"


BASICS = tsv("title.basics.tsv.gz", ["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult", "startYear",
                                      "endYear", "runtimeMinutes", "genres"])
parents = con.execute(f"""SELECT tconst AS title, parentTconst FROM {tsv('title.episode.tsv.gz', ['tconst', 'parentTconst',
    'seasonNumber', 'episodeNumber'])}""").df()


def with_work(d):
    d = d.merge(parents, on="title", how="left")
    d["work"] = d.parentTconst.fillna(d.title.str.extract(r"^(tmdb:\d+)", expand=False)).fillna(d.title)
    return d


# --- study 9's classifier, as score_sample.py fits it --------------------------------------------------------------
L = DERIVED / "labelling"
T9 = ["any_stereotype", "gender_any", "group_stereotyped", "prejudice_theme"]


def measures9(t):
    return pd.DataFrame({
        "looks": t.women_appearance.fillna(0) > 0, "victim": t.women_victim.fillna(0) > 0,
        "coded": t.gender_congruence.fillna(0), "coded_missing": t.gender_congruence.isna(),
        "group_pejorative": t.group_offensive.fillna(False), "group_only": t.group_mention_only.fillna(False),
        "slur": t.slur_mask.fillna(0) != 0, "viewer_tag": t.y.fillna(0) > 0, "theme_classifier": t.stereotype_prob.fillna(0),
        "log_words": np.log1p(t.words)}).astype(float).to_numpy()


def attach9(t):
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

# --- study 11's classifier, as social_classifier.py fits it ----------------------------------------------------------
S = DERIVED / "labelling_social"
GROUPS = json.load(open(DERIVED / "title_groups_bits_social.json"))
T11 = ["any_group", "any_stereotyped", "discrimination"] + [f"m_{g}" for g in GROUPS]
THEMES = (r"immigra|refugee|asylum seeker|deportation|disabilit|wheelchair|blindness|deafness|paraplegi|amputee|autism|"
          r"down syndrome|mental illness|schizophreni|bipolar|psychiatric|mental hospital|mental patient|insanity|"
          r"old age|elderly|retirement home|nursing home|ageism|senior citizen|poverty|working class|slum|homeless")
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]


def measures11(t):
    full = pd.read_parquet(DERIVED / "title_groups_social.parquet")
    off = pd.read_parquet(DERIVED / "title_groups_social_offensive.parquet", columns=["title", "stereotype_mask"])
    con.register("wanted", t[["title"]])
    kw = con.execute(f"""SELECT title, string_agg(lower(keywords), '|') AS kw FROM '{DERIVED / "texts.parquet"}'
        WHERE keywords <> '' AND title IN (SELECT title FROM wanted) GROUP BY title""").df()
    t = (t[["title", "words"]].merge(full, on="title", how="left")
         .merge(off.rename(columns={"stereotype_mask": "pejorative_mask"}), on="title", how="left")
         .merge(kw, on="title", how="left"))
    for c in ["mention_mask", "stereotype_mask", "slur_mask", "pejorative_mask"]:
        t[c] = t[c].fillna(0).astype("int64")
    cols = {f"mention_{g}": (t.mention_mask & (1 << i)) != 0 for i, g in enumerate(GROUPS)}
    cols.update({"paired": (t.mention_mask & t.stereotype_mask) != 0,
                 "paired_pejorative": (t.mention_mask & t.pejorative_mask) != 0, "label": t.slur_mask != 0,
                 "theme_tag": t.kw.fillna("").str.contains(THEMES, regex=True), "log_words": np.log1p(t.words)})
    return pd.DataFrame(cols).astype(float).to_numpy()


def pick_c(X, yv, groups, seed):
    best, best_auc = C_GRID[0], -1
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, yv, groups=groups))
    for C in C_GRID:
        aucs = []
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), yv[tr])
            aucs.append(roc_auc_score(yv[te], m.decision_function(sc.transform(X[te]))))
        if np.mean(aucs) > best_auc:
            best, best_auc = C, np.mean(aucs)
    return best

def apply(models, X):
    return {t: platt.predict_proba(m.decision_function(sc.transform(X))[:, None])[:, 1]
            for t, (sc, m, platt) in models.items()}


def fit():
    """Both classifiers, refitted and checked; returns (models9, models11)."""
    items = pd.DataFrame([json.loads(line) for line in open(L / "items.jsonl")])
    labels = [json.loads(line) for line in open(L / "labels.jsonl")]
    y9 = pd.DataFrame({"id": [r["id"] for r in labels],
                       "gender_any": [r["gender"]["portrayal"] != "none" for r in labels],
                       "group_stereotyped": [any(g["portrayal"] == "stereotyped" for g in r["groups"]) for r in labels],
                       "prejudice_theme": [r["prejudice_theme"]["present"] for r in labels]})
    y9["any_stereotype"] = y9.gender_any | y9.group_stereotyped
    lab9 = with_work(items.merge(y9, on="id").merge(pd.read_parquet(L / "key.parquet"), on="id")
                     .sort_values("id", ignore_index=True))
    lab9["random"] = lab9.stratum.str.startswith("random")
    X9 = np.hstack([np.load(DERIVED / "label_embeddings/gemma2.npy"), measures9(lab9)])
    models9 = {}
    for target in T9:
        yv = lab9[target].astype(int).to_numpy()
        best, best_auc = 1.0, -1
        for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
            aucs = []
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X9, yv, groups=lab9.work):
                sc = StandardScaler().fit(X9[tr])
                m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X9[tr]), yv[tr])
                aucs.append(roc_auc_score(yv[te], m.predict_proba(sc.transform(X9[te]))[:, 1]))
            if np.mean(aucs) > best_auc:
                best, best_auc = C, np.mean(aucs)
        oof = np.zeros(len(lab9))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(X9, yv, groups=lab9.work):
            sc = StandardScaler().fit(X9[tr])
            m = LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X9[tr]), yv[tr])
            oof[te] = m.decision_function(sc.transform(X9[te]))
        rnd = lab9.random.to_numpy()
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof[rnd][:, None], yv[rnd])
        sc = StandardScaler().fit(X9)
        models9[target] = (sc, LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X9), yv), platt)
    print("study 9 models fitted", flush=True)

    rows = []
    for r in map(json.loads, open(S / "labels.jsonl")):
        entries = {e["group"]: e for e in r["social_groups"]}
        rows.append({"id": r["id"], "any_group": bool(entries), "discrimination": r["discrimination_theme"]["present"],
                     "any_stereotyped": any(e["portrayal"] == "stereotyped" for e in entries.values()),
                     **{f"m_{g}": g in entries for g in GROUPS}})
    lab11 = with_work(pd.DataFrame([json.loads(line) for line in open(S / "items.jsonl")]).merge(pd.DataFrame(rows), on="id")
                      .merge(pd.read_parquet(S / "key.parquet"), on="id").sort_values("id", ignore_index=True))
    X11 = np.hstack([np.load(DERIVED / "label_embeddings/social_gemma2.npy"), measures11(lab11)])
    rnd11 = lab11.stratum.str.startswith("random").to_numpy()
    models11 = {}
    for target in T11:
        yv = lab11[target].astype(int).to_numpy()
        oof = np.zeros(len(lab11))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X11, yv, groups=lab11.work):
            C = pick_c(X11[tr], yv[tr], lab11.work.iloc[tr], seed=99)
            sc = StandardScaler().fit(X11[tr])
            oof[te] = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X11[tr]), yv[tr]).decision_function(
                sc.transform(X11[te]))
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof[rnd11][:, None], yv[rnd11])
        C = pick_c(X11, yv, lab11.work, seed=0)
        sc = StandardScaler().fit(X11)
        models11[target] = (sc, LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X11), yv), platt)
    print("study 11 models fitted", flush=True)

    # check: the refitted models reproduce the sample scores the two scripts saved
    s9 = pd.read_parquet(DERIVED / "label_classifier_sample_scores.parquet")
    E9 = np.load(DERIVED / "label_embeddings/sample_gemma2.npy")
    p9 = apply(models9, np.hstack([E9, measures9(s9)]))
    s11 = pd.read_parquet(DERIVED / "label_classifier_social_sample_scores.parquet")
    row = {t: i for i, t in enumerate(s9.title)}
    p11 = apply(models11, np.hstack([E9[[row[t] for t in s11.title]], measures11(s11)]))
    for name, saved, new in [("study 9", s9, p9), ("study 11", s11, p11)]:
        gaps = {t: float(np.abs(saved[t].to_numpy() - new[t]).max()) for t in new}
        print(f"{name} refit vs saved sample scores, largest gap per target: "
              + ", ".join(f"{t} {g:.4f}" for t, g in gaps.items()), flush=True)
        assert max(gaps.values()) < 0.02, f"{name}: refitted models do not reproduce the saved sample scores"
    # films are scored with today's measure files; study 7's rerun moved its scores slightly, so measure that drift too
    now9 = apply(models9, np.hstack([E9, measures9(attach9(s9[["title", "words"]]))]))
    print("study 9 sample with today's measure files, mean change per target: "
          + ", ".join(f"{t} {float(np.mean(now9[t] - s9[t])):+.4f}" for t in now9), flush=True)
    return models9, models11
