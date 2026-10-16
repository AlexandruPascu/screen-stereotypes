"""Study 11, step 2: a classifier trained on the 2,685 social-group labels, used on study 9's fresh sample for trends.

The word lists of study 11 catch only about a third of the members of the six groups that a careful reader finds, so
their decade trends follow explicit naming. This classifier reads the whole summary instead.
Targets (the labelling model's reading under labelling/codebook_social.md): one of the six groups appears, one of them
is portrayed through a stereotype, a story about discrimination against them, and each group's appearance.
Features: EmbeddingGemma-2 (text encoder, "Classification" prompt; the first pass's cached vectors are reused for the
summaries both passes labelled) and the study's own measures (each group's mentions, stereotype and pejorative words
next to a mention, insulting labels, viewer theme tags, length), with a logistic regression on top, compared with
each part alone. Evaluation: 5-fold cross-validation with every series kept whole on one side (3 repeats for the
three main targets, 1 for the groups); AUC on all items and on the 1,400 random ones.
Then, per target, a model trained on all labels and calibrated on the random items' out-of-fold scores (Platt) scores
study 9's sample of 37,049 summaries, weighted by cell size; the 33 sample summaries that were labelled here are left
out. A check compares the sample's era estimates with the labels' own.
Outputs: results/label_classifier_social/; DERIVED/label_classifier_social_sample_scores.parquet (private).
"""
import json
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from paths import DERIVED, IMDB, RESULTS

torch.set_num_threads(8)
OUT = RESULTS / "label_classifier_social"
OUT.mkdir(parents=True, exist_ok=True)
S = DERIVED / "labelling_social"
CACHE = DERIVED / "label_embeddings"
GROUPS = json.load(open(DERIVED / "title_groups_bits_social.json"))
MAIN = {"any_group": "one of the six groups appears", "any_stereotyped": "portrayed through a stereotype",
        "discrimination": "a story about discrimination against them"}
TARGETS = list(MAIN) + [f"m_{g}" for g in GROUPS]
THEMES = (r"immigra|refugee|asylum seeker|deportation|disabilit|wheelchair|blindness|deafness|paraplegi|amputee|autism|"
          r"down syndrome|mental illness|schizophreni|bipolar|psychiatric|mental hospital|mental patient|insanity|"
          r"old age|elderly|retirement home|nursing home|ageism|senior citizen|poverty|working class|slum|homeless")
con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")


def with_measures(t):
    """The study's own measures for each title, from the current scans and viewer keywords."""
    full = pd.read_parquet(DERIVED / "title_groups_social.parquet")
    off = pd.read_parquet(DERIVED / "title_groups_social_offensive.parquet", columns=["title", "stereotype_mask"])
    con.register("wanted", t[["title"]])
    kw = con.execute(f"""SELECT title, string_agg(lower(keywords), '|') AS kw FROM '{DERIVED / "texts.parquet"}'
        WHERE keywords <> '' AND title IN (SELECT title FROM wanted) GROUP BY title""").df()
    t = (t.drop(columns=[c for c in ["mention_mask", "stereotype_mask", "slur_mask"] if c in t.columns])
         .merge(full, on="title", how="left")
         .merge(off.rename(columns={"stereotype_mask": "pejorative_mask"}), on="title", how="left")
         .merge(kw, on="title", how="left"))
    for c in ["mention_mask", "stereotype_mask", "slur_mask", "pejorative_mask"]:
        t[c] = t[c].fillna(0).astype("int64")
    cols = {f"mention_{g}": (t.mention_mask & (1 << i)) != 0 for i, g in enumerate(GROUPS)}
    cols.update({"paired": (t.mention_mask & t.stereotype_mask) != 0,
                 "paired_pejorative": (t.mention_mask & t.pejorative_mask) != 0, "label": t.slur_mask != 0,
                 "theme_tag": t.kw.fillna("").str.contains(THEMES, regex=True), "log_words": np.log1p(t.words)})
    return t, pd.DataFrame(cols).astype(float).to_numpy()


# --- the labelled summaries ---------------------------------------------------------------------------------------
items = pd.DataFrame([json.loads(line) for line in open(S / "items.jsonl")])
rows = []
for r in map(json.loads, open(S / "labels.jsonl")):
    entries = {e["group"]: e for e in r["social_groups"]}
    rows.append({"id": r["id"], "any_group": bool(entries), "discrimination": r["discrimination_theme"]["present"],
                 "any_stereotyped": any(e["portrayal"] == "stereotyped" for e in entries.values()),
                 **{f"m_{g}": g in entries for g in GROUPS}})
lab = (items.merge(pd.DataFrame(rows), on="id").merge(pd.read_parquet(S / "key.parquet"), on="id")
       .sort_values("id", ignore_index=True))
lab["random"] = lab.stratum.str.startswith("random")
parents = con.execute(f"""SELECT tconst AS title, parentTconst FROM read_csv('{IMDB / "title.episode.tsv.gz"}',
    delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','parentTconst':'VARCHAR',
    'seasonNumber':'VARCHAR','episodeNumber':'VARCHAR'}})""").df()
lab = lab.merge(parents, on="title", how="left")
lab["work"] = lab.parentTconst.fillna(lab.title.str.extract(r"^(tmdb:\d+)", expand=False)).fillna(lab.title)
lab, M_lab = with_measures(lab)

cache = CACHE / "social_gemma2.npy"
if cache.exists() and np.load(cache).shape[0] == len(lab):
    E_lab = np.load(cache)
else:  # first-pass vectors for the shared summaries (same text), new ones for the rest
    first = np.load(CACHE / "gemma2.npy")
    first_ids = sorted(json.loads(line)["id"] for line in open(DERIVED / "labelling/items.jsonl"))
    row = {i: n for n, i in enumerate(first_ids)}
    E_lab = np.zeros((len(lab), first.shape[1]), dtype=first.dtype)
    shared = lab.first_id.notna().to_numpy()
    E_lab[shared] = first[[row[i] for i in lab.first_id[shared]]]
    from sentence_transformers import SentenceTransformer
    started = time.time()
    model = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
    E_lab[~shared] = model.encode(lab.text[~shared].tolist(), prompt_name="Classification", batch_size=16,
                                  normalize_embeddings=True, show_progress_bar=False)
    np.save(cache, E_lab)
    print(f"encoded {(~shared).sum():,} new summaries in {(time.time() - started) / 60:.1f} min", flush=True)
FEATURES = {"measures": M_lab, "gemma2": E_lab, "gemma2 + measures": np.hstack([E_lab, M_lab])}
print(f"items {len(lab):,} ({lab.random.sum():,} random) in {lab.work.nunique():,} films/series; positives: "
      + ", ".join(f"{t} {int(lab[t].sum())} ({int(lab[lab.random][t].sum())} random)" for t in TARGETS), flush=True)

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]


def pick_c(X, yv, groups, seed):
    """The penalty with the best mean AUC over 3 series-grouped folds of the training data."""
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


def oof_scores(X, target, repeat):
    """Out-of-fold decision scores from 5 series-grouped folds, the penalty chosen inside each training fold."""
    yv = lab[target].astype(int).to_numpy()
    p = np.zeros(len(lab))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=repeat).split(X, yv, groups=lab.work):
        C = pick_c(X[tr], yv[tr], lab.work.iloc[tr], seed=99)
        sc = StandardScaler().fit(X[tr])
        p[te] = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), yv[tr]).decision_function(
            sc.transform(X[te]))
    return p


metrics, oof_best = [], {}
for target in TARGETS:
    for name, X in FEATURES.items():
        if target not in MAIN and name != "gemma2 + measures":
            continue
        ps = [oof_scores(X, target, r) for r in range(3 if target in MAIN else 1)]
        for scope, mask in [("all items", np.ones(len(lab), bool)), ("random items", lab.random.to_numpy())]:
            yv = lab[target][mask]
            aucs = [roc_auc_score(yv, p[mask]) for p in ps] if yv.nunique() == 2 else [np.nan]
            metrics.append((target, name, scope, int(mask.sum()), int(yv.sum()), np.mean(aucs), np.std(aucs)))
        if name == "gemma2 + measures":
            oof_best[target] = ps[0]
    print(f"{target} done", flush=True)
metrics = pd.DataFrame(metrics, columns=["target", "model", "scope", "items", "positives", "auc", "auc_sd"])
metrics.to_csv(OUT / "cv_metrics.csv", index=False)
pd.set_option("display.width", 220)
print(metrics.pivot_table(index=["target", "model"], columns="scope", values="auc").round(3).to_string(), flush=True)

# --- the fresh sample of study 9 ------------------------------------------------------------------------------------
s = pd.read_parquet(DERIVED / "label_classifier_sample_scores.parquet",
                    columns=["title", "words", "kind", "year", "medium", "decade", "cell_size", "weight"])
E = np.load(CACHE / "sample_gemma2.npy")
assert len(E) == len(s), "the cached sample vectors no longer match the sample"
keep = ~s.title.isin(lab.title).to_numpy()  # summaries labelled here are not scored
s, E = s[keep].reset_index(drop=True), E[keep]
s, M = with_measures(s)
X = np.hstack([E, M])
X_lab = FEATURES["gemma2 + measures"]
rnd = lab.random.to_numpy()
for target in TARGETS:
    yv = lab[target].astype(int).to_numpy()
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof_best[target][rnd][:, None], yv[rnd])
    C = pick_c(X_lab, yv, lab.work, seed=0)
    sc = StandardScaler().fit(X_lab)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X_lab), yv)
    s[target] = platt.predict_proba(m.decision_function(sc.transform(X))[:, None])[:, 1]
s["weight"] = s.cell_size / s.groupby(["decade", "medium"]).title.transform("size")
s.drop(columns=["kw"]).to_parquet(DERIVED / "label_classifier_social_sample_scores.parquet", index=False)

by = s.groupby(["decade", "medium"])[TARGETS].mean().reset_index()
by["summaries"] = s.groupby(["decade", "medium"]).size().values
by.to_csv(OUT / "sample_prevalence_by_decade.csv", index=False)
print(by.pivot(index="decade", columns="medium", values=["any_group", "any_stereotyped"]).round(3).to_string(), flush=True)

# check: the sample's era estimates against the labels' own (random items, 95% intervals)
s["era"] = pd.cut(s.year, [0, 1969, 1999, 2100], labels=["before 1970", "1970-99", "2000+"]).astype("string")
prev = pd.read_csv(RESULTS / "labels_social/prevalence.csv")
prev = prev[(prev.labels_used == "all labels") & (prev.medium != "all")]
check = []
for (medium, era), x in s.groupby(["medium", "era"]):
    for target in ["any_group", "any_stereotyped", "discrimination"]:
        p = prev[(prev.medium == medium) & (prev.era == era) & (prev.label == target)].iloc[0]
        est = np.average(x[target], weights=x.weight)
        check.append((medium, era, target, est, p.share, p.lo, p.hi, p.lo <= est <= p.hi))
check = pd.DataFrame(check, columns=["medium", "era", "target", "sample_estimate", "labels", "labels_lo", "labels_hi",
                                     "inside_interval"])
check.to_csv(OUT / "calibration_check.csv", index=False)
print(check.round(3).to_string(index=False))
print(f"{check.inside_interval.sum()} of {len(check)} sample estimates inside the labels' 95% intervals", flush=True)

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
bg[bg.summaries >= 300].sort_values("any_stereotyped").to_csv(OUT / "sample_prevalence_by_genre.csv", index=False)
print(bg[bg.summaries >= 300].sort_values("any_stereotyped")[["genre", "any_group", "any_stereotyped", "summaries"]]
      .round(3).to_string(index=False))
