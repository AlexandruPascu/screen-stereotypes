"""Step 2: a stereotype classifier built properly (the 2022 BERT notebook, redone).

Label: a title is positive when viewers tagged it with an on-topic stereotype or prejudice keyword (IMDb keywords
from the 2022 scrape for episodes, TMDb keywords for films; the 2022 assessment's keyword family plus antisemitism,
islamophobia, xenophobia, bigotry, slurs; age, disability and class discrimination left out).
Negatives: titles that viewers tagged with at least MIN_KEYWORDS keywords but none of those, matched to each positive on
medium, decade, number of keywords (tagging effort) and text length — the 2022 model learned "long film-style text =
stereotype" because its negatives were short episode plots and its positives long film synopses.
Evaluation: 5-fold cross-validation with every series kept whole on one side (StratifiedGroupKFold), against a
baseline that only knows the text length; AUC, average precision and Brier score, overall and within length tertiles.
Models: TF-IDF words and pairs + logistic regression; average GloVe vector + logistic regression.
Then the best model is refit on all labelled data and every summary is scored; probabilities are corrected from the
matched sample's share of positives to the share among all tagged titles.
Outputs: results/classifier/*.csv (figures drawn by figures.py); DERIVED/title_classifier_scores.parquet (private).
"""
import re

import duckdb
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from paths import DATA, DERIVED, IMDB, RESULTS
from publish import without_slurs

OUT = RESULTS / "classifier"
OUT.mkdir(parents=True, exist_ok=True)
MIN_KEYWORDS = 5
MIN_WORKS = 20  # a word or pair must appear in at least this many different films/series (drops show-specific names)
NEG_PER_POS = 5
ON_TOPIC = re.compile(r"stereotyp|racis|racial|sexis|homophob|discriminat|heterosexis|antisemit|anti-semit|islamophob|"
                      r"xenophob|bigot|slur|transphob|misogyn")
OFF_TOPIC = re.compile(r"\bage\b|age-|disab|class|weight|fat|caste")
rng = np.random.default_rng(0)
con = duckdb.connect()
con.execute("SET memory_limit = '10GB'")  # the longest-summary pick holds each text twice while grouping

texts = con.execute(f"""SELECT title, arg_max(text, (words, text)) AS text, max(words) AS words,
        string_agg(lower(keywords), '|') FILTER (WHERE keywords <> '' AND source IN ('imdb22', 'tmdb')) AS kw,
        arg_max(kind, (words, text)) AS kind
    FROM '{DERIVED / "texts.parquet"}' GROUP BY title""").df()
extra = pd.read_parquet(DATA / "gcp-2022/bigquery/extra_titles.parquet").rename(
    columns={"string_field_0": "title", "string_field_1": "kw_extra"})
texts = texts.merge(extra[["title", "kw_extra"]], on="title", how="left")
texts = texts.sort_values("title", ignore_index=True)  # DuckDB's GROUP BY order varies; sorting makes the sample repeatable
texts["kw"] = texts.kw.fillna("") + "|" + texts.kw_extra.fillna("")
meta = con.execute(f"""
    WITH b AS (SELECT tconst, titleType, TRY_CAST(startYear AS INT) AS year FROM read_csv('{IMDB / "title.basics.tsv.gz"}',
               delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR',
               'titleType':'VARCHAR','primaryTitle':'VARCHAR','originalTitle':'VARCHAR','isAdult':'VARCHAR',
               'startYear':'VARCHAR','endYear':'VARCHAR','runtimeMinutes':'VARCHAR','genres':'VARCHAR'}})),
    e AS (SELECT tconst, parentTconst FROM read_csv('{IMDB / "title.episode.tsv.gz"}', delim='\\t', header=true, quote='',
          escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','parentTconst':'VARCHAR','seasonNumber':'VARCHAR',
          'episodeNumber':'VARCHAR'}}))
    SELECT b.tconst AS title, b.year, COALESCE(e.parentTconst, b.tconst) AS work FROM b LEFT JOIN e USING (tconst)""").df()
texts = texts.merge(meta, on="title", how="left")
texts["work"] = texts.work.fillna(texts.title.str.extract(r"^(tmdb:\d+)", expand=False)).fillna(texts.title)
texts["medium"] = np.where(texts.kind == "film", "film", "tv")
texts["decade"] = (texts.year // 10 * 10).fillna(-1)


def keywords(s):
    return [k.strip().replace("-", " ") for k in re.split(r"[|,]", s) if k.strip() and k.strip() != "NULL"]


kws = texts.kw.map(keywords)
texts["n_kw"] = kws.map(len)
texts["y"] = kws.map(lambda ks: any(ON_TOPIC.search(k) and not OFF_TOPIC.search(k) for k in ks)).astype(int)
pool = texts[(texts.words >= 10) & ((texts.n_kw >= MIN_KEYWORDS) | (texts.y == 1))].copy()
pool["kw_bin"] = pd.qcut(pool.n_kw.rank(method="first"), 5, labels=False)
pool["len_bin"] = pd.qcut(pool.words.rank(method="first"), 5, labels=False)
pool["stratum"] = pool.medium + "|" + pool.decade.astype(int).astype(str) + "|" + pool.kw_bin.astype(str) + "|" + \
    pool.len_bin.astype(str)
pos = pool[pool.y == 1]
neg_pool = pool[pool.y == 0]
negs = []
for stratum, n in pos.stratum.value_counts().sort_index().items():
    cand = neg_pool[neg_pool.stratum == stratum]
    if len(cand):
        negs.append(cand.sample(min(len(cand), n * NEG_PER_POS), random_state=0))
data = pd.concat([pos] + negs).sample(frac=1, random_state=0).reset_index(drop=True)
prior_pool = pool.y.mean()
print(f"tagged titles {len(pool):,} (positives {pos.shape[0]:,}, {prior_pool:.2%}); matched sample {len(data):,} "
      f"(positives {data.y.mean():.1%}); by medium: {data.groupby('medium').y.agg(['size', 'sum']).to_dict()}", flush=True)

glove = KeyedVectors.load_word2vec_format(str(next((DATA / "embeddings").rglob("glove-wiki-gigaword-300.gz"))),
                                          binary=False)
glove.fill_norms()
NORMED = glove.get_normed_vectors()  # computed once: get_normed_vectors() builds a new 400k x 300 matrix on every call
TOKEN = re.compile(r"[a-z]+")


def glove_features(series):
    out = np.zeros((len(series), glove.vector_size), dtype=np.float32)
    for i, text in enumerate(series):
        ws = [glove.key_to_index[w] for w in TOKEN.findall(text.lower()) if w in glove.key_to_index]
        if ws:
            out[i] = NORMED[ws].mean(axis=0)
    return out


def tfidf(train_text, train_work):
    """TF-IDF over words and word pairs that occur across at least MIN_WORKS different films or series, so the model
    cannot learn which shows viewers tagged by their character names."""
    counts = CountVectorizer(ngram_range=(1, 2), min_df=5, binary=True, stop_words="english")
    X = counts.fit_transform(train_text)
    codes, works = pd.factorize(train_work)
    by_work = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(works), len(codes)))
    n_works = np.asarray(((by_work @ X) > 0).sum(axis=0)).ravel()
    vocab = counts.get_feature_names_out()[n_works >= MIN_WORKS]
    return TfidfVectorizer(ngram_range=(1, 2), vocabulary=vocab, sublinear_tf=True, stop_words="english")


models = {
    "length only": (lambda tr, te: (np.log1p(tr.words.to_numpy())[:, None], np.log1p(te.words.to_numpy())[:, None]), 1.0),
    "tf-idf + logistic": (None, 4.0),
    "glove mean + logistic": (None, 1.0),
}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
oof = {m: np.zeros(len(data)) for m in models}
data["fold"] = -1
for fold, (_, te_i) in enumerate(cv.split(data, data.y, groups=data.work)):
    data.loc[te_i, "fold"] = fold
# the same sample and folds for the transformer models (private: holds text)
data[["title", "text", "words", "y", "work", "medium", "decade", "fold"]].to_parquet(DERIVED / "classifier_sample.parquet",
                                                                                     index=False)
for fold, (tr_i, te_i) in enumerate(cv.split(data, data.y, groups=data.work)):
    tr, te = data.iloc[tr_i], data.iloc[te_i]
    for name, (feat, C) in models.items():
        if name == "length only":
            Xtr, Xte = feat(tr, te)
        elif name.startswith("tf-idf"):
            vec = tfidf(tr.text, tr.work)
            Xtr, Xte = vec.fit_transform(tr.text), vec.transform(te.text)
        else:
            Xtr, Xte = glove_features(tr.text), glove_features(te.text)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight=None)
        clf.fit(Xtr, tr.y)
        oof[name][te_i] = clf.predict_proba(Xte)[:, 1]
    print(f"fold {fold + 1} done", flush=True)

pd.DataFrame({"title": data.title, "fold": data.fold, **oof}).to_parquet(DERIVED / "classifier_oof.parquet", index=False)
rows = []
data["len_tertile"] = pd.qcut(data.words.rank(method="first"), 3, labels=["short", "medium", "long"])
for name, p in oof.items():
    for scope, mask in [("all", np.ones(len(data), bool)), ("film", data.medium == "film"), ("tv", data.medium == "tv")] + \
            [(f"length {t}", data.len_tertile == t) for t in ["short", "medium", "long"]] + \
            [(f"fold {f}", data.fold == f) for f in sorted(data.fold.unique())]:
        y = data.y[mask]
        if y.nunique() == 2:
            rows.append((name, scope, mask.sum(), y.mean(), roc_auc_score(y, p[mask]), average_precision_score(y, p[mask]),
                         brier_score_loss(y, p[mask])))
metrics = pd.DataFrame(rows, columns=["model", "scope", "titles", "positives", "auc", "average_precision", "brier"])
metrics.to_csv(OUT / "cv_metrics.csv", index=False)
pd.set_option("display.width", 200)
print(metrics.round(3).to_string(index=False), flush=True)

best = metrics[(metrics.scope == "all") & (metrics.model != "length only")].sort_values("auc").model.iloc[-1]
print(f"best model: {best}", flush=True)
# Calibration: isotonic regression on the out-of-fold scores. To check it honestly, each fold is calibrated with a map
# fitted on the other folds only; the final map (for scoring every summary) is fitted on all out-of-fold scores.
raw = oof[best]
calibrated = np.zeros(len(data))
for fold in sorted(data.fold.unique()):
    other, this = (data.fold != fold).to_numpy(), (data.fold == fold).to_numpy()
    calibrated[this] = IsotonicRegression(out_of_bounds="clip").fit(raw[other], data.y[other]).predict(raw[this])
calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, data.y)
rows = []
for label, p in [("raw", raw), ("calibrated", calibrated)]:
    bins = pd.qcut(p, 10, duplicates="drop")
    c = pd.DataFrame({"p": p, "y": data.y}).groupby(bins, observed=True).agg(predicted=("p", "mean"),
                                                                             observed=("y", "mean")).reset_index(drop=True)
    c["scores"] = label
    rows.append(c)
pd.concat(rows).to_csv(OUT / "calibration.csv", index=False)
print(f"Brier score, cross-validated: raw {brier_score_loss(data.y, raw):.4f}, calibrated "
      f"{brier_score_loss(data.y, calibrated):.4f}", flush=True)

# refit on everything, score every summary, correct the prior from the matched sample to all tagged titles
rho = data.y.mean()
shift = np.log(prior_pool / (1 - prior_pool)) - np.log(rho / (1 - rho))
# The words behind a stereotype tag, from the TF-IDF model (readable, whichever model scores best); slurs left out.
vec_all = tfidf(data.text, data.work)
words_model = LogisticRegression(C=models["tf-idf + logistic"][1], max_iter=2000).fit(vec_all.fit_transform(data.text), data.y)
coef = pd.Series(words_model.coef_[0], index=vec_all.get_feature_names_out()).rename("weight")
top = pd.concat([coef.nlargest(60), coef.nsmallest(40)]).rename_axis("term").reset_index()
without_slurs(top, "term").to_csv(OUT / "top_features.csv", index=False)
if best.startswith("tf-idf"):
    final, transform = words_model, vec_all.transform
else:
    final = LogisticRegression(C=models[best][1], max_iter=2000).fit(glove_features(data.text), data.y)
    transform = glove_features
scores = []
for i in range(0, len(texts), 200_000):
    chunk = texts.iloc[i:i + 200_000]
    p = calibrator.predict(final.predict_proba(transform(chunk.text))[:, 1]).clip(1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p)) + shift
    scores.append(pd.DataFrame({"title": chunk.title, "stereotype_prob": 1 / (1 + np.exp(-logit))}))
scores = pd.concat(scores).merge(texts[["title", "decade", "medium", "y", "n_kw"]], on="title")
scores.to_parquet(DERIVED / "title_classifier_scores.parquet", index=False)
by = scores[scores.decade.between(1950, 2020)].groupby(["decade", "medium"]).agg(titles=("stereotype_prob", "size"),
                                                                      mean_prob=("stereotype_prob", "mean")).reset_index()
by.to_csv(OUT / "score_by_decade.csv", index=False)
print(by.pivot(index="decade", columns="medium", values="mean_prob").round(4).to_string())
