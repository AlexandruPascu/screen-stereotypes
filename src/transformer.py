"""Step 2b: BERT-family models on the classifier's matched sample and series-grouped folds (classifier.py).

A. Frozen sentence embeddings (sentence-transformers/all-mpnet-base-v2, Apache-2.0) + logistic regression.
B. A fine-tuned compact BERT (microsoft/MiniLM-L12-H384-uncased, MIT): the text goes through the whole network, which
   is trained end to end with a classification head. CPU-only (no data leaves the machine), so it runs once, on one
   series-grouped split (fold 0 held out; FINETUNE_FOLDS=0,1,... for more), and is compared on that same fold.
Same metrics as classifier.py (AUC, average precision, Brier; by medium and by text-length tertile), so the models can
be compared directly. Outputs: results/classifier/transformer_cv_metrics.csv; DERIVED/transformer_oof.parquet.
Usage: transformer.py [frozen|finetune|both]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from paths import DERIVED, RESULTS

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"
torch.manual_seed(0)
data = pd.read_parquet(DERIVED / "classifier_sample.parquet")
data["len_tertile"] = pd.qcut(data.words.rank(method="first"), 3, labels=["short", "medium", "long"])
print(f"sample {len(data):,}, positives {data.y.mean():.1%}, folds {sorted(data.fold.unique())}", flush=True)
oof = pd.DataFrame({"title": data.title})


def frozen():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    model.max_seq_length = 256
    started = time.time()
    X = model.encode(data.text.tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True,
                     normalize_embeddings=True)
    print(f"encoded {len(X):,} summaries in {(time.time() - started) / 60:.1f} min", flush=True)
    p = np.zeros(len(data))
    for fold in sorted(data.fold.unique()):
        tr, te = data.fold != fold, data.fold == fold
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], data.y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def finetune(epochs=2, max_len=160, batch=32, lr=5e-5):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    name = "microsoft/MiniLM-L12-H384-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    p = np.full(len(data), np.nan)
    for fold in [int(f) for f in os.environ.get("FINETUNE_FOLDS", "0").split(",")]:
        started = time.time()
        torch.manual_seed(fold)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
        tr = data[data.fold != fold].sample(frac=1, random_state=fold)
        te = data[data.fold == fold]
        steps = epochs * int(np.ceil(len(tr) / batch))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        model.train()
        for epoch in range(epochs):
            order = np.random.default_rng(fold * 10 + epoch).permutation(len(tr))
            for i in range(0, len(tr), batch):
                idx = order[i:i + batch]
                enc = tok(tr.text.iloc[idx].tolist(), truncation=True, max_length=max_len, padding=True,
                          return_tensors="pt")
                out = model(**enc, labels=torch.tensor(tr.y.iloc[idx].to_numpy()))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(te), 128):
                enc = tok(te.text.iloc[i:i + 128].tolist(), truncation=True, max_length=max_len, padding=True,
                          return_tensors="pt")
                probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy())
        p[(data.fold == fold).to_numpy()] = np.concatenate(probs)
        print(f"fold {fold}: {(time.time() - started) / 60:.1f} min, AUC {roc_auc_score(te.y, np.concatenate(probs)):.3f}",
              flush=True)
    return p


if MODE in ("frozen", "both"):
    oof["mpnet frozen + logistic"] = frozen()
if MODE in ("finetune", "both"):
    oof["MiniLM fine-tuned"] = finetune()
oof.to_parquet(DERIVED / f"transformer_oof_{MODE}.parquet", index=False)

rows = []
for name in [c for c in oof.columns if c != "title"]:
    p = oof[name].to_numpy()
    scored = ~np.isnan(p)
    for scope, mask in [("all", np.ones(len(data), bool)), ("film", data.medium == "film"), ("tv", data.medium == "tv")] + \
            [(f"length {t}", data.len_tertile == t) for t in ["short", "medium", "long"]] + \
            [(f"fold {f}", data.fold == f) for f in sorted(data.fold.unique())]:
        mask = np.asarray(mask) & scored
        if mask.sum() == 0:
            continue
        y = data.y[mask]
        rows.append((name, scope, int(mask.sum()), y.mean(), roc_auc_score(y, p[mask]), average_precision_score(y, p[mask]),
                     brier_score_loss(y, p[mask])))
m = pd.DataFrame(rows, columns=["model", "scope", "titles", "positives", "auc", "average_precision", "brier"])
path = RESULTS / "classifier/transformer_cv_metrics.csv"
if path.exists():
    m = pd.concat([pd.read_csv(path).query("model not in @m.model.unique()"), m])
m.to_csv(path, index=False)
pd.set_option("display.width", 200)
print(m.round(3).to_string(index=False))
