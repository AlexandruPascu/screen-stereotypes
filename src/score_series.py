"""Score the first season of finished scripted TV series with the two label-trained classifiers, for study 12's
question of staying power: are series that open with stereotyped portrayals renewed less often?

Series: scripted series in TMDb (not miniseries, which are not meant to be renewed) that have finished (status Ended
or Canceled), first aired 1990–2022, with an IMDb id and at least MIN_VOTES IMDb votes. Episodes: up to 6 of the first
season (the lowest episode numbers) with a summary of 15–350 words (the longest such, ties broken as elsewhere);
series with fewer than 3 are left out. Scores: the classifiers of label_models.py on EmbeddingGemma-2 vectors, encoded
in resumable parts of 2,000; each series gets the mean over its first-season episodes.
Usage: score_series.py [count]   (count: only report how many series and episodes would be scored)
Output: DERIVED/series_classifier_scores.parquet (private; per series).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from label_models import apply, attach9, con, fit, measures9, measures11, tsv
from paths import DERIVED

torch.set_num_threads(8)
PARTS = DERIVED / "series_embeddings"
PARTS.mkdir(exist_ok=True)
PART_SIZE = 2000
MIN_VOTES = int(os.environ.get("MIN_VOTES", "500"))
PER_SERIES = 6

tm = pd.read_parquet(DERIVED / "tmdb_titles.parquet", columns=["kind", "imdb", "tv_type", "status", "first_air_date"])
s = tm[(tm.kind == "series") & (tm.tv_type == "Scripted") & tm.status.isin(["Ended", "Canceled"])].copy()
s["start"] = pd.to_datetime(s.first_air_date, errors="coerce").dt.year
s = s[s.start.between(1990, 2022)].drop_duplicates("imdb")
con.register("candidates", s[["imdb"]])
rated = con.execute(f"""SELECT tconst AS imdb FROM {tsv('title.ratings.tsv.gz', ['tconst', 'averageRating', 'numVotes'])}
    WHERE CAST(numVotes AS INT) >= {MIN_VOTES} AND tconst IN (SELECT imdb FROM candidates)""").df()
con.register("rated", rated)
eps = con.execute(f"""
    WITH e AS (SELECT tconst AS title, parentTconst AS series, TRY_CAST(episodeNumber AS INT) AS episode
               FROM {tsv('title.episode.tsv.gz', ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber'])}
               WHERE TRY_CAST(seasonNumber AS INT) = 1 AND parentTconst IN (SELECT imdb FROM rated)),
    t AS (SELECT title, arg_max(text, (words, text)) AS text, max(words) AS words FROM '{DERIVED / "texts.parquet"}'
          WHERE words BETWEEN 15 AND 350 AND title IN (SELECT title FROM e) GROUP BY title)
    SELECT e.series, e.episode, t.title, t.text, t.words FROM e JOIN t USING (title)""").df()
eps = eps.sort_values(["series", "episode", "title"]).groupby("series").head(PER_SERIES)
eps = eps[eps.groupby("series").title.transform("size") >= 3].reset_index(drop=True)
print(f"finished scripted series 1990–2022: {len(s):,}; with >= {MIN_VOTES} votes: {len(rated):,}; with 3+ first-season "
      f"summaries: {eps.series.nunique():,}; episodes to score: {len(eps):,}", flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "count":
    raise SystemExit

models9, models11 = fit()
manifest = PARTS / "manifest.parquet"
if manifest.exists():
    assert pd.read_parquet(manifest).title.tolist() == eps.title.tolist(), "the episode list changed; start a fresh run"
else:
    eps[["series", "title", "words"]].to_parquet(manifest, index=False)
parts = [(i, PARTS / f"part_{i // PART_SIZE:05d}.npy") for i in range(0, len(eps), PART_SIZE)]
todo = [(i, p) for i, p in parts if not p.exists()]
if todo:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
    started, done = time.time(), 0
    for i, path in todo:
        chunk = eps.text.iloc[i:i + PART_SIZE].tolist()
        E = model.encode(chunk, prompt_name="Classification", batch_size=16, normalize_embeddings=True,
                         show_progress_bar=False)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, E)
        os.replace(tmp, path)
        done += len(chunk)
        rate = done / (time.time() - started)
        left = sum(min(PART_SIZE, len(eps) - j) for j, p in todo if not p.exists())
        print(f"  part {i // PART_SIZE + 1}/{len(parts)} saved; {rate:.1f} episodes/s; about {left / rate / 3600:.1f} h left",
              flush=True)
E = np.vstack([np.load(p) for _, p in parts])
assert len(E) == len(eps)

scores = eps[["series", "title"]].copy()
for t, p in apply(models9, np.hstack([E, measures9(attach9(eps[["title", "words"]]))])).items():
    scores[t] = p
for t, p in apply(models11, np.hstack([E, measures11(eps[["title", "words"]])])).items():
    scores[f"social_{t}"] = p
per_series = scores.drop(columns="title").groupby("series").mean()
per_series["episodes_scored"] = scores.groupby("series").size()
per_series.reset_index().to_parquet(DERIVED / "series_classifier_scores.parquet", index=False)
print(f"wrote series_classifier_scores.parquet: {len(per_series):,} series", flush=True)
print(per_series.describe().loc[["mean", "50%"]].round(3).to_string())
