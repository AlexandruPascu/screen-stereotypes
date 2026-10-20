"""Score every rated feature film with the two label-trained classifiers, for the film half of study 4.

Films: the rated feature films of ratings_films.py (IMDb, >= 20 votes, 1950–2026, no documentary or reality formats)
that have a summary of at least 15 words. Text: each film's longest summary of 15–350 words, the range the classifiers
were trained and calibrated on; a film whose every summary is longer uses its longest one cut to its first 350 words.
Classifiers: refitted the way score_sample.py (study 9: any stereotyped portrayal, gender, group, prejudice theme) and
social_classifier.py (study 11: the six groups) fit them, and checked against the sample scores those scripts saved,
before any film is encoded. Embeddings: EmbeddingGemma-2 (text encoder, "Classification" prompt) as for the labelled
and sampled summaries, encoded in parts of 2,000 films saved as they finish, so the run (10–17 hours on this CPU) can
stop and resume where it left off.
Usage: score_films.py [test]   (test: 300 films, separate files, to try the whole run quickly)
Output: DERIVED/film_classifier_scores.parquet (private; per film).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from label_models import BASICS, apply, attach9, con, fit, measures9, measures11, tsv
from paths import DERIVED

torch.set_num_threads(8)
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
PARTS = DERIVED / ("film_embeddings_test" if TEST else "film_embeddings")
PARTS.mkdir(exist_ok=True)
OUT = DERIVED / ("film_classifier_scores_test.parquet" if TEST else "film_classifier_scores.parquet")
PART_SIZE = 2000
MAX_WORDS = 350
models9, models11 = fit()

# --- the films ------------------------------------------------------------------------------------------------------
films = con.execute(f"""
    WITH f AS (SELECT b.tconst AS title FROM {BASICS} b
               JOIN {tsv('title.ratings.tsv.gz', ['tconst', 'averageRating', 'numVotes'])} r USING (tconst)
               WHERE b.titleType = 'movie' AND b.isAdult = '0' AND TRY_CAST(b.startYear AS INT) BETWEEN 1950 AND 2026
                 AND CAST(r.numVotes AS INT) >= 20
                 AND NOT regexp_matches(COALESCE(b.genres, ''), 'Documentary|Reality-TV|Talk-Show|Game-Show|News')),
    s AS (SELECT title, text, words FROM '{DERIVED / "texts.parquet"}' WHERE words >= 15 AND title IN (SELECT title FROM f))
    SELECT title,
           arg_max(text, (words, text)) FILTER (WHERE words <= {MAX_WORDS}) AS text_in,
           max(words) FILTER (WHERE words <= {MAX_WORDS}) AS words_in,
           arg_max(text, (words, text)) AS text_long
    FROM s GROUP BY title ORDER BY title""").df()
films["cut"] = films.text_in.isna()
films["text"] = films.text_in.where(~films.cut, films.text_long.str.split().str[:MAX_WORDS].str.join(" "))
films["words"] = films.words_in.where(~films.cut, MAX_WORDS).astype(int)
films = films[["title", "text", "words", "cut"]]
if TEST:
    films = films.sample(300, random_state=1).sort_values("title", ignore_index=True)
print(f"films to score: {len(films):,} ({films.cut.sum():,} with every summary over {MAX_WORDS} words, cut)", flush=True)

manifest = PARTS / "manifest.parquet"
if manifest.exists():
    assert pd.read_parquet(manifest).title.tolist() == films.title.tolist(), "the film list changed; start a fresh run"
else:
    films[["title", "words", "cut"]].to_parquet(manifest, index=False)
parts = [(i, PARTS / f"part_{i // PART_SIZE:05d}.npy") for i in range(0, len(films), PART_SIZE)]
todo = [(i, p) for i, p in parts if not p.exists()]
if todo:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
    started, done = time.time(), 0
    for i, path in todo:
        chunk = films.text.iloc[i:i + PART_SIZE].tolist()
        E = model.encode(chunk, prompt_name="Classification", batch_size=16, normalize_embeddings=True,
                         show_progress_bar=False)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, E)
        os.replace(tmp, path)  # a part counts as done only once it is fully written
        done += len(chunk)
        rate = done / (time.time() - started)
        left = sum(min(PART_SIZE, len(films) - j) for j, p in todo if not p.exists())
        print(f"  part {i // PART_SIZE + 1}/{len(parts)} saved; {rate:.1f} films/s; about {left / rate / 3600:.1f} h left",
              flush=True)
E = np.vstack([np.load(p) for _, p in parts])
assert len(E) == len(films)

f9 = attach9(films.drop(columns="text"))
scores = pd.DataFrame({"title": films.title, "words": films.words, "cut": films.cut})
for t, p in apply(models9, np.hstack([E, measures9(f9)])).items():
    scores[t] = p
for t, p in apply(models11, np.hstack([E, measures11(films.drop(columns="text"))])).items():
    scores[f"social_{t}"] = p
scores.to_parquet(OUT, index=False)
print(f"wrote {OUT.name}: {len(scores):,} films", flush=True)
print(scores.drop(columns=["title", "words", "cut"]).describe().loc[["mean", "50%", "max"]].round(3).to_string())
