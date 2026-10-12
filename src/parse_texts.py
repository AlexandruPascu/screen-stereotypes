"""Parse every plot summary into who-does-what tuples (private intermediate output, resumable in parts).

For each summary (one text per title: the longest of its sources) spaCy finds:
  subj     subject of an active verb          "Maya rescues him"      -> (maya, rescue)
  pass     subject of a passive verb          "her father is kidnapped" -> (father, kidnap)
  obj      direct object                      "rescues him"           -> (him, rescue)
  desc     what a subject is called/described "she is ruthless", "a ruthless woman" -> (she, ruthless)
The mention is kept as written (pronoun, noun, or the first and last token of a name); its gender is resolved later
from pronouns, gendered nouns and the title's own cast list.

Output: DERIVED/parse/part-NNN.parquet with columns title, role, mention, mention_kind, last, lemma.
"""
import sys
from multiprocessing import Pool

import pandas as pd
import spacy

from paths import DERIVED

OUT = DERIVED / "parse"
OUT.mkdir(exist_ok=True)
CHUNK = 200_000
PRONOUNS = {"she", "her", "herself", "he", "him", "himself"}

texts = pd.read_parquet(DERIVED / "texts.parquet", columns=["title", "source", "text", "words"])
texts = texts[texts.words >= 5].sort_values("words", ascending=False).drop_duplicates("title")
texts = texts.sort_values("title").reset_index(drop=True)
print(f"summaries to parse: {len(texts):,} ({texts.words.sum():,} words)", flush=True)
nlp = None


def init():
    global nlp
    nlp = spacy.load("en_core_web_sm", disable=["ner"])


def work(batch):
    """Parse a batch inside the worker and return only the tuples (sending whole Doc objects back is the slow part)."""
    titles, docs = batch
    rows = []
    for doc, title in zip(nlp.pipe(docs, batch_size=256), titles):
        rows += [(title, *t) for t in tuples(doc)]
    return rows


def mention(tok):
    """(mention text, kind, last token) for a noun phrase head."""
    low = tok.lower_
    if low in PRONOUNS:
        return low, "pronoun", ""
    if tok.pos_ == "PROPN":
        parts = [c for c in tok.children if c.dep_ == "compound" and c.pos_ == "PROPN" and c.i < tok.i] + [tok]
        return parts[0].lower_, "name", tok.lower_
    if tok.pos_ == "NOUN":
        return tok.lemma_.lower() if tok.tag_ != "NNS" else low, "noun", ""
    return None


def tuples(doc):
    out = []
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass", "dobj"):
            m = mention(tok)
            if not m:
                continue
            head = tok.head
            role = {"nsubj": "subj", "nsubjpass": "pass", "dobj": "obj"}[tok.dep_]
            if role == "subj" and head.lemma_ == "be":
                for c in head.children:
                    if c.dep_ in ("acomp", "attr") and c.pos_ in ("ADJ", "NOUN"):
                        out.append(("desc", *m, c.lemma_.lower()))
                continue
            if head.pos_ in ("VERB", "AUX"):
                out.append((role, *m, head.lemma_.lower()))
        if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
            m = mention(tok.head)
            if m:
                out.append(("desc", *m, tok.lemma_.lower()))
    return out


workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
pool = Pool(workers, initializer=init)
for part, start in enumerate(range(0, len(texts), CHUNK)):
    path = OUT / f"part-{part:03d}.parquet"
    if path.exists():
        continue
    chunk = texts.iloc[start:start + CHUNK]
    batches = [(chunk.title.iloc[i:i + 2000].tolist(), chunk.text.iloc[i:i + 2000].tolist())
               for i in range(0, len(chunk), 2000)]
    rows = [r for part_rows in pool.imap_unordered(work, batches) for r in part_rows]
    pd.DataFrame(rows, columns=["title", "role", "mention", "mention_kind", "last", "lemma"]).to_parquet(path, index=False)
    print(f"part {part}: {start + len(chunk):,}/{len(texts):,} summaries, {len(rows):,} tuples", flush=True)
print("done", flush=True)
