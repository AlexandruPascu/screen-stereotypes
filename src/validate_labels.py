"""Step 3: check every measure against the labelling model's reading of the 3,000 sampled summaries.

Usage: validate_labels.py labels.jsonl [labels_second.jsonl]
  labels.jsonl          one record per item, valid against labelling/schema.json
  labels_second.jsonl   optional second pass on the items marked "double", for label reliability
Reports (results/labels/):
  reliability.csv   agreement between the two passes (share agreeing and Cohen's kappa) per label
  prevalence.csv    share of summaries with each label, from the random strata, by medium and era (95% intervals)
  measures.csv      each measure against the label it should track: precision in the stratum it flags, and AUC on the
                    random strata (does a higher score mean the label is more likely?)
"""
import json
import sys

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from paths import DERIVED, RESULTS

OUT = RESULTS / "labels"
OUT.mkdir(parents=True, exist_ok=True)
validator = Draft202012Validator(json.load(open(RESULTS.parent / "labelling/schema.json")))
key = pd.read_parquet(DERIVED / "labelling/key.parquet")


def load(path):
    records, bad = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if list(validator.iter_errors(r)):
            bad += 1
            continue
        records.append(r)
    print(f"{path}: {len(records):,} valid records, {bad:,} invalid", flush=True)
    return pd.DataFrame({
        "id": [r["id"] for r in records],
        "gender_any": [r["gender"]["portrayal"] != "none" for r in records],
        "gender_clear": [r["gender"]["portrayal"] == "clear" for r in records],
        "appearance": ["appearance" in r["gender"]["types"] for r in records],
        "damsel_victim": ["damsel_victim" in r["gender"]["types"] for r in records],
        "counter_stereotype": [r["gender"]["counter_stereotype"] for r in records],
        "group_mentioned": [len(r["groups"]) > 0 for r in records],
        "group_stereotyped": [any(g["portrayal"] == "stereotyped" for g in r["groups"]) for r in records],
        "prejudice_theme": [r["prejudice_theme"]["present"] for r in records],
        "slur": [r["slur_present"] for r in records],
        "any_stereotype": [r["gender"]["portrayal"] != "none" or any(g["portrayal"] == "stereotyped" for g in r["groups"])
                           for r in records],
        "confidence": [r["confidence"] for r in records],
    })


LABELS = ["gender_any", "gender_clear", "appearance", "damsel_victim", "counter_stereotype", "group_mentioned",
          "group_stereotyped", "prejudice_theme", "slur", "any_stereotype"]
first = load(sys.argv[1]).merge(key, on="id")
first[["women_appearance", "women_victim"]] = first[["women_appearance", "women_victim"]].fillna(0)  # no women: none
missing = set(key.id) - set(first.id)
print(f"labelled {len(first):,} of {len(key):,} items; missing {len(missing):,}", flush=True)

if len(sys.argv) > 2:
    second = load(sys.argv[2])
    both = first[["id"] + LABELS].merge(second[["id"] + LABELS], on="id", suffixes=("_1", "_2"))
    rel = pd.DataFrame([(label, len(both), (both[f"{label}_1"] == both[f"{label}_2"]).mean(),
                         cohen_kappa_score(both[f"{label}_1"], both[f"{label}_2"]))
                        for label in LABELS], columns=["label", "items", "agreement", "kappa"])
    rel.to_csv(OUT / "reliability.csv", index=False)
    print(rel.round(3).to_string(index=False), flush=True)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


rand = first[first.stratum.str.startswith("random")]
prev = []
for (medium, era), d in rand.groupby(["medium", "era"]):
    for label in LABELS:
        k, n = int(d[label].sum()), len(d)
        lo, hi = wilson(k, n)
        prev.append((medium, era, label, n, k / n, lo, hi))
prev = pd.DataFrame(prev, columns=["medium", "era", "label", "summaries", "share", "lo", "hi"])
prev.to_csv(OUT / "prevalence.csv", index=False)
pd.set_option("display.width", 200)
print(prev.pivot_table(index="label", columns=["medium", "era"], values="share").round(3).to_string(), flush=True)

# measure vs the label it should track: (measure, stratum it flags, label, score column on the random strata)
CHECKS = [
    ("classifier probability", "classifier top 2%", "any_stereotype", "stereotype_prob"),
    ("classifier probability (prejudice theme)", "classifier top 2%", "prejudice_theme", "stereotype_prob"),
    ("viewer stereotype tag", "viewer-tagged stereotype", "any_stereotype", "y"),
    ("group + pejorative words", "group + pejorative words", "group_stereotyped", "group_offensive"),
    ("group, no stereotype words", "group, no stereotype words", "group_stereotyped", "group_mention_only"),
    ("slur detector", "slur detected", "slur", "slur_mask"),
    ("woman described by looks", "woman described by looks", "appearance", "women_appearance"),
    ("woman as victim", "woman as victim", "damsel_victim", "women_victim"),
    ("gender-coded wording (top 5%)", "most gender-coded wording (top 5%)", "gender_any", "gender_congruence"),
    ("gender-coded wording (bottom 5%)", "least gender-coded wording (bottom 5%)", "gender_any", "gender_congruence"),
]
rows = []
for name, stratum, label, score in CHECKS:
    flagged = first[first.stratum == stratum]
    precision = flagged[label].mean() if len(flagged) else np.nan
    base = rand[label].mean()
    r = rand.dropna(subset=[score])
    x = (r[score] != 0).astype(float) if score in ("slur_mask",) else r[score].astype(float)
    auc = roc_auc_score(r[label], x) if r[label].nunique() == 2 and x.nunique() > 1 else np.nan
    rows.append((name, label, len(flagged), precision, base, precision / base if base else np.nan, len(r), auc))
m = pd.DataFrame(rows, columns=["measure", "label", "flagged_items", "share_labelled_in_flagged", "share_in_random",
                                "lift", "random_items", "auc_on_random"])
m.to_csv(OUT / "measures.csv", index=False)
print(m.round(3).to_string(index=False))
