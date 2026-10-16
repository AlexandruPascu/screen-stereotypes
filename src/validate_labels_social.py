"""Check the social-group measures against the labelling model's reading of the 2,685 sampled summaries.

Usage: validate_labels_social.py labels.jsonl [labels_second.jsonl]   (files in DERIVED/labelling_social/)
Reports (results/labels_social/):
  reliability.csv   agreement between the two passes (share agreeing and Cohen's kappa) per label
  prevalence.csv    share of random summaries with each label, by medium and era (95% intervals)
  measures.csv      each measure against the label it should track: share confirmed in the stratum it flags, and
                    for the mention lists the share of labelled mentions among random summaries they catch (recall)
  portrayal.csv     per group, among summaries the mention lists flag: how often the group is portrayed through a
                    stereotype, and which types of stereotype
  pejorative_pruning.csv  the pejorative labels before and after pruning them with these labels, with an
                    out-of-sample estimate for the pruning rule
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GROUP_SET", "social")  # the matcher below is the social one

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from sklearn.metrics import cohen_kappa_score

from group_stereotypes import SLUR, grams
from paths import DERIVED, LEXICONS, RESULTS

OUT = Path(os.environ["LABELS_OUT"]) if "LABELS_OUT" in os.environ else RESULTS / "labels_social"
OUT.mkdir(parents=True, exist_ok=True)
validator = Draft202012Validator(json.load(open(RESULTS.parent / "labelling/schema_social.json")))
key = pd.read_parquet(DERIVED / "labelling_social/key.parquet")
GROUPS = json.load(open(DERIVED / "title_groups_bits_social.json"))
BIT = {g: 1 << i for i, g in enumerate(GROUPS)}
TYPES = validator.schema["properties"]["social_groups"]["items"]["properties"]["types"]["items"]["enum"]


def load(path):
    records, bad = [], 0
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if list(validator.iter_errors(r)):
                bad += 1
                continue
            records.append(r)
    print(f"{path}: {len(records):,} valid records, {bad:,} invalid", flush=True)
    rows = []
    for r in records:
        entries = {e["group"]: e for e in r["social_groups"]}
        row = {"id": r["id"], "any_group": bool(entries),
               "any_stereotyped": any(e["portrayal"] == "stereotyped" for e in entries.values()),
               "pejorative_label": r["pejorative_label"], "discrimination": r["discrimination_theme"]["present"],
               "confidence": r["confidence"]}
        for g in GROUPS:
            e = entries.get(g)
            row[f"m_{g}"] = e is not None
            row[f"st_{g}"] = e is not None and e["portrayal"] == "stereotyped"
            row[f"ct_{g}"] = e is not None and e["portrayal"] == "counter_stereotyped"
            row[f"types_{g}"] = e["types"] if e is not None and e["portrayal"] == "stereotyped" else []
        rows.append(row)
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


first = load(DERIVED / "labelling_social" / sys.argv[1]).merge(key, on="id")
LABELS = ["any_group", "any_stereotyped", "pejorative_label", "discrimination"] + [f"m_{g}" for g in GROUPS]
if len(sys.argv) > 2:
    second = load(DERIVED / "labelling_social" / sys.argv[2])
    both = first[["id"] + LABELS].merge(second[["id"] + LABELS], on="id", suffixes=("_1", "_2"))
    rel = pd.DataFrame([(label, len(both), (both[f"{label}_1"] == both[f"{label}_2"]).mean(),
                         cohen_kappa_score(both[f"{label}_1"], both[f"{label}_2"]))
                        for label in LABELS], columns=["label", "items", "agreement", "kappa"])
    rel.to_csv(OUT / "reliability.csv", index=False)
    print(rel.round(3).to_string(index=False), flush=True)

pd.set_option("display.width", 220)
for scope, d in [("all labels", first), ("confidence 2-3", first[first.confidence >= 2])]:
    rand = d[d.stratum.str.startswith("random")]
    prev = []
    for (medium, era), c in rand.groupby(["medium", "era"]):
        for label in ["any_group", "any_stereotyped", "pejorative_label", "discrimination"]:
            k, n = int(c[label].sum()), len(c)
            prev.append((scope, medium, era, label, n, k / n, *wilson(k, n)))
    for g in GROUPS:  # groups are rare in random summaries: one estimate across eras and media
        for label in [f"m_{g}", f"st_{g}"]:
            k, n = int(rand[label].sum()), len(rand)
            prev.append((scope, "all", "all", label, n, k / n, *wilson(k, n)))
    prev = pd.DataFrame(prev, columns=["labels_used", "medium", "era", "label", "summaries", "share", "lo", "hi"])
    if scope == "all labels":
        prevalence = prev
        print(prev[prev.medium != "all"].pivot_table(index="label", columns=["medium", "era"], values="share").round(3)
              .to_string(), flush=True)
    else:
        prevalence = pd.concat([prevalence, prev])
prevalence.to_csv(OUT / "prevalence.csv", index=False)

rows, portrayal = [], []
for scope, d in [("all labels", first), ("confidence 2-3", first[first.confidence >= 2])]:
    rand = d[d.stratum.str.startswith("random")]
    for g in GROUPS:
        mention = d[d.stratum == f"mention: {g}"]
        paired = d[d.stratum == f"group + pejorative words: {g}"]
        labelled = d[d.stratum == f"pejorative label: {g}"]
        caught = rand[rand[f"m_{g}"]]
        found = (caught.mention_mask & BIT[g]) != 0
        rows += [(scope, g, "mention list", "group present", len(mention), mention[f"m_{g}"].mean()),
                 (scope, g, "mention list", "group stereotyped", len(mention), mention[f"st_{g}"].mean()),
                 (scope, g, "group + pejorative words", "group stereotyped", len(paired), paired[f"st_{g}"].mean()),
                 (scope, g, "pejorative label", "pejorative label", len(labelled), labelled.pejorative_label.mean()),
                 (scope, g, "mention list (recall on random)", "group present", len(caught),
                  found.mean() if len(caught) else np.nan)]
        if scope == "all labels":
            st = mention[mention[f"st_{g}"]]
            types = pd.Series([t for ts in st[f"types_{g}"] for t in ts]).value_counts()
            portrayal.append({"group": g, "flagged_summaries": len(mention), "group_present": mention[f"m_{g}"].sum(),
                              "stereotyped": mention[f"st_{g}"].sum(), "counter_stereotyped": mention[f"ct_{g}"].sum(),
                              **{f"type_{t}": int(types.get(t, 0)) for t in TYPES}})
    tagged = d[d.stratum == "viewer-tagged social theme"]
    noted = d[d.stratum == "first pass: other discrimination"]
    caught = rand[rand.any_group]
    rows += [(scope, "any", "viewer-tagged social theme", "group present", len(tagged), tagged.any_group.mean()),
             (scope, "any", "viewer-tagged social theme", "group stereotyped", len(tagged), tagged.any_stereotyped.mean()),
             (scope, "any", "viewer-tagged social theme", "discrimination theme", len(tagged), tagged.discrimination.mean()),
             (scope, "any", "first pass: other discrimination", "discrimination theme", len(noted),
              noted.discrimination.mean()),
             (scope, "any", "mention list (recall on random)", "group present", len(caught),
              (caught.mention_mask != 0).mean())]
measures = pd.DataFrame(rows, columns=["labels_used", "group", "measure", "label", "items", "share"])
measures.to_csv(OUT / "measures.csv", index=False)
print(measures[measures.labels_used == "all labels"].pivot_table(index=["measure", "label"], columns="group",
                                                                  values="share").round(2).to_string(), flush=True)
portrayal = pd.DataFrame(portrayal)
portrayal.to_csv(OUT / "portrayal.csv", index=False)
print(portrayal.loc[:, (portrayal != 0).any()].to_string(index=False))

# The pejorative labels were pruned with these labels (LEXICONS/social_pejoratives.json, "pruned"), so their precision
# on the items the pruned list still flags is in-sample. The rule behind the pruning (drop the labels these labels
# never confirm) is scored out of sample instead: chosen on a random half of the flagged items, scored on the other.
texts = {r["id"]: r["text"] for r in map(json.loads, open(DERIVED / "labelling_social/items.jsonl"))}
full_list = SLUR.keys() | set(json.load(open(LEXICONS / "social_pejoratives.json"))["pruned"]["terms"])
flagged = first[first.stratum.str.startswith("pejorative label:")]
hits = [grams(texts[i]) & full_list for i in flagged.id]
now = np.array([bool(grams(texts[i]) & SLUR.keys()) for i in flagged.id])
y = flagged.pejorative_label.to_numpy()
rng, held_out = np.random.default_rng(0), []
for _ in range(1000):
    half = rng.permutation(len(flagged)) < len(flagged) // 2
    seen = set().union(*(h for h, a in zip(hits, half) if a))
    confirmed = set().union(*(h for h, a, c in zip(hits, half, y) if a and c))
    keep = [c for h, a, c in zip(hits, half, y) if not a and h - (seen - confirmed)]
    if keep:
        held_out.append(np.mean(keep))
pr = pd.DataFrame([
    ("full list", len(flagged), y.mean(), *wilson(y.sum(), len(flagged)), 1.0),
    ("pruned list (in-sample)", now.sum(), y[now].mean(), *wilson(y[now].sum(), now.sum()), y[now].sum() / y.sum()),
    ("pruning rule on held-out halves", np.nan, np.mean(held_out), *np.percentile(held_out, [2.5, 97.5]), np.nan)],
    columns=["pejorative_list", "flagged_items", "share_confirmed", "lo", "hi", "confirmed_labels_still_found"])
pr.to_csv(OUT / "pejorative_pruning.csv", index=False)
print(pr.round(3).to_string(index=False))
