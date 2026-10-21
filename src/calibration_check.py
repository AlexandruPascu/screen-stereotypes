"""Study 9's check of the label-trained classifier against the labels: its estimates on the fresh sample (score_sample.py),
per medium and era and weighted to the population, beside the careful labels' own share and 95% interval for the same
medium and era (validate_labels.py, random items).
Output: results/label_classifier/calibration_check_vs_labels.csv.
"""
import numpy as np
import pandas as pd

from paths import DERIVED, RESULTS

TARGETS = ["any_stereotype", "gender_any", "group_stereotyped", "prejudice_theme"]
s = pd.read_parquet(DERIVED / "label_classifier_sample_scores.parquet", columns=["year", "medium", "weight"] + TARGETS)
s["era"] = pd.cut(s.year, [0, 1969, 1999, 2100], labels=["before 1970", "1970-99", "2000+"]).astype("string")
prev = pd.read_csv(RESULTS / "labels/prevalence.csv")
rows = []
for (medium, era), x in s.groupby(["medium", "era"]):
    for target in TARGETS:
        p = prev[(prev.medium == medium) & (prev.era == era) & (prev.label == target)].iloc[0]
        est = np.average(x[target], weights=x.weight)
        rows.append((medium, era, target, est, p.share, p.lo, p.hi, p.lo <= est <= p.hi))
check = pd.DataFrame(rows, columns=["medium", "era", "target", "classifier", "labels", "labels_lo", "labels_hi",
                                    "inside_interval"])
check.to_csv(RESULTS / "label_classifier/calibration_check_vs_labels.csv", index=False)
print(check.round(3).to_string(index=False))
print(f"{check.inside_interval.sum()} of {len(check)} estimates inside the labels' 95% intervals", flush=True)
