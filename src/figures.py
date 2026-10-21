"""README figures built only from the published tables in results/, so they can be redrawn without the private data."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RESULTS

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
WOMEN, MEN, GREY = "#c0392b", "#2c3e50", "#7f8c8d"


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# 3. verbs and descriptions most tied to women vs men (identity words such as "gay" or "American" left out: they
# say who a character is, not how they are portrayed)
IDENTITY = set("""gay lesbian queer american british english french german italian irish black white jewish muslim
christian catholic african asian mexican russian chinese japanese korean indian native latino latina hispanic""".split())
w = pd.read_csv(RESULTS / "agency/distinctive_words.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 6))
for ax, (role, title) in zip(axes, [("subj", "what they do"), ("pass", "what is done to them"),
                                    ("desc", "how they are described")]):
    d = w[(w.role == role) & (w.era == "all") & ((w.f + w.m) >= 200) & ~w.lemma.isin(IDENTITY)].sort_values("z")
    d = pd.concat([d.head(12), d.tail(12)])
    colours = [WOMEN if z > 0 else MEN for z in d.z]
    ax.barh(range(len(d)), d.z, color=colours)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.lemma)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("← men    association (z)    women →")
fig.suptitle("Plot summaries, 1920–2026: words most tied to women (red) and men (dark); identity words left out",
             fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "agency/distinctive_words.png")
plt.close(fig)

# 4. ratings and votes
m = pd.read_csv(RESULTS / "ratings/models.csv")
labels = {"cast_women_share": "women's share of the credited cast (0→100%)",
          "guest_women_share": "women's share of guest stars (0→100%)",
          "summary_women_active": "women's share of who acts in the summary (0→100%)",
          "summary_women_looks": "summary describes a woman by her looks",
          "summary_women_victim": "summary has a woman as a victim of violence",
          "group_stereotype": "summary pairs a group with its stereotype words",
          "offensive_stereotype": "same, pejorative stereotype words only",
          "slur": "summary contains a slur",
          "social_named": "summary names one of study 11's six groups",
          "social_stereotype": "names one of them with its stereotype words",
          "social_pejorative": "names one of them with pejorative stereotype words",
          "social_label": "calls one of them by an insulting label"}
order = list(labels)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, (outcome, title, scale) in zip(axes, [("rating", "change in IMDb rating (points out of 10)", 1),
                                              ("log_votes", "change in number of votes (%)", None)]):
    d = m[m.outcome == outcome].set_index("measure").reindex(order[::-1]).dropna(subset=["estimate"])
    est, lo, hi = d.estimate, d.lo, d.hi
    if scale is None:
        est, lo, hi = (np.exp(est) - 1) * 100, (np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100
    y = np.arange(len(d))
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color=MEN, ecolor=GREY, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in d.index])
    ax.set_title(title, fontsize=10)
axes[0].set_xlim(-0.15, 0.15)
fig.suptitle("Episodes compared within the same series: the differences are tiny (95% intervals)", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "ratings/effects.png")
plt.close(fig)

# 5a. slurs in credited character names
s = pd.read_csv(RESULTS / "groups/slurs_in_character_names_by_decade.csv")
s = s[s.decade >= 1920]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(s.decade.astype(int).astype(str) + "s", s.per_100k, color=MEN)
ax.set_ylabel("per 100,000 credited roles")
ax.set_title("Characters credited with an ethnic, religious or anti-gay slur in their name")
fig.tight_layout()
fig.savefig(RESULTS / "groups/slurs_in_character_names.png")
plt.close(fig)

# 5b. pejorative stereotype pairing over time
g = pd.read_csv(RESULTS / "groups_offensive/group_lift_by_decade.csv")
g = g[g.decade.between(1950, 2020) & g.lift_length_adjusted.notna()]  # summary length held fixed
names = {"native_american": "Native American", "german": "German", "italian": "Italian", "russian": "Russian",
         "muslim": "Muslim", "arab": "Arab", "mexican": "Mexican", "chinese": "Chinese", "black": "Black"}
fig, ax = plt.subplots(figsize=(10, 5.5))
for grp, label in names.items():
    d = g[(g.group == grp) & (g.n_in >= 100)].sort_values("decade")
    ax.plot(d.decade, d.lift_length_adjusted, marker="o", ms=3, label=label)
pooled = g[g.n_in >= 50].groupby("decade").apply(
    lambda x: np.exp(np.average(np.log(x.lift_length_adjusted), weights=x.n_in)), include_groups=False)
ax.plot(pooled.index, pooled.values, color="black", lw=3, label="all groups (weighted)")
ax.axhline(1, color=GREY, lw=0.8)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10])
ax.set_yticklabels(["0.5×", "1×", "2×", "5×", "10×"])
ax.set_ylabel("how much more often pejorative stereotype words appear\nwhen the summary mentions the group "
              "(same summary length)")
ax.set_title("Pejorative stereotype words next to group mentions in plot summaries")
ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(RESULTS / "groups_offensive/pejorative_pairing_by_decade.png")
plt.close(fig)

# 5c. jobs given to characters credited by a group label ("Mexican bandit", "Russian agent")
c = pd.read_csv(RESULTS / "groups/group_labelled_characters.csv")
c = c[c.with_job >= 40].sort_values("crime_share_of_jobs")
lo, hi = wilson(c.crime, c.with_job)
base = c.crime_share_all_characters.iloc[0]
fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(c))
ax.barh(y, c.crime_share_of_jobs * 100, xerr=[(c.crime_share_of_jobs - lo) * 100, (hi - c.crime_share_of_jobs) * 100],
        color=MEN, ecolor=GREY)
ax.axvline(base * 100, color=WOMEN, lw=2, label=f"all characters with a job ({base:.1%})")
ax.set_yticks(y)
ax.set_yticklabels([f"{x.replace('_', ' ').title()} (n={n})" for x, n in zip(c.groups, c.with_job)])
ax.set_xlabel("% of their jobs that are criminal (gangster, thief, drug dealer, killer, terrorist…)")
ax.set_title("Characters credited by group: how often their job is a crime")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS / "groups/criminal_jobs_by_group_label.png")
plt.close(fig)
print("figures written")

# ---- step 1 and step 2 figures (palette: reference categorical slots in fixed order; solid reference lines) ----
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
PERIODS = ["pre1950"] + [str(d) for d in range(1950, 2030, 10)]
LABELS = ["before\n1950"] + [f"{d}s" for d in range(1950, 2030, 10)]


def end_labels(ax, series, x_end=None):
    """Direct labels at the right end of each line, in ink colours, nudged apart so they never overlap."""
    x_end = len(PERIODS) - 1 if x_end is None else x_end
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    placed = []
    for name, y in sorted(series, key=lambda t: t[1]):
        y_text = max([y] + [p + gap for p in placed])
        placed.append(y_text)
        ax.annotate(name, (x_end, y), xytext=(x_end + 0.15, y_text), textcoords="data", va="center", fontsize=8,
                    color=INK)


sb = pd.read_csv(RESULTS / "embeddings/set_bias_by_period.csv", dtype={"period": str})
pick = lambda contrast, s: sb[(sb.contrast == contrast) & (sb.set == s)].set_index("period")["mean"].reindex(PERIODS)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
x = np.arange(len(PERIODS))
ax = axes[0]
ends = []
for (s, name), colour in zip([("adjective:appearance", "appearance"), ("adjective:intelligence", "intelligence"),
                              ("adjective:williams-best", "personality (general)"), ("occupation", "occupations")], SLOTS):
    y = pick("gender", s)
    ax.plot(x, y, color=colour, lw=2, marker="o", ms=5, label=name)
    ends.append((name, y.iloc[-1]))
gender_ends = ends
ax.axhline(0, color=RULE, lw=1)
ax.set_title("Gender: which words sit closer to women's words", fontsize=10, color=INK)
ax.set_ylabel("← nearer men's words      nearer women's words →", color=MUTED)
ax = axes[1]
ends = []
for grp, colour in zip(["asian", "chinese", "hispanic", "russian"], SLOTS):
    c = f"{grp} vs white surnames"
    y = pick(c, "adjective:otherization") - pick(c, "adjective:williams-best")
    ax.plot(x, y, color=colour, lw=2, marker="o", ms=5, label=f"{grp.title()} surnames")
    ends.append((grp.title(), y.iloc[-1]))
group_ends = ends
ax.axhline(0, color=RULE, lw=1)
ax.set_title("'Outsider' adjectives near group surnames\n(beyond ordinary personality adjectives, vs White surnames)",
             fontsize=10, color=INK)
ax.set_ylabel("extra closeness of outsider adjectives", color=MUTED)
ax = axes[2]
y = pick("islam vs christianity", "terrorism")
ax.plot(x, y, color=SLOTS[0], lw=2, marker="o", ms=5)
ax.axhline(0, color=RULE, lw=1)
ax.set_title("Terrorism words: nearer Islam than\nChristianity words", fontsize=10, color=INK)
ax.set_ylabel("← nearer Christianity      nearer Islam →", color=MUTED)
for ax in axes:
    ax.set_xticks(x[::2])
    ax.set_xticklabels(LABELS[::2], fontsize=8)
    ax.set_xlim(-0.3, len(PERIODS) - 0.3 + (1.6 if ax is not axes[2] else 0))
axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
end_labels(axes[0], gender_ends)
end_labels(axes[1], group_ends)
fig.suptitle("Word embeddings trained on plot summaries, by decade (Garg et al. 2018 word lists; 3 training runs)",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(RESULTS / "embeddings/bias_by_decade.png")
plt.close(fig)

# language vs casting: occupation word bias against women's on-screen share of the job, 2010s
wb = pd.read_csv(RESULTS / "embeddings/word_bias_by_period.csv", dtype={"period": str})
wb = wb[(wb.contrast == "gender") & (wb.set == "occupation") & (wb.period == "2010")]
cast = pd.read_csv(RESULTS / "occupations/occupation_by_decade_english.csv")
cast = cast[cast.decade == 2010].groupby("occupation")[["roles", "women"]].sum()
cast = cast[cast.roles >= 30]
v = wb.merge((cast.women / cast.roles).rename("share").reset_index(), left_on="word", right_on="occupation")
census_r = pd.read_csv(RESULTS / "embeddings/validation_census.csv", dtype={"period": str}).set_index("period").r
r = np.corrcoef(v["mean"], np.log(v.share.clip(0.01, 0.99) / (1 - v.share.clip(0.01, 0.99))))[0, 1]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(v["mean"], v.share * 100, s=40, color=SLOTS[0], edgecolor="white", linewidth=1)
SHOW = {"nurse", "librarian", "housekeeper", "dancer", "teacher", "secretary", "clerk", "doctor", "lawyer", "judge",
        "scientist", "professor", "soldier", "mechanic", "engineer", "clergy", "dentist", "surgeon"}
for word, bx, sy in zip(v.word, v["mean"], v.share * 100):
    if word in SHOW:  # a selection, so labels never collide; every point is in the published table
        ax.annotate(word, (bx, sy), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK)
ax.axvline(0, color=RULE, lw=1)
ax.set_xlabel("← job word sits nearer men's words      nearer women's words →  (plot-summary embeddings, 2010s)",
              color=MUTED)
ax.set_ylabel("% of the job's roles played by women (2010s)", color=MUTED)
ax.set_title(f"The language of plot summaries tracks casting (r = {r:.2f}); against the census r = "
             f"{census_r.min():.2f}–{census_r.max():.2f} by decade", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(RESULTS / "embeddings/language_vs_casting.png")
plt.close(fig)

# classifier: calibration and scores by decade
cal_all = pd.read_csv(RESULTS / "classifier/calibration.csv")
dec = pd.read_csv(RESULTS / "classifier/score_by_decade.csv")
dec = dec[dec.decade.between(1950, 2020)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
ax.plot([0, 0.7], [0, 0.7], color=RULE, lw=1)
for (label, name), colour in zip([("raw", "model scores"), ("calibrated", "calibrated probabilities")], SLOTS):
    c = cal_all[cal_all.scores == label]
    ax.plot(c.predicted, c.observed, color=colour, lw=2, marker="o", ms=6, label=name)
ax.set_xlabel("predicted probability (cross-validated, matched sample)", color=MUTED)
ax.set_ylabel("share actually tagged by viewers", color=MUTED)
ax.set_title("Calibration: raw scores are under-confident; calibrated\nprobabilities follow the diagonal", fontsize=10,
             color=INK)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax = axes[1]
for (medium, label), colour in zip([("film", "films"), ("tv", "TV episodes")], SLOTS):
    d = dec[dec.medium == medium].sort_values("decade")
    ax.plot(d.decade, d.mean_prob * 100, color=colour, lw=2, marker="o", ms=5, label=label)
    ax.annotate(label, (d.decade.iloc[-1], d.mean_prob.iloc[-1] * 100), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK)
ax.set_xlim(1945, 2030)
ax.set_xticks(range(1950, 2030, 10))
ax.set_ylim(0, None)
ax.set_ylabel("average probability of a stereotype tag (%)", color=MUTED)
ax.set_title("Classifier score of every summary, by decade", fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(RESULTS / "classifier/classifier.png")
plt.close(fig)
print("step 1-2 figures written")

# ---- labels: prevalence (random strata) and how each measure holds up ----
prev = pd.read_csv(RESULTS / "labels/prevalence.csv")
ERAS = ["before 1970", "1970-99", "2000+"]
show = [("any_stereotype", "any stereotyped portrayal"), ("gender_any", "gender stereotype"),
        ("group_stereotyped", "group stereotype"), ("prejudice_theme", "story about prejudice")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (medium, title) in zip(axes, [("film", "Films"), ("tv", "TV")]):
    ends = []
    for (label, name), colour in zip(show, SLOTS):
        d = prev[(prev.medium == medium) & (prev.label == label)].set_index("era").reindex(ERAS)
        x = np.arange(len(ERAS))
        ax.errorbar(x, d.share * 100, yerr=[(d.share - d.lo) * 100, (d.hi - d.share) * 100], color=colour, lw=2,
                    marker="o", ms=6, capsize=3, label=name)
        ends.append((name, d.share.iloc[-1] * 100))
    ax.set_xticks(range(len(ERAS)))
    ax.set_xticklabels(ERAS)
    ax.set_xlim(-0.3, len(ERAS) - 0.3 + 1.1)
    lo_, hi_ = ax.get_ylim()
    placed = []
    for name, y in sorted(ends, key=lambda t: t[1]):
        y_text = max([y] + [p + (hi_ - lo_) * 0.055 for p in placed])
        placed.append(y_text)
        ax.annotate(name, (len(ERAS) - 1, y), xytext=(len(ERAS) - 1 + 0.12, y_text), textcoords="data", va="center",
                    fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
axes[0].set_ylabel("% of summaries (random sample, 95% intervals)", color=MUTED)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
fig.suptitle("How common stereotypes are in plot summaries, read by a labelling model (1,400 random summaries)",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(RESULTS / "labels/prevalence.png")
plt.close(fig)

ms = pd.read_csv(RESULTS / "labels/measures.csv")
ms = ms[(ms.labels_used == "all labels") & ~ms.measure.isin(["group, no stereotype words",
                                                              "gender-coded wording (bottom 5%)"])]
names = {"woman described by looks": "Woman described by looks → appearance stereotype",
         "gender-coded wording (top 5%)": "Most gender-coded wording → any gender stereotype",
         "woman as victim": "Woman as victim → damsel stereotype",
         "group + pejorative words": "Group + pejorative words → group stereotype",
         "slur detector": "Slur detector, first list → slur",
         "slur detector, pruned": "Slur detector, pruned list (held-out estimate) → slur",
         "classifier probability (prejudice theme)": "Classifier top 2% → story about prejudice",
         "classifier probability": "Classifier top 2% → stereotyped portrayal",
         "viewer stereotype tag (prejudice theme)": "Viewer stereotype tag → story about prejudice",
         "viewer stereotype tag": "Viewer stereotype tag → stereotyped portrayal"}
# the slur list was pruned with these labels; its out-of-sample precision stands in for the in-sample one
sp = pd.read_csv(RESULTS / "labels/slur_pruning.csv").set_index("slur_list")
ms = pd.concat([ms, pd.DataFrame([{
    "measure": "slur detector, pruned", "share_labelled_in_flagged": sp.share_with_slur["pruning rule on held-out halves"],
    "share_in_random": ms.loc[ms.measure == "slur detector", "share_in_random"].iloc[0]}])])
ms = ms[ms.measure.isin(names)].assign(name=lambda d: d.measure.map(names)).sort_values("share_labelled_in_flagged")
fig, ax = plt.subplots(figsize=(10, 5.2))
y = np.arange(len(ms))
for i, (b, p) in enumerate(zip(ms.share_in_random * 100, ms.share_labelled_in_flagged * 100)):
    ax.plot([b, p], [i, i], color=RULE, lw=2, zorder=1)
ax.scatter(ms.share_in_random * 100, y, color=MUTED, s=45, zorder=2, label="share in random summaries", edgecolor="white")
ax.scatter(ms.share_labelled_in_flagged * 100, y, color=SLOTS[0], s=55, zorder=3, label="share in what the measure flags",
           edgecolor="white")
ax.set_yticks(y)
ax.set_yticklabels(ms.name, fontsize=8)
ax.set_xlabel("% of summaries the labelling model judged positive", color=MUTED)
ax.set_title("How each measure holds up against a careful reading of 3,000 summaries", fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS / "labels/measures.png")
plt.close(fig)
print("label figures written")

# ---- label-trained classifier: which text representation predicts the careful labels best ----
cv = pd.read_csv(RESULTS / "label_classifier/cv_metrics.csv")
cv = cv[cv.scope == "random items"]
TNAMES = {"any_stereotype": "any stereotyped portrayal", "gender_any": "gender stereotype",
          "group_stereotyped": "group stereotype", "prejudice_theme": "story about prejudice"}
MNAMES = {"glove": "GloVe word vectors (2014)", "tf-idf": "word counts (TF-IDF)", "measures": "hand-built measures",
          "mpnet": "mpnet sentence embeddings (2021)", "qwen3": "Qwen3-Embedding-0.6B (2025)",
          "gemma2": "EmbeddingGemma-2 (2026)"}
order = ["glove", "measures", "tf-idf", "mpnet", "qwen3", "gemma2"]
fig, ax = plt.subplots(figsize=(10, 4.8))
offsets = np.linspace(0.27, -0.27, len(TNAMES))  # first legend entry on top, as in the legend
for (target, name), colour, off in zip(TNAMES.items(), SLOTS, offsets):
    d = cv[cv.target == target].set_index("model").reindex(order)
    ax.scatter(d.auc, np.arange(len(order)) + off, color=colour, s=50, edgecolor="white", linewidth=1, label=name, zorder=3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([MNAMES[m] for m in order], fontsize=9)
ax.axvline(0.5, color=RULE, lw=1)
ax.set_xlim(0.6, 0.95)
ax.set_xlabel("AUC on the 1,400 random labelled summaries (series held out, 3 repeats)", color=MUTED)
ax.set_title("Predicting the careful labels: newer embedding models do best", fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS / "label_classifier/model_comparison.png")
plt.close(fig)
print("label classifier figure written")

# ---- label-trained classifier applied to a fresh random sample: trends by decade and genre ----
sp = pd.read_csv(RESULTS / "label_classifier/sample_prevalence_by_decade.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (medium, title) in zip(axes, [("film", "Films"), ("tv", "TV")]):
    d = sp[(sp.medium == medium) & (sp.summaries >= 500)].sort_values("decade")
    ends = []
    for (target, name), colour in zip(TNAMES.items(), SLOTS):
        ax.plot(d.decade, d[target] * 100, color=colour, lw=2, marker="o", ms=5, label=name)
        ends.append((name, d[target].iloc[-1] * 100))
    lo_, hi_ = 0, max(22, sp[["any_stereotype"]].max().iloc[0] * 100 + 2)
    ax.set_ylim(lo_, hi_)
    placed = []
    for name, yv in sorted(ends, key=lambda t: t[1]):
        y_text = max([yv] + [p + (hi_ - lo_) * 0.055 for p in placed])
        placed.append(y_text)
        ax.annotate(name, (d.decade.iloc[-1], yv), xytext=(d.decade.iloc[-1] + 2, y_text), textcoords="data",
                    va="center", fontsize=8, color=INK)
    ax.set_xlim(d.decade.min() - 4, d.decade.max() + 28)
    ax.set_xticks(range(int(d.decade.min()), 2030, 10))
    ax.set_xticklabels([f"{x}s" for x in range(int(d.decade.min()), 2030, 10)], fontsize=8)
    ax.set_title(title, fontsize=10, color=INK)
axes[0].set_ylabel("% of summaries (calibrated classifier)", color=MUTED)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
fig.suptitle("Stereotypes in a fresh random sample of 37,049 summaries, scored by the label-trained classifier",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(RESULTS / "label_classifier/sample_trends.png")
plt.close(fig)

sg = pd.read_csv(RESULTS / "label_classifier/sample_prevalence_by_genre.csv").sort_values("any_stereotype")
fig, ax = plt.subplots(figsize=(9, 6.5))
y = np.arange(len(sg))
ax.barh(y, sg.any_stereotype * 100, color=SLOTS[0], height=0.65, label="any stereotyped portrayal")
# one summary can carry both kinds, so the parts are shown as marks on the bar, never stacked into a false sum
ax.scatter(sg.group_stereotyped * 100, y, color=SLOTS[2], s=40, edgecolor="white", linewidth=1, zorder=3,
           label="of which: group stereotype")
for yi, total in zip(y, sg.any_stereotype * 100):
    ax.annotate(f"{total:.0f}%", (total + 0.4, yi), va="center", fontsize=8, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels(sg.genre, fontsize=8)
ax.set_xlabel("% of summaries (calibrated classifier, sample weighted to the corpus)", color=MUTED)
ax.set_title("Stereotyped portrayals by genre, all decades", fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS / "label_classifier/sample_by_genre.png")
plt.close(fig)
print("sample figures written")

# ---- behind the camera, leads, ages, writers (IMDb crew data) ----
BC = RESULTS / "behind_camera"
jb = pd.read_csv(BC / "jobs_by_decade.csv")
jb = jb.assign(w=jb.women_share * jb.with_gender).groupby(["job", "medium", "decade"])[["w", "with_gender"]].sum().reset_index()
jb["share"] = jb.w / jb.with_gender
JOBS = [("director", "Directors"), ("writer", "Writers"), ("producer", "Producers"), ("editor", "Editors"),
        ("cinematographer", "Cinematographers"), ("composer", "Composers"),
        ("production_designer", "Production designers"), ("casting_director", "Casting directors")]
fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharey=True, sharex=True)
for ax, (job, title) in zip(axes.ravel(), JOBS):
    for (medium, name), colour in zip([("film", "films"), ("tv", "TV")], SLOTS):
        d = jb[(jb.job == job) & (jb.medium == medium) & (jb.with_gender >= 200)].sort_values("decade")
        ax.plot(d.decade, d.share * 100, color=colour, lw=2, marker="o", ms=3, label=name)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xticks([1950, 1970, 1990, 2010])
    ax.set_xticklabels(["1950s", "1970s", "1990s", "2010s"], fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("% women", color=MUTED)
axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Women's share of jobs behind the camera, fiction film and TV (IMDb credits; gender from credits or first "
             "name; the first point pools 1920–49)", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(BC / "jobs_by_decade.png")
plt.close(fig)

ld = pd.read_csv(BC / "leads_by_decade.csv")
ld = ld.assign(w=ld.women_share * ld.credits).groupby(["medium", "decade", "position"])[["w", "credits"]].sum().reset_index()
ld["share"] = ld.w / ld.credits
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (medium, title) in zip(axes, [("film", "Films"), ("tv", "TV series")]):
    ends = []
    for (pos, name), colour in zip([("1st listed", "1st listed"), ("2nd-3rd listed", "2nd–3rd listed"),
                                    ("4th or later", "4th or later")], SLOTS):
        d = ld[(ld.medium == medium) & (ld.position == pos) & (ld.credits >= 500)].sort_values("decade")
        ax.plot(d.decade, d.share * 100, color=colour, lw=2, marker="o", ms=4, label=name)
        ends.append((name, d.share.iloc[-1] * 100, d.decade.iloc[-1]))
    ax.set_ylim(15, 50)
    placed = []
    for name, yv, xv in sorted(ends, key=lambda t: t[1]):
        y_text = max([yv] + [p + 1.8 for p in placed])
        placed.append(y_text)
        ax.annotate(name, (xv, yv), xytext=(xv + 2, y_text), textcoords="data", va="center", fontsize=8, color=INK)
    ax.set_xlim(1935, 2048)
    ax.set_xticks(range(1940, 2030, 20))
    ax.set_xticklabels(["1920–49", "1960s", "1980s", "2000s", "2020s"])
    ax.set_title(title, fontsize=10, color=INK)
axes[0].set_ylabel("% of acting credits held by women", color=MUTED)
axes[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("Women in leading roles: share by position in the credited cast", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(BC / "leads_by_decade.png")
plt.close(fig)

ag = pd.read_csv(BC / "ages_by_decade.csv")
ag = ag.assign(a=ag.mean_age * ag.with_age).groupby(["decade", "female"])[["a", "with_age"]].sum().reset_index()
ag["age"] = ag.a / ag.with_age
gp = pd.read_csv(BC / "lead_age_gap_by_decade.csv")
gp = gp.assign(g=gp.mean_gap * gp.films).groupby("decade")[["g", "films"]].sum().reset_index()
gp["gap"] = gp.g / gp.films
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for (female, name), colour in zip([(False, "men in top-3 roles"), (True, "women in top-3 roles")], SLOTS):
    d = ag[ag.female == female].sort_values("decade")
    axes[0].plot(d.decade, d.age, color=colour, lw=2, marker="o", ms=4, label=name)
    axes[0].annotate(name, (d.decade.iloc[-1], d.age.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                     fontsize=8, color=INK)
axes[0].set_xlim(1935, 2045)
axes[0].set_xticks(range(1940, 2030, 20))
axes[0].set_xticklabels(["1930–49", "1960s", "1980s", "2000s", "2020s"])
axes[0].set_ylabel("average age at release", color=MUTED)
axes[0].set_title("Ages of the top-3 cast (films with 100+ votes); birth years known for\n98% of leads before 1950 but 54% of "
                  "2020s leads, so recent ages are likely too high", fontsize=10, color=INK)
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
axes[1].plot(gp.decade, gp.gap, color=SLOTS[0], lw=2, marker="o", ms=4)
axes[1].axhline(0, color=RULE, lw=1)
axes[1].set_ylim(0, None)
axes[1].set_xticks(range(1940, 2030, 20))
axes[1].set_xticklabels(["1930–49", "1960s", "1980s", "2000s", "2020s"])
axes[1].set_ylabel("years", color=MUTED)
axes[1].set_title("How much older the first-listed man is than the first-listed woman\n(same film; films where both "
                  "birth years are known)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(BC / "ages_by_decade.png")
plt.close(fig)

wm = pd.read_csv(BC / "writer_models.csv")
wm = wm[wm.variable == "writers_women"]
ONAMES = {"cast_women": "women's share of the credited cast", "top3_women": "women's share of the top-3 roles",
          "first_woman": "first-listed role is a woman's"}
designs = ["TV episodes: same series and year", "films: same director", "films: same genre and decade"]
fig, ax = plt.subplots(figsize=(10, 4))
for (outcome, name), colour, off in zip(ONAMES.items(), SLOTS, [0.22, 0, -0.22]):
    d = wm[wm.outcome == outcome].set_index("design").reindex(designs)
    yv = np.arange(len(designs)) + off
    ax.errorbar(d.estimate * 100, yv, xerr=[(d.estimate - d.lo) * 100, (d.hi - d.estimate) * 100], fmt="o",
                color=colour, ecolor=RULE, capsize=3, label=name)
ax.axvline(0, color=RULE, lw=1)
ax.set_yticks(range(len(designs)))
ax.set_yticklabels(designs, fontsize=9)
ax.set_xlabel("difference in percentage points, all-women vs all-men writing team (95% intervals)", color=MUTED)
ax.set_title("Casting differs with who writes — strongly in films, modestly in TV episodes", fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(BC / "writer_effects.png")
plt.close(fig)
print("behind-camera figures written")

# ---- study 4 for films: the same question asked three ways ----
fm = pd.read_csv(RESULTS / "ratings/film_models.csv")
FNAMES = {"cast_women": "women's share of the credited cast (0→100%)",
          "top3_women": "women's share of the top-3 roles (0→100%)",
          "first_woman": "a woman in the first-listed role",
          "writers_women": "women's share of the writers (0→100%)",
          "director_women": "a woman director",
          "summary_women_active": "women's share of who acts in the summary (0→100%)",
          "summary_women_looks": "summary describes a woman by her looks",
          "summary_women_victim": "summary has a woman as a victim of violence",
          "group_stereotype": "summary pairs a group with its stereotype words",
          "offensive_stereotype": "same, pejorative stereotype words only",
          "prejudice_story": "a story about prejudice (study 7 classifier, top 2%)",
          "stereotyped_portrayal": "a stereotyped portrayal (study 9 classifier, 0→1)",
          "social_named": "summary names one of study 11's six groups",
          "social_stereotype": "names one of them with its stereotype words",
          "social_pejorative": "names one of them with pejorative stereotype words",
          "social_label": "calls one of them by an insulting label",
          "social_presence": "one of them appears (study 11 classifier, 0→1)",
          "social_portrayal": "one of them stereotyped, presence held fixed (study 11 classifier, 0→1)"}
FDESIGNS = {"same director": "a director's films compared with each other",
            "same franchise": "films of one franchise compared with each other",
            "same genre and language": "films of the same genre, decade and language"}
order = [m for m in FNAMES if m in set(fm.measure)][::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
for ax, (outcome, title) in zip(axes, [("rating", "change in IMDb rating (points out of 10)"),
                                       ("log_votes", "change in number of votes (%)")]):
    for design, colour, off in zip(FDESIGNS, SLOTS, [0.24, 0, -0.24]):
        d = fm[(fm.outcome == outcome) & (fm.design == design)].set_index("measure").reindex(order)
        est, lo, hi = d.estimate, d.lo, d.hi
        if outcome == "log_votes":
            est, lo, hi = (np.exp(est) - 1) * 100, (np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100
        yv = np.arange(len(order)) + off
        ax.errorbar(est, yv, xerr=[est - lo, hi - est], fmt="o", ms=4, color=colour, ecolor=RULE, capsize=2,
                    label=FDESIGNS[design])
    ax.axvline(0, color=RULE, lw=1)
    ax.set_title(title, fontsize=10, color=INK)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([FNAMES[m] for m in order], fontsize=8)  # slurs are too rare in film summaries to chart
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.suptitle("Feature films, 1950–2026: what goes with higher or lower ratings and more votes (95% intervals)",
             fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(RESULTS / "ratings/film_effects.png")
plt.close(fig)
print("film ratings figure written")

# ---- study 11: six social groups ----
SG = RESULTS / "groups_social"
SNAMES = {"immigrant_refugee": "Immigrants and refugees", "disability": "Disabled people",
          "mental_illness": "People with a mental illness", "older_people": "Older people",
          "poverty_class": "Poor and working-class people", "homelessness": "Homeless people"}
sprev = pd.read_csv(RESULTS / "labels_social/prevalence.csv")
sprev = sprev[sprev.labels_used == "all labels"]
show = [("any_group", "one of the six groups appears"), ("any_stereotyped", "portrayed through a stereotype"),
        ("discrimination", "a story about discrimination against them"), ("pejorative_label", "an insulting label")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (medium, title) in zip(axes, [("film", "Films"), ("tv", "TV")]):
    ends = []
    for (label, name), colour in zip(show, SLOTS):
        d = sprev[(sprev.medium == medium) & (sprev.label == label)].set_index("era").reindex(ERAS)
        x = np.arange(len(ERAS))
        ax.errorbar(x, d.share * 100, yerr=[(d.share - d.lo) * 100, (d.hi - d.share) * 100], color=colour, lw=2,
                    marker="o", ms=6, capsize=3, label=name)
    ax.set_xticks(range(len(ERAS)))
    ax.set_xticklabels(ERAS)
    ax.set_xlim(-0.3, len(ERAS) - 0.7)
    ax.set_title(title, fontsize=10, color=INK)
axes[0].set_ylabel("% of summaries (random sample, 95% intervals)", color=MUTED)
# one legend below both panels: on TV several series end at 0%, where labels at the line ends would collide
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Immigrants, disability, mental illness, age, class and homelessness in plot summaries, read by a "
             "labelling model (1,400 random summaries)", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(RESULTS / "labels_social/prevalence.png")
plt.close(fig)

# naming and stereotype pairing by decade, films and TV apart (recent decades are mostly short TV overviews), one
# panel per group; cells with too few summaries or mentions are left out
sl = pd.read_csv(SG / "group_lift_by_decade_medium.csv")
sl = sl[sl.decade.between(1920, 2020)].sort_values("decade")
fig, axes = plt.subplots(2, 6, figsize=(16, 6.4), sharex=True)
for col, (grp, name) in enumerate(SNAMES.items()):
    for medium, colour, label in [("film", SLOTS[0], "films"), ("tv", SLOTS[1], "TV episodes")]:
        d = sl[(sl.group == grp) & (sl.medium == medium)]
        d = d[(d.n_in + d.n_out) >= 2000]
        axes[0, col].plot(d.decade, d.share_mentioning * 1000, color=colour, lw=2, marker="o", ms=3, label=label)
        d = d[d.n_in >= 30]
        axes[1, col].fill_between(d.decade, d.lift_adj_lo, d.lift_adj_hi, color=colour, alpha=0.15, lw=0)
        axes[1, col].plot(d.decade, d.lift_length_adjusted, color=colour, lw=2, marker="o", ms=3, label=label)
    axes[0, col].set_ylim(0, None)
    axes[0, col].set_title(name, fontsize=9, color=INK)
    axes[1, col].axhline(1, color=RULE, lw=1)
    axes[1, col].set_ylim(0, None)
    axes[1, col].set_xticks([1920, 1970, 2020])
    axes[1, col].set_xticklabels(["1920s", "1970s", "2020s"], fontsize=8)
axes[0, 0].set_ylabel("summaries naming the group\n(per 1,000)", color=MUTED)
axes[1, 0].set_ylabel("stereotype words next to a mention\n(× as often, same summary length)", color=MUTED)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("Six groups across a century of plot summaries: how often they are named, and how tightly stereotype "
             "words cling to them", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(SG / "by_decade.png")
plt.close(fig)
print("social-group figures written")

# ---- study 11, step 2: the classifier trained on the social labels, on study 9's fresh sample ----
LC = RESULTS / "label_classifier_social"
cs = pd.read_csv(LC / "sample_prevalence_by_decade.csv")
SHOW = {"any_group": "one of the six groups appears", "any_stereotyped": "one of them portrayed through a stereotype"}


def decade_axis(ax, d):
    ax.set_xticks(range(int(d.decade.min()), 2030, 20))
    ax.set_xticklabels([f"{x}s" for x in range(int(d.decade.min()), 2030, 20)], fontsize=8)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (medium, title) in zip(axes, [("film", "Films"), ("tv", "TV episodes")]):
    d = cs[(cs.medium == medium) & (cs.summaries >= 500)].sort_values("decade")
    for (target, name), colour in zip(SHOW.items(), SLOTS):
        ax.plot(d.decade, d[target] * 100, color=colour, lw=2, marker="o", ms=5, label=name)
    ax.set_ylim(0, None)
    decade_axis(ax, d)
    ax.set_title(title, fontsize=10, color=INK)
axes[0].set_ylabel("% of summaries (calibrated classifier)", color=MUTED)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle(f"The six groups of study 11 in a fresh sample of {int(cs.summaries.sum()):,} summaries, read by a "
             "classifier trained on the labels", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(LC / "sample_trends.png")
plt.close(fig)

fig, axes = plt.subplots(1, 6, figsize=(16, 3.8), sharex=True)
for ax, (grp, name) in zip(axes, SNAMES.items()):
    for medium, colour, label in [("film", SLOTS[0], "films"), ("tv", SLOTS[1], "TV episodes")]:
        d = cs[(cs.medium == medium) & (cs.summaries >= 500)].sort_values("decade")
        ax.plot(d.decade, d[f"m_{grp}"] * 100, color=colour, lw=2, marker="o", ms=3, label=label)
    ax.set_ylim(0, None)
    decade_axis(ax, cs[cs.summaries >= 500])
    ax.set_title(name, fontsize=9, color=INK)
axes[0].set_ylabel("% of summaries where\nthe group appears", color=MUTED)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("How often each group appears in plot summaries, named or not (calibrated classifier)", fontsize=11,
             color=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(LC / "groups_by_decade.png")
plt.close(fig)

cg = pd.read_csv(LC / "sample_prevalence_by_genre.csv").sort_values("any_stereotyped")
fig, ax = plt.subplots(figsize=(9, 6.5))
y = np.arange(len(cg))
ax.barh(y, cg.any_stereotyped * 100, color=SLOTS[0], height=0.65)
for yi, total in zip(y, cg.any_stereotyped * 100):
    ax.annotate(f"{total:.1f}%", (total + 0.1, yi), va="center", fontsize=8, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels(cg.genre, fontsize=8)
ax.set_xlabel("% of summaries portraying one of the six groups through a stereotype (calibrated classifier, "
              "weighted to the corpus)", color=MUTED, fontsize=8)
ax.set_title("Stereotypes of immigrants, disability, mental illness, age, class and homelessness by genre",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(LC / "sample_by_genre.png")
plt.close(fig)
print("social classifier figures written")

# ---- study 12: money, critics and awards ----
RC = RESULTS / "reception"
cva = pd.read_csv(RC / "critics_vs_audiences.csv")
RNAMES = {"rating": "IMDb rating (audience)", "rt_audience": "Rotten Tomatoes audience score",
          "mc_user": "Metacritic user score", "tomatometer": "Tomatometer (critics)",
          "rt_average": "critics' average rating", "metascore": "Metascore (critics)"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for ax, site in zip(axes, ["Rotten Tomatoes", "Metacritic"]):
    d = cva[cva.site == site]
    order = [r for r in RNAMES if r in set(d.rating)]
    for design, colour, off in [("same director", SLOTS[0], 0.15), ("same genre and language", SLOTS[2], -0.15)]:
        e = d[d.design == design].set_index("rating").reindex(order)
        yv = np.arange(len(order))[::-1] + off
        ax.errorbar(e.estimate, yv, xerr=[e.estimate - e.lo, e.hi - e.estimate], fmt="o", ms=5, color=colour,
                    ecolor=RULE, capsize=3, label={"same director": "a director's films compared with each other",
                                                   "same genre and language": "films of the same genre, decade and language"}[design])
    ax.axvline(0, color=RULE, lw=1)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels([RNAMES[r] for r in order], fontsize=9)
    ax.set_title(f"Films with all {site} and IMDb ratings ({int(d.films.max()):,})", fontsize=10, color=INK)
    ax.set_xlabel("change in the rating, in standard deviations (95% intervals)", color=MUTED)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("A summary the classifier reads as a stereotyped portrayal: how critics and audiences rate the film",
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(RC / "critics_vs_audiences.png")
plt.close(fig)

mo = pd.read_csv(RC / "models.csv")
mo = mo[mo.design == "same director"]
MNAMES = {"stereotyped_portrayal": "a stereotyped portrayal (study 9 classifier, 0→1)",
          "social_presence": "one of study 11's six groups appears (0→1)",
          "social_portrayal": "one of them stereotyped, presence held fixed (0→1)",
          "prejudice_story": "a story about prejudice (study 7 classifier, top 2%)",
          "cast_women": "women's share of the credited cast (0→100%)",
          "first_woman": "a woman in the first-listed role"}
order = list(MNAMES)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (outcome, title, scale) in zip(axes, [("log_box_office", "change in worldwide box office at the same budget (%)", "pct"),
                                              ("oscar_nominated", "change in the chance of an Oscar nomination (points)", "pp")]):
    d = mo[mo.outcome == outcome].set_index("measure").reindex(order)
    est, lo, hi = d.estimate, d.lo, d.hi
    est, lo, hi = ((np.exp(est) - 1) * 100, (np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100) if scale == "pct" else \
        (est * 100, lo * 100, hi * 100)
    ax.errorbar(est, range(len(order)), xerr=[est - lo, hi - est], fmt="o", ms=5, color=SLOTS[0], ecolor=RULE, capsize=3)
    ax.axvline(0, color=RULE, lw=1)
    ax.set_title(title, fontsize=10, color=INK)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([MNAMES[m] for m in order], fontsize=8)
fig.suptitle("A director's own films compared with each other (genres and release year held fixed; 95% intervals)",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(RC / "money_and_oscars.png")
plt.close(fig)
print("study 12 figures written")

# ---- study 13: who makes stereotyped films, and where ----
MK = RESULTS / "makers"
who = pd.read_csv(MK / "who.csv")
ROWS13 = [("any_stereotype", "any stereotyped portrayal"),
          ("gender_any", "a gender stereotype (women's share of the cast held fixed)"),
          ("group_stereotyped", "a group stereotype"),
          ("social_any_stereotyped", "one of study 11's six groups stereotyped")]
SERIES13 = [("film", "director_women", "same genre and language", "film: a woman director (0→1)", SLOTS[0]),
            ("film", "writers_women", "same genre and language", "film: women's share of the writers (0→100%)", SLOTS[1]),
            ("film", "writers_women", "same director", "film: women writers, a director's own films", SLOTS[2]),
            ("series", "creators_women_share", "same genre and language",
             "series: women's share of the creators (0→100%)", SLOTS[3])]
fig, ax = plt.subplots(figsize=(10, 5))
for k, (medium, measure, base, label, colour) in enumerate(SERIES13):
    rows13 = []
    for outcome, _ in ROWS13:
        design = base + (", cast held fixed" if outcome == "gender_any" else "")
        r = who[(who.medium == medium) & (who.outcome == outcome) & (who.measure == measure) & (who.design == design)]
        rows13.append((r.estimate.iloc[0], r.lo.iloc[0], r.hi.iloc[0]) if len(r) else (np.nan, np.nan, np.nan))
    if np.isnan([e for e, _, _ in rows13]).all():
        continue
    est, lo, hi = (np.array(v) * 100 for v in zip(*rows13))
    yv = np.arange(len(ROWS13))[::-1] + 0.27 - 0.18 * k
    ax.errorbar(est, yv, xerr=[est - lo, hi - est], fmt="o", ms=5, color=colour, ecolor=RULE, capsize=3, label=label)
ax.axvline(0, color=RULE, lw=1)
ax.set_yticks(np.arange(len(ROWS13))[::-1])
ax.set_yticklabels([r[1] for r in ROWS13], fontsize=9)
ax.set_xlabel("difference in the probability of the portrayal, percentage points (95% intervals)\n"
              "against titles of the same genre, decade and language unless noted", color=MUTED, fontsize=8)
ax.set_title("Women behind the camera and stereotyped portrayals", fontsize=10, color=INK)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(MK / "who.png")
plt.close(fig)

where = pd.read_csv(MK / "where.csv")
NAMES = {"US": "United States", "GB": "United Kingdom", "FR": "France", "DE": "Germany", "IT": "Italy", "JP": "Japan",
         "IN": "India", "ES": "Spain", "CA": "Canada", "SU": "Soviet Union", "KR": "South Korea", "CN": "China",
         "HK": "Hong Kong", "TR": "Turkey", "MX": "Mexico", "BR": "Brazil", "PH": "Philippines", "RU": "Russia",
         "IR": "Iran", "GR": "Greece", "SE": "Sweden", "AU": "Australia", "AR": "Argentina", "DK": "Denmark"}
fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1.3, 1, 1.2]})
d = where[(where.medium == "film") & (where.grouping == "country_top") & (where.outcome == "any_stereotype")].sort_values("rate")
axes[0].barh(range(len(d)), d.rate * 100, color=SLOTS[0], height=0.65)
axes[0].set_yticks(range(len(d)))
axes[0].set_yticklabels([NAMES.get(v, v) for v in d.value], fontsize=8)
axes[0].set_title("Films by country of origin", fontsize=10, color=INK)
b = where[(where.medium == "film") & (where.grouping == "budget_band") & (where.outcome == "any_stereotype")]
order = ["under $1m", "$1–10m", "$10–50m", "$50–100m", "over $100m"]
b = b.set_index("value").reindex(order)
axes[1].barh(range(len(b)), b.rate * 100, xerr=[(b.rate - b.lo) * 100, (b.hi - b.rate) * 100], color=SLOTS[0],
             ecolor=RULE, height=0.6)
axes[1].set_yticks(range(len(b)))
axes[1].set_yticklabels([f"{v} ({int(n):,})" for v, n in zip(b.index, b.titles)], fontsize=8)
axes[1].set_title("Films by budget (2025 dollars)", fontsize=10, color=INK)
n = where[(where.medium == "series") & (where.grouping == "network_top") & (where.outcome == "any_stereotype")]
n = n.sort_values("rate")
axes[2].barh(range(len(n)), n.rate * 100, xerr=[(n.rate - n.lo) * 100, (n.hi - n.rate) * 100], color=SLOTS[3],
             ecolor=RULE, height=0.65)
axes[2].set_yticks(range(len(n)))
axes[2].set_yticklabels([f"{v} ({int(t):,})" for v, t in zip(n.value, n.titles)], fontsize=8)
axes[2].set_title("Series by network (first seasons)", fontsize=10, color=INK)
fig.supxlabel("% of summaries portraying a stereotype (genre, decade and summary length held fixed)", color=MUTED,
              fontsize=9)
fig.suptitle("Where stereotyped portrayals are made", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(MK / "where.png")
plt.close(fig)
print("study 13 figures written")
