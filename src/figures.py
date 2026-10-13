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
    ax.set_yticklabels(d.lemma.replace({"rap": "rape"}))
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
          "slur": "summary contains a slur"}
order = list(labels)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
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
fig.suptitle("Episodes compared within the same series: effects are tiny (95% intervals)", fontsize=12)
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
g = g[g.decade.between(1950, 2020) & g.stereotype_lift.notna()]
names = {"native_american": "Native American", "german": "German", "italian": "Italian", "russian": "Russian",
         "muslim": "Muslim", "arab": "Arab", "mexican": "Mexican", "chinese": "Chinese", "black": "Black"}
fig, ax = plt.subplots(figsize=(10, 5.5))
for grp, label in names.items():
    d = g[(g.group == grp) & (g.n_in >= 100)].sort_values("decade")
    ax.plot(d.decade, d.stereotype_lift, marker="o", ms=3, label=label)
pooled = g[g.n_in >= 50].groupby("decade").apply(lambda x: np.exp(np.average(np.log(x.stereotype_lift), weights=x.n_in)),
                                                  include_groups=False)
ax.plot(pooled.index, pooled.values, color="black", lw=3, label="all groups (weighted)")
ax.axhline(1, color=GREY, lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10])
ax.set_yticklabels(["1×", "2×", "5×", "10×"])
ax.set_ylabel("how much more often pejorative stereotype words appear\nwhen the summary mentions the group")
ax.set_title("Pejorative stereotype words next to group mentions in plot summaries")
ax.legend(ncol=2, fontsize=8)
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
