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
