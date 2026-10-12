"""Who plays whom: women's share of on-screen occupations, by decade, medium and country.

The comparison that matters is not the raw share but the share against women's share of all named roles in the same
slice (the baseline): a job is gender-typed on screen when it is cast far from that baseline.
Skew = log odds ratio of a woman being cast in the job vs any named role (0 = baseline; +1 ≈ 2.7x the odds).

Main scope: English-language productions (TMDb original language), because the job words are English. In gendered
languages a woman's job is usually credited in a feminine form ("doctora", "Kommissarin") that an English list misses,
which would make those countries look more male than they are. "All titles" is kept as a robustness check.

Writes aggregate tables (no character names) and figures to results/occupations/.
"""
import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from occupations import GENDERED, GROUP, MODIFIERS, alternatives, occupation
from paths import DERIVED, RESULTS

OUT = RESULTS / "occupations"
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit = '7GB'")

NOT_CHARACTERS = ["self", "himself", "herself", "themselves", "themself", "narrator", "host", "co-host", "various",
                  "various characters", "additional voices", "additional voice", "voice", "voices", "extra", "guest",
                  "presenter", "announcer", "commentator", "contestant", "performer", "uncredited", "cameo",
                  "audience member", "background", "featured", "ensemble"]
ERAS = [(1920, 1969, "1920-69"), (1970, 1989, "1970-89"), (1990, 2009, "1990-2009"), (2010, 2026, "2010-26")]
ERA_SQL = "CASE " + " ".join(f"WHEN r.year <= {hi} THEN '{label}'" for _, hi, label in ERAS) + " END"
ENGLISH_COUNTRIES = ["US", "GB", "CA", "AU", "IE", "NZ"]

# Fiction only: drop works whose genres mark them as non-fiction, and names that are not characters.
con.execute(f"""CREATE TEMP VIEW roles AS
    SELECT r.*, CASE WHEN r.year < 1950 THEN 1940 ELSE (r.year // 10) * 10 END AS decade, {ERA_SQL} AS era,
           t.language, t.country, t.language = 'en' AS english
    FROM '{DERIVED / "roles.parquet"}' r LEFT JOIN '{DERIVED / "tmdb_titles.parquet"}' t ON t.imdb = r.work
    WHERE r.year BETWEEN 1920 AND 2026 AND r.medium IN ('tv', 'film')
      AND NOT regexp_matches(COALESCE(r.genres, ''), 'Reality-TV|Talk-Show|Game-Show|News|Documentary')
      AND r.name NOT IN ({", ".join(f"'{n}'" for n in NOT_CHARACTERS)})""")

# Baseline counts for every slice we use, in one pass.
base = con.execute("""SELECT medium, decade, era, COALESCE(english, false) AS english, COALESCE(country, '?') AS country,
    COUNT(*) AS roles, SUM(female::INT) AS women FROM roles GROUP BY ALL""").df()
base.to_csv(OUT / "baseline_named_roles.csv", index=False)

# Classify each distinct character name once (prefilter in DuckDB, exact rules in Python).
names = con.execute(f"""SELECT DISTINCT name FROM roles
    WHERE regexp_matches(name, '\\b({alternatives()}|dr\\.?|{"|".join(GENDERED)})\\b')""").df()
names["occupation"] = names.name.map(occupation)
gendered = names.name.str.extract(rf"\b({'|'.join(sorted(GENDERED, key=len, reverse=True))})\b", expand=False)
# same surname rule as the job words: "Rhett Butler" or "Mrs. Butler" is not a butler
first = names.name.str.extract(r"^([a-z'.\-]+) [a-z]+$", expand=False)
surname_like = gendered.notna() & (names.name.str.split().str[-1] == gendered) & first.notna() & ~first.isin(MODIFIERS)
names["gendered_word"] = gendered.where(~surname_like)
names = names.dropna(subset=["occupation", "gendered_word"], how="all")
con.register("names", names)
roles = con.execute("""SELECT r.medium, r.decade, r.era, r.year, r.female, COALESCE(r.english, false) AS english,
        COALESCE(r.country, '?') AS country, n.occupation, n.gendered_word
    FROM roles r JOIN names n USING (name)""").df()
occ = roles.dropna(subset=["occupation"])
print(f"named fictional roles: {base.roles.sum():,} (English-language productions {base[base.english].roles.sum():,}); "
      f"with a job: {len(occ):,} (English {occ.english.sum():,}); with a gendered job word: {roles.gendered_word.notna().sum():,}")


def table(o, b, by):
    """Women's share and casting skew per occupation within each slice `by`, against that slice's baseline."""
    t = o.groupby(["occupation"] + by, observed=True).agg(roles=("female", "size"), women=("female", "sum")).reset_index()
    bb = b.groupby(by, observed=True)[["roles", "women"]].sum().reset_index()
    bb["baseline"] = bb.women / bb.roles
    t = t.merge(bb[by + ["baseline"]], on=by)
    t["women_share"] = t.women / t.roles
    w, m = t.women + 0.5, t.roles - t.women + 0.5  # 0.5 avoids zero cells
    t["skew"] = np.log(w / m) - np.log(t.baseline / (1 - t.baseline))
    se = np.sqrt(1 / w + 1 / m)
    t["skew_lo"], t["skew_hi"] = t["skew"] - 1.96 * se, t["skew"] + 1.96 * se
    t["group"] = t.occupation.map(GROUP)
    return t


en_occ, en_base = occ[occ.english], base[base.english]
by_era = table(en_occ, en_base, ["era"])
by_era.to_csv(OUT / "occupation_by_era_english.csv", index=False)
table(occ, base, ["era"]).to_csv(OUT / "occupation_by_era_all_titles.csv", index=False)
table(en_occ, en_base, ["medium", "decade"]).to_csv(OUT / "occupation_by_decade_english.csv", index=False)
recent = occ[occ.era == "2010-26"].assign(place=lambda d: d.country.where(d.country.isin(ENGLISH_COUNTRIES), "other"))
rb = base[base.era == "2010-26"].assign(place=lambda d: d.country.where(d.country.isin(ENGLISH_COUNTRIES), "other"))
table(recent[recent.english], rb[rb.english], ["place"]).to_csv(OUT / "occupation_by_country_2010s_english.csv", index=False)

gw = roles.dropna(subset=["gendered_word"]).groupby(["gendered_word", "era"], observed=True).agg(
    roles=("female", "size"), women=("female", "sum")).reset_index()
gw["word_gender"] = gw.gendered_word.map(GENDERED)
gw.to_csv(OUT / "gendered_job_words.csv", index=False)

# Figure 1: casting skew per job, English-language productions 2010-26 vs 1920-69
recent_h = by_era[(by_era.era == "2010-26") & (by_era.roles >= 200) & (by_era.group != "ambiguous")].sort_values("skew")
old = by_era[by_era.era == "1920-69"].set_index("occupation").reindex(recent_h.occupation)
fig, ax = plt.subplots(figsize=(8, 12))
y = np.arange(len(recent_h))
ax.errorbar(recent_h["skew"], y, xerr=[recent_h["skew"] - recent_h["skew_lo"], recent_h["skew_hi"] - recent_h["skew"]],
            fmt="o", color="#333", ecolor="#aaa", ms=4, label="2010–26")
ax.scatter(old["skew"], y, marker="|", color="#c0392b", s=90, label="1920–69")
ax.axvline(0, color="#888", lw=1)
ax.set_yticks(y)
ax.set_yticklabels([f"{o}  ({s:.0%} women, n={n:,})" for o, s, n in zip(recent_h.occupation, recent_h.women_share,
                                                                       recent_h.roles)], fontsize=8)
ax.set_xlabel("casting skew: log odds of a woman vs all named roles (0 = no skew)")
ax.set_title("Who plays whom: English-language film and TV")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "skew_by_job_english.png", dpi=150)

# Figure 2: women's share by decade for selected jobs, with the all-roles baseline
show = ["nurse", "secretary", "reporter", "teacher", "judge", "lawyer", "doctor", "scientist", "detective",
        "professor", "police officer", "soldier"]
pooled = en_occ.groupby(["occupation", "decade"]).agg(roles=("female", "size"), share=("female", "mean")).reset_index()
bl = en_base.groupby("decade")[["roles", "women"]].sum()
fig, ax = plt.subplots(figsize=(10, 6))
for o in show:
    d = pooled[(pooled.occupation == o) & (pooled.roles >= 80)]
    ax.plot(d.decade, d.share, marker="o", ms=3, label=o)
ax.plot(bl.index, bl.women / bl.roles, color="black", lw=3, label="all named roles")
ax.set_ylabel("share of roles played by women")
ax.set_ylim(0, 1)
ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.set_title("Women's share of on-screen jobs by decade (English-language film + TV, IMDb credits)")
fig.tight_layout()
fig.savefig(OUT / "women_share_by_decade_english.png", dpi=150)

pd.set_option("display.width", 220)
print(bl.assign(share=lambda d: (d.women / d.roles).round(3)).share.to_string())
print(by_era.pivot(index="occupation", columns="era", values="women_share").round(2).join(
    by_era[by_era.era == "2010-26"].set_index("occupation")[["roles", "skew"]].round(2)).sort_values("skew").to_string())
