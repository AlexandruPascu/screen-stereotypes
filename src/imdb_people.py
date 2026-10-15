"""Gender for everyone in IMDb's credits, including writers, directors and other crew (IMDb records no gender).

1. Performers: gender from IMDb's actor/actress credits (a person credited only as one of the two).
2. Everyone else: from the first name, using a map learned from those performers (first names with at least MIN_PEOPLE
   performers). Each person gets p_female, the share of performers with that first name who are women: averaging it
   gives an unbiased share of women in a group, where hard labels would not (men outnumber women among performers, so
   the few women with a mostly-male name add up). Hard labels from names misclassify women even at 99% (checked
   below), so name-based gender is used only as a probability. Titles, initials and single-word names are not used.
Checks: (a) held-out performers: the map is learned on 80% of performers and tested on the other 20%;
        (b) TMDb's own gender field for writers and directors, matched by full name where the name is unique on both.
Output: DERIVED/people.parquet (nconst, name, birth year, gender, how it was decided). Report: results/people/.
"""
import json
import re
import sqlite3
import unicodedata
import zlib

import duckdb
import numpy as np
import pandas as pd

from paths import DERIVED, IMDB, RESULTS, TMDB_DB

OUT = RESULTS / "people"
OUT.mkdir(parents=True, exist_ok=True)
MIN_PEOPLE, HARD = 20, 0.99
TITLES = {"dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "sir", "dame", "lord", "lady", "rev", "rev.",
          "prof", "prof.", "st", "st.", "the", "von", "van", "de", "del", "la", "le", "el", "di", "da"}
con = duckdb.connect()
con.execute("SET memory_limit = '7GB'")


def tsv(name, columns):
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return f"read_csv('{IMDB / name}', delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{{cols}}})"


def first_name(name):
    """Lower-case first given name: skips titles and initials; None for single-word names."""
    if not isinstance(name, str):
        return None
    t = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    words = [w for w in re.split(r"[\s\-]+", t) if w]
    if len(words) < 2:
        return None
    for w in words[:-1]:
        w = w.strip(".,'\"()")
        if w in TITLES or len(w) <= 1 or re.fullmatch(r"[a-z]\.?", w):
            continue
        return w if w.isalpha() else None
    return None


people = con.execute(f"""
    WITH perf AS (SELECT nconst, SUM((category = 'actress')::INT) AS f, SUM((category = 'actor')::INT) AS m
                  FROM {tsv('title.principals.tsv.gz', ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters'])}
                  WHERE category IN ('actor', 'actress') GROUP BY nconst)
    SELECT n.nconst, n.primaryName AS name, TRY_CAST(n.birthYear AS INT) AS birth_year, n.primaryProfession AS professions,
           p.f, p.m
    FROM {tsv('name.basics.tsv.gz', ['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession',
                                     'knownForTitles'])} n
    LEFT JOIN perf p USING (nconst)""").df()
people["first"] = people.name.map(first_name)
known = people.f.notna() & ((people.f == 0) | (people.m == 0))
people["performer_gender"] = np.where(known, np.where(people.f > 0, "female", "male"), None)
print(f"people {len(people):,}; performers with one credited gender {known.sum():,} "
      f"(both actor and actress credits: {(people.f.notna() & ~known).sum():,})", flush=True)


def name_map(perf):
    g = perf.groupby("first").performer_gender.agg(n="size", p_female=lambda x: (x == "female").mean())
    return g[g.n >= MIN_PEOPLE]


perf = people[known & people["first"].notna()][["nconst", "first", "performer_gender"]]
rng = np.random.default_rng(0)
test = rng.random(len(perf)) < 0.2
t = perf[test].merge(name_map(perf[~test])[["p_female"]], left_on="first", right_index=True, how="left")
covered = t.p_female.notna()
tc = t[covered].assign(is_f=lambda d: d.performer_gender == "female")
share_err = tc.p_female.mean() - tc.is_f.mean()
hard = tc[(tc.p_female >= HARD) | (tc.p_female <= 1 - HARD)].assign(pred_f=lambda d: d.p_female >= HARD)
hard_acc = hard.assign(ok=lambda d: d.pred_f == d.is_f).groupby("performer_gender").ok.mean()
print(f"held-out performers: name known for {covered.mean():.1%}; women's share predicted {tc.p_female.mean():.4f} vs "
      f"actual {tc.is_f.mean():.4f} (error {share_err:+.4f}); hard labels for {len(hard) / len(tc):.1%}, correct for "
      f"{hard_acc.get('female', np.nan):.2%} of women and {hard_acc.get('male', np.nan):.2%} of men", flush=True)

nm = name_map(perf)
people = people.merge(nm[["p_female"]].rename(columns={"p_female": "name_p"}), left_on="first", right_index=True,
                      how="left")
people["p_female"] = np.where(people.performer_gender.notna(), (people.performer_gender == "female").astype(float),
                              people.name_p)
# Hard labels from names are kept only for this check (they misclassify women: see the report); the output's `gender`
# is exact (performer credits) or missing, and all name-based work uses p_female.
people["gender"] = np.select([people.p_female >= HARD, people.p_female <= 1 - HARD], ["female", "male"], None)
people["gender_source"] = np.select([people.performer_gender.notna(), people.name_p.notna()],
                                    ["performer credits", "first name"], "unknown")

# Check against TMDb's gender field for writers and directors (unique names on both sides)
db = sqlite3.connect(f"file:{TMDB_DB}?mode=ro", uri=True)
crew = {}
for path, body in db.execute("SELECT path, body FROM responses WHERE path LIKE '/tv/%/season/%' AND status = 200"):
    for e in json.loads(zlib.decompress(body)).get("episodes") or []:
        for c in e.get("crew") or []:
            if c.get("job") in ("Writer", "Director", "Teleplay", "Story", "Screenplay") and c.get("gender") in (1, 2):
                crew.setdefault(c.get("name"), set()).add(c["gender"])
tm = pd.DataFrame([(k, "female" if v == {1} else "male") for k, v in crew.items() if len(v) == 1 and k],
                  columns=["name", "tmdb_gender"])
unique = people[~people.name.duplicated(keep=False)]
crew_people = unique[unique.professions.fillna("").str.contains("writer|director") & unique.performer_gender.isna()]
v = crew_people.merge(tm, on="name")
vc = v[v.name_p.notna()].assign(is_f=lambda d: d.tmdb_gender == "female")
tm_share = (vc.name_p.mean(), vc.is_f.mean())
vh = vc[vc.gender.notna()].assign(ok=lambda d: (d.gender == "female") == d.is_f)
tm_acc = vh.groupby("tmdb_gender").ok.mean()
print(f"TMDb check: {len(v):,} writers/directors matched by unique name; name known for {len(vc) / max(1, len(v)):.1%}; "
      f"women's share predicted {tm_share[0]:.4f} vs TMDb {tm_share[1]:.4f}; hard labels correct for "
      f"{tm_acc.get('female', np.nan):.2%} of women and {tm_acc.get('male', np.nan):.2%} of men", flush=True)

people["gender"] = people.performer_gender
people[["nconst", "name", "birth_year", "professions", "p_female", "gender", "gender_source"]].to_parquet(
    DERIVED / "people.parquet", index=False)
summary = pd.DataFrame([
    ("people in IMDb", len(people)),
    ("gender from performer credits", int((people.gender_source == "performer credits").sum())),
    ("gender probability from first name", int((people.gender_source == "first name").sum())),
    ("unknown", int((people.gender_source == "unknown").sum())),
    ("first names in the map", len(nm)),
    ("held-out performers: name known", round(covered.mean(), 4)),
    ("held-out performers: women's share predicted", round(tc.p_female.mean(), 4)),
    ("held-out performers: women's share actual", round(tc.is_f.mean(), 4)),
    ("held-out performers: hard labels correct, women", round(hard_acc.get("female", np.nan), 4)),
    ("held-out performers: hard labels correct, men", round(hard_acc.get("male", np.nan), 4)),
    ("TMDb writers/directors matched", len(v)),
    ("TMDb check: women's share predicted", round(tm_share[0], 4)),
    ("TMDb check: women's share in TMDb", round(tm_share[1], 4)),
    ("TMDb check: hard labels correct, women", round(tm_acc.get("female", np.nan), 4)),
    ("TMDb check: hard labels correct, men", round(tm_acc.get("male", np.nan), 4))], columns=["item", "value"])
summary.to_csv(OUT / "gender_inference.csv", index=False)
print(summary.to_string(index=False))
