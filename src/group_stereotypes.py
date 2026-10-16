"""Stereotypes and slurs about groups in plot summaries and character names.

  1. Mentions: which groups a summary mentions (curated terms in groups.py).
  2. Stereotype lift: how much more often a group's stereotype words appear in summaries that mention the group than in
     summaries that do not (lift 1 = no association). Per group, per decade, and per stereotype word.
  3. Slurs: share of summaries, and of credited character names, containing a slur against each group.
  4. Group-labelled characters ("Mexican Bandit", "Chinese Waiter"): which jobs they are given, against all characters.
  5. Validation: titles that viewers tagged "racism", "<group> stereotype", "antisemitism"... should score higher.

Aggregates only (no text) go to results/groups/. Words are matched as whole words or 2-3 word phrases.
GROUP_SET=social scans the social groups of social_groups.py (immigrants and refugees, disability, mental illness,
older people, poverty and class, homelessness) into files of their own (results/groups_social/...); there the
pejorative labels take the place of slurs, and mentions, being common nouns, are matched in lower case.
"""
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

import duckdb
import numpy as np
import pandas as pd

from groups import GROUPS as GROUP_DEFS, NOT_A_MENTION, cased_variants
from social_groups import AGE_TOKEN, NOT_A_MENTION as SOCIAL_NOT_A_MENTION
from occupations import GROUP as OCC_GROUP, occupation
from publish import without_slurs
from paths import DERIVED, IMDB, LEXICONS, RESULTS

# INDEX=offensive uses only the pejorative stereotype words (SeeGULL offensiveness >= 1, HurtLex, pejorative list)
INDEX = os.environ.get("INDEX", "all")
GROUP_SET = os.environ.get("GROUP_SET", "core")
SOCIAL = GROUP_SET == "social"
SUFFIX = "_social" if SOCIAL else ""
OUT = RESULTS / (("groups" if INDEX == "all" else "groups_offensive") + SUFFIX)
OUT.mkdir(parents=True, exist_ok=True)
lex = json.load(open(DERIVED / f"lexicon{SUFFIX}.json"))
GROUPS = list(lex)
MENTION = {t: g for g, v in lex.items() for t in v["mentions"]}
PROPER_CATEGORIES = {"nationality", "ethnicity", "religion"}
CASED = {}
for g, d in ({} if SOCIAL else GROUP_DEFS).items():
    for t in d[1]:
        for v in cased_variants(t, d[0] in PROPER_CATEGORIES):
            CASED[v] = g
if SOCIAL:
    CASED = dict(MENTION)  # common nouns: matched in lower case
CASED_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
ATTR = {g: set(v["attributes"] if INDEX == "all" else v["offensive_attributes"]) - set(v["mentions"])
        for g, v in lex.items()}
SLUR = {}
for g, v in lex.items():
    for t in v["slurs"]:
        SLUR.setdefault(t, set()).add(g)
# Everyday idioms that contain a slur ("spick and span") are removed before matching; the phrases live in the private
# exclusions file next to the source lexicons.
_ex = LEXICONS / "slur_exclusions.json"
SLUR_PHRASES = json.load(open(_ex))["remove_phrases"] if _ex.exists() else []
NOT_RE = re.compile(r"\b(" + "|".join(SOCIAL_NOT_A_MENTION if SOCIAL else NOT_A_MENTION + SLUR_PHRASES) + r")\b",
                    re.I)
# "a 72-year-old widow": numeric ages of 65 and over become one word that the older-people mentions include
AGE_RE = re.compile(r"\b(?:6[5-9]|[7-9][0-9]|1[01][0-9])[- ]years?[- ]old\b", re.I)
WORD = re.compile(r"[a-z][a-z'\-]*")


def clean(text):
    text = NOT_RE.sub(" ", text)
    return AGE_RE.sub(f" {AGE_TOKEN} ", text) if SOCIAL else text


def cased_grams(text):
    """Mention matching keeps the case of the text, so "Polish" counts and "polish the silver" does not."""
    words = CASED_WORD.findall(clean(text))
    return set(words) | {" ".join(words[i:i + 2]) for i in range(len(words) - 1)} | \
        {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def grams(text):
    words = WORD.findall(clean(text.lower()))
    return set(words) | {" ".join(words[i:i + 2]) for i in range(len(words) - 1)} | \
        {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


BIT = {g: 1 << i for i, g in enumerate(GROUPS)}


def scan(text):
    """Bitmasks of groups mentioned, groups whose stereotype words appear, groups targeted by a slur; and the
    (group, word) stereotype hits."""
    g = grams(text)
    mentioned = 0
    for t in (g if SOCIAL else cased_grams(text)) & CASED.keys():
        mentioned |= BIT[CASED[t]]
    attr_mask, hits = 0, []
    for grp in GROUPS:
        found = ATTR[grp] & g
        if found:
            attr_mask |= BIT[grp]
            hits += [(grp, w) for w in found]
    slur_mask = 0
    for t in g & SLUR.keys():
        for grp in SLUR[t]:
            slur_mask |= BIT[grp]
    return mentioned, attr_mask, slur_mask, hits


def run(batch):
    rows, all_hits, in_hits = [], Counter(), Counter()
    for text in batch:
        m, a, s, hits = scan(text)
        rows.append((m, a, s))
        all_hits.update(hits)
        in_hits.update(h for h in hits if m & BIT[h[0]])
    return rows, all_hits, in_hits


def names_scan(names):
    """Groups named and groups targeted by a slur in character names (only non-empty results)."""
    out = {}
    for n in names:
        g = grams(n)
        m = {CASED[t] for t in (g if SOCIAL else cased_grams(n)) & CASED.keys()}
        s = {grp for t in g & SLUR.keys() for grp in SLUR[t]}
        if m or s:
            out[n] = (m, s)
    return out


if __name__ == "__main__":
    con = duckdb.connect()
    con.execute("SET memory_limit = '5GB'")
    texts = con.execute(f"""SELECT title, arg_max(text, words) AS text, any_value(kind) AS kind
        FROM '{DERIVED / "texts.parquet"}' WHERE words >= 5 GROUP BY title""").df()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        texts = texts.sample(limit, random_state=1)
    # year: IMDb titles from basics, TMDb-only episodes from the TMDb episode table
    years = con.execute(f"""SELECT tconst AS title, TRY_CAST(startYear AS INT) AS year, titleType FROM read_csv(
        '{IMDB / "title.basics.tsv.gz"}', delim='\\t', header=true, quote='', escape='', nullstr='\\N',
        columns={{'tconst':'VARCHAR','titleType':'VARCHAR','primaryTitle':'VARCHAR','originalTitle':'VARCHAR',
        'isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR','runtimeMinutes':'VARCHAR','genres':'VARCHAR'}})""").df()
    te = pd.read_parquet(DERIVED / "tmdb_episodes.parquet", columns=["series", "season", "episode", "year"])
    te["title"] = "tmdb:" + te.series.astype(str) + ":" + te.season.astype(str) + ":" + te.episode.astype(str)
    years = pd.concat([years[["title", "year"]], te[["title", "year"]]])
    texts = texts.merge(years, on="title", how="left")
    texts["medium"] = np.where(texts.kind == "film", "film", "tv")
    texts["decade"] = (texts.year // 10 * 10).where(texts.year.between(1920, 2026))
    print(f"summaries: {len(texts):,} (with a year {texts.decade.notna().sum():,})", flush=True)

    batches = [texts.text.iloc[i:i + 20000].tolist() for i in range(0, len(texts), 20000)]
    masks, all_hits, in_hits = [], Counter(), Counter()
    with Pool(8) as pool:
        for rows, a_h, i_h in pool.imap(run, batches):
            masks += rows
            all_hits.update(a_h)
            in_hits.update(i_h)
    del batches
    masks = np.array(masks, dtype=np.int64)
    texts = texts.drop(columns="text")
    # per-title bitmasks for the ratings models (private: keyed by title, no text); bit order = GROUPS
    pd.DataFrame({"title": texts.title, "mention_mask": masks[:, 0], "stereotype_mask": masks[:, 1],
                  "slur_mask": masks[:, 2]}).to_parquet(
        DERIVED / f"title_groups{SUFFIX}{'' if INDEX == 'all' else '_offensive'}.parquet", index=False)
    json.dump(GROUPS, open(DERIVED / f"title_groups_bits{SUFFIX}.json", "w"))

    def has(col, grp):
        return (masks[:, col] & BIT[grp]) > 0

    long = pd.concat([pd.DataFrame({"group": grp, "decade": texts.decade, "medium": texts.medium,
                                    "mention": has(0, grp), "attr": has(1, grp), "slur": has(2, grp)})
                      for grp in GROUPS])

    def lift_table(d, by):
        t = d.groupby(by + ["mention"], dropna=False).agg(n=("attr", "size"), attr=("attr", "sum"),
                                                         slur=("slur", "sum")).reset_index()
        w = t.pivot_table(index=by, columns="mention", values=["n", "attr", "slur"], aggfunc="sum").fillna(0)
        w.columns = [f"{a}_{'in' if b else 'out'}" for a, b in w.columns]
        w = w.reset_index()
        w["share_mentioning"] = w.n_in / (w.n_in + w.n_out)
        p_in = (w.attr_in + 0.5) / (w.n_in + 1)
        p_out = (w.attr_out + 0.5) / (w.n_out + 1)
        w["stereotype_lift"] = p_in / p_out
        se = np.sqrt(1 / (w.attr_in + 0.5) - 1 / (w.n_in + 1) + 1 / (w.attr_out + 0.5) - 1 / (w.n_out + 1))
        w["lift_lo"], w["lift_hi"] = np.exp(np.log(w.stereotype_lift) - 1.96 * se), np.exp(np.log(w.stereotype_lift) + 1.96 * se)
        none = (w.attr_in + w.attr_out) == 0  # no stereotype word for this group ever appears: lift is undefined
        w.loc[none, ["stereotype_lift", "lift_lo", "lift_hi"]] = np.nan
        w["slur_rate_per_100k"] = (w.slur_in + w.slur_out) / (w.n_in + w.n_out) * 1e5
        return w

    overall = lift_table(long, ["group"])
    overall.to_csv(OUT / "group_lift_overall.csv", index=False)
    by_decade = lift_table(long.dropna(subset=["decade"]), ["group", "decade"])
    by_decade.to_csv(OUT / "group_lift_by_decade.csv", index=False)
    lift_table(long, ["group", "medium"]).to_csv(OUT / "group_lift_by_medium.csv", index=False)
    pd.set_option("display.width", 200)
    print(overall[["group", "n_in", "share_mentioning", "attr_in", "stereotype_lift", "lift_lo", "lift_hi",
                   "slur_rate_per_100k"]].sort_values("stereotype_lift", ascending=False).round(4).to_string(index=False))

    # per-word lift: which stereotype words are most tied to each group's mentions
    words = []
    for grp in GROUPS:
        n_in = int(has(0, grp).sum())
        n_out = len(masks) - n_in
        for (g2, w), total in all_hits.items():
            if g2 != grp:
                continue
            c_in = in_hits[(grp, w)]
            c_out = total - c_in
            words.append((grp, w, c_in, c_out, ((c_in + 0.5) / (n_in + 1)) / ((c_out + 0.5) / (n_out + 1)),
                          ",".join(lex[grp]["attributes"].get(w, []))))
    words = pd.DataFrame(words, columns=["group", "word", "texts_with_group", "texts_without_group", "lift", "source"])
    without_slurs(words, "word").to_csv(OUT / "stereotype_word_lift.csv", index=False)

    # validation against viewer tags
    tags = con.execute(f"""SELECT title, string_agg(lower(keywords), '|') AS kw FROM '{DERIVED / "texts.parquet"}'
        WHERE keywords <> '' GROUP BY title""").df()
    tagged = texts[["title"]].merge(tags, on="title", how="left").kw.fillna("")
    checks = []
    for grp in GROUPS:
        m, s = has(0, grp), has(2, grp)
        a = has(1, grp) & m
        stem = grp.replace("_", " ").replace("native american", "native-american|native american|american-indian")
        pat = rf"({stem})[- ]stereotype"
        t = tagged.str.contains(pat, regex=True).to_numpy()
        if t.sum() >= 10:
            checks.append((f"{grp} stereotype tag", t.sum(), m[t].mean(), m[~t].mean(), a[t].mean(), a[~t].mean(),
                           s[t].mean(), s[~t].mean()))
    any_slur = masks[:, 2] > 0
    # (tag, pattern, group whose mentions and labels it should go with; None = any group)
    TAGS = [("racism tag", r"racis", None), ("antisemitism tag", r"antisemit|anti-semit", None),
            ("homophobia tag", r"homophob", None), ("islamophobia tag", r"islamophob", None),
            ("racial/ethnic slur tag", r"racial[- ]slur|ethnic[- ]slur|racist[- ]slur", None)]
    if SOCIAL:
        TAGS = [("immigration tag", r"immigra|refugee|asylum seeker|deportation", "immigrant_refugee"),
                ("disability tag", r"disabilit|wheelchair|blindness|deafness|paraplegi|amputee|autism|down syndrome",
                 "disability"),
                ("mental illness tag", r"mental illness|schizophreni|bipolar|psychiatric|mental hospital|"
                 r"mental patient|insanity", "mental_illness"),
                ("old age tag", r"old age|elderly|retirement home|nursing home|ageism|senior citizen", "older_people"),
                ("poverty tag", r"poverty|working class|slum", "poverty_class"),
                ("homelessness tag", r"homeless", "homelessness")]
    for label, pat, grp in TAGS:
        t = tagged.str.contains(pat, regex=True).to_numpy()
        anym = masks[:, 0] > 0 if grp is None else has(0, grp)
        sl = any_slur if grp is None else has(2, grp)
        checks.append((label, t.sum(), anym[t].mean(), anym[~t].mean(), np.nan, np.nan, sl[t].mean(), sl[~t].mean()))
    val = pd.DataFrame(checks, columns=["tag", "tagged_titles", "mention_if_tagged", "mention_if_not",
                                        "stereotype_word_with_mention_if_tagged", "same_if_not", "slur_if_tagged",
                                        "slur_if_not"])
    val.to_csv(OUT / "validation_against_tags.csv", index=False)
    print(val.round(4).to_string(index=False))

    if INDEX != "all":
        raise SystemExit  # the character-name results do not depend on the stereotype words

    # character names: slurs and group labels in credited roles
    roles = con.execute(f"""SELECT name, character, female, year, medium, genres FROM '{DERIVED / "roles.parquet"}'
        WHERE year BETWEEN 1920 AND 2026 AND medium IN ('tv', 'film')""").df()
    # group labels need the credit as written ("French Maid", not "french maid"); jobs use the normalised name
    chars = roles.character.unique().tolist()
    with Pool(8) as pool:
        scanned = {}
        for part in pool.imap(names_scan, [chars[i:i + 200000] for i in range(0, len(chars), 200000)]):
            scanned.update(part)
    roles["groups"] = roles.character.map(lambda n: scanned.get(n, (set(), set()))[0])
    roles["slurs"] = roles.character.map(lambda n: scanned.get(n, (set(), set()))[1])
    occ_names = pd.Series(roles.name.unique())
    roles["occupation"] = roles.name.map(dict(zip(occ_names, occ_names.map(occupation))))
    roles["decade"] = roles.year // 10 * 10
    sl = roles.assign(has=roles.slurs.map(len) > 0).groupby("decade").agg(roles=("has", "size"), slur_names=("has", "sum"))
    sl["per_100k"] = sl.slur_names / sl.roles * 1e5
    sl.to_csv(OUT / "slurs_in_character_names_by_decade.csv")
    print(sl.round(1).to_string())
    lab = roles[roles.groups.map(len) > 0].explode("groups")
    lab["crime"] = lab.occupation.map(OCC_GROUP).eq("crime")
    lab["service"] = lab.occupation.map(OCC_GROUP).isin(["domestic", "hospitality", "personal"])
    allr = roles.assign(crime=roles.occupation.map(OCC_GROUP).eq("crime"),
                        service=roles.occupation.map(OCC_GROUP).isin(["domestic", "hospitality", "personal"]),
                        has_job=roles.occupation.notna())
    gl = lab.assign(has_job=lab.occupation.notna()).groupby("groups").agg(
        labelled_roles=("name", "size"), with_job=("has_job", "sum"), crime=("crime", "sum"), service=("service", "sum"),
        women=("female", "mean")).reset_index()
    gl["crime_share_of_jobs"] = gl.crime / gl.with_job
    gl["service_share_of_jobs"] = gl.service / gl.with_job
    base = allr[allr.has_job]
    gl["crime_share_all_characters"] = base.crime.mean()
    gl["service_share_all_characters"] = base.service.mean()
    gl.to_csv(OUT / "group_labelled_characters.csv", index=False)
    print(gl.sort_values("labelled_roles", ascending=False).round(3).to_string(index=False))
