"""Build DERIVED/lexicon.json: per group, its mention terms, stereotype words (with sources) and slurs.

Slurs that are also everyday words ("coconut", "cracker"...) would flood plot summaries with false hits, so single-word
slurs are kept only when rare in general English (wordfreq Zipf < 3.0; Zipf 3 ≈ once per million words) and longer
than three letters; the rest are recorded as `ambiguous_slurs` and not used for counting.
"""
import io
import json
import re

import pandas as pd
from wordfreq import zipf_frequency

from groups import GROUPS
from paths import DERIVED, LEXICONS

STOP = set("""love loves loved good at like likes eat eats ate eating make making made very the a an of and to in is are
be not people country person have has lot lots much many always often they them their with for on all most too so bad
great nice do does being been food prefer work sense humour humor kind type style really well best better big small
little high low know known get gets got go going thing things way use used look looks looking also just more less""".split())
SLUR_ZIPF_MAX = 3.0
NAME_TOKEN_MAX = 20  # a "slur" that is a given name or surname of >= 20 people on IMDb is a name in credits and plots
# Everyday words, places, products, characters or non-slurs that pass the two filters above (checked by hand on the
# terms that matched most often in summaries and character names, 2026-10-12).
EXCLUDE_SLURS = {"kimchi", "jaffa", "teapot", "baboon", "tonk", "spook", "skippy", "bimbo", "child molester", "jim crow",
                 "tonto", "pancho", "abbie", "timur", "haji", "kala", "jeet", "dogan", "gora", "mangal", "miya", "vanja",
                 "indio", "moro", "nance", "kaew", "abid", "zang", "polak", "froggy", "pansy", "sooty", "queers",
                 "nigar", "cholo", "blackie", "gringo", "aunt mary", "aunt jane", "aunt sally", "pocahontas", "ali baba",
                 "buckwheat", "vrindavan", "kaliya", "teabag", "gayest", "eight ball", "off the boat", "orangie",
                 "twinkie", "mullah", "coreano", "baluba", "greaser", "greasers"}
TARGET_TO_GROUP = [  # (regex on Wikipedia's target text, group); first match wins
    (r"african american|black", "black"), (r"italian", "italian"), (r"irish", "irish"), (r"mexican", "mexican"),
    (r"russian", "russian"), (r"german", "german"), (r"french", "french"), (r"british|english", "british"),
    (r"polish|poles", "polish"), (r"greek", "greek"), (r"chinese", "chinese"), (r"japanese", "japanese"),
    (r"korean", "korean"), (r"pakistani|south asian|indian people|bangladeshi|indians\b", "south_asian"),
    (r"arab|palestin|lebanese|syrian|iraqi|saudi", "arab"), (r"iranian|persian", "iranian"), (r"turk", "turkish"),
    (r"nigerian|somali|african|ethiopian|ghanaian|kenyan", "african"),
    (r"hispanic|latin|puerto|cuban|dominican|colombian", "latino"),
    (r"asian|vietnam|filipino|thai|cambodian|lao", "asian"), (r"native american|american indian|indigenous", "native_american"),
    (r"roma|gypsy", "roma"), (r"white", "white"), (r"jew", "jewish"), (r"muslim|islam", "muslim"),
    (r"catholic|christian|protestant|mormon|latter day|baptist|evangelical|methodist|pentecostal", "christian"),
    (r"hindu|brahmin", "hindu"),
]
mention_words = {w for g in GROUPS.values() for t in g[1] for w in t.split()}


def content_words(phrase):
    return [w for w in re.findall(r"[a-z][a-z\-']+", str(phrase).lower())
            if w not in STOP and w not in mention_words and len(w) >= 3 and zipf_frequency(w, "en") < 5.5]


lex = {name: {"category": g[0], "mentions": g[1], "attributes": {}, "offensive_attributes": [], "slurs": {},
              "ambiguous_slurs": []} for name, g in GROUPS.items()}
offensive = {name: set() for name in GROUPS}
NEGATION = re.compile(r"\b(not|no|never|don't|dont|can't|cant|cannot|unable|lack|lacks|lacking|without)\b")
SEEGULL_OFFENSIVE = 1.0  # SeeGULL raters' mean offensiveness, -1 (not offensive) to 4 (very offensive)
# Pejorative words among the StereoSet and curated stereotype words (SeeGULL words carry their own rating).
CURATED_PEJORATIVE = set("""mafia mob mobster gangster gangsters capo vendetta drunk drunken drunkard brawl temper
hot-tempered illegal bandit smuggler lazy gang gangs cartel drug drugs spy ruthless nazi nazis rude arrogant snob
snobbish stupid triad opium terrorist terrorists terrorism bomb bomber fanatic extremist extremism jihad harem warlord
pirate scam starving ghetto hood thug criminal crime welfare dealer submissive savage savages scalp scalped warpath
alcoholic thief steal curse con racist inbred supremacist greedy stingy cheap neurotic radical oppressed hypocrite
zealot cult bigot superstitious flamboyant promiscuous effeminate hiv butch man-hating predatory deceive deception
freak prostitute seductive humorless cold barbaric corrupt dirty violent dangerous backward primitive uncivilized
poor poverty illiterate""".split())


def add_attr(group, word, source, offensiveness=None):
    lex[group]["attributes"].setdefault(word, [])
    if source not in lex[group]["attributes"][word]:
        lex[group]["attributes"][word].append(source)
    if (offensiveness is not None and offensiveness >= SEEGULL_OFFENSIVE) or word in CURATED_PEJORATIVE:
        offensive[group].add(word)


# SeeGULL
s = pd.read_csv(LEXICONS / "seegull/stereotypes_global_v2.csv")
s = s[(s.region_stereo + s.NA_stereo) >= 3]
for name, g in GROUPS.items():
    for ident in g[2]:
        rows = s[s.identity.str.lower() == ident.lower()]
        for phrase, score in zip(rows.attribute, rows["mean offensiveness_score"]):
            if NEGATION.search(str(phrase).lower()):
                continue  # "not generous" must not turn into "generous"
            words = content_words(phrase)
            for w in words:
                # a rating marks a word offensive only when the attribute is that word alone ("child molester" would
                # otherwise flag "child"); words from longer phrases need HurtLex or the pejorative list
                add_attr(name, w, "seegull", score if len(words) == 1 else None)

# StereoSet: the word that fills BLANK in the stereotype sentence
ss = json.load(open(LEXICONS / "stereoset/dev.json"))["data"]["intrasentence"]
for item in ss:
    groups = [n for n, g in GROUPS.items() if item["target"] in g[3]]
    if not groups:
        continue
    context = set(re.findall(r"[a-z']+", item["context"].lower()))
    for sent in item["sentences"]:
        if sent["gold_label"] == "stereotype":
            for w in content_words(" ".join(x for x in re.findall(r"[a-z']+", sent["sentence"].lower()) if x not in context)):
                for n in groups:
                    add_attr(n, w, "stereoset")

for name, g in GROUPS.items():
    for w in g[5]:
        add_attr(name, w.lower(), "curated")


def slur_terms(cell):
    cell = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", str(cell)).lower()
    parts = re.split(r"\s*/\s*|,\s*|\s+or\s+|;\s*", cell)
    return [p.strip(" .\"'") for p in parts if p.strip(" .\"'")]


import duckdb  # noqa: E402

from paths import IMDB  # noqa: E402

name_tokens = dict(duckdb.connect().execute(f"""SELECT token, COUNT(*) FROM (
    SELECT unnest(string_split(lower(primaryName), ' ')) AS token FROM read_csv('{IMDB / "name.basics.tsv.gz"}',
    delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'nconst':'VARCHAR','primaryName':'VARCHAR',
    'birthYear':'VARCHAR','deathYear':'VARCHAR','primaryProfession':'VARCHAR','knownForTitles':'VARCHAR'}}))
    GROUP BY token HAVING COUNT(*) >= {NAME_TOKEN_MAX}""").fetchall())


# Terms pruned with the labelled sample (names, places, foods, descriptive terms in this corpus); kept in a private
# file next to the source lexicons because it holds the terms themselves.
_pruned = LEXICONS / "slur_exclusions.json"
PRUNED = set(json.load(open(_pruned))["drop_terms"]) if _pruned.exists() else set()


def keep_slur(term):
    if term in EXCLUDE_SLURS or term in PRUNED:
        return False
    words = term.split()
    if len(words) > 1:
        return True
    return len(term) > 3 and zipf_frequency(term, "en") < SLUR_ZIPF_MAX and term not in name_tokens


for page, target_col in [("List_of_ethnic_slurs", "Targets"), ("List_of_religious_slurs", "Targeted demographic")]:
    html = json.load(open(LEXICONS / f"wikipedia/{page}.json"))["parse"]["text"]
    tables = [t for t in pd.read_html(io.StringIO(html)) if "Term" in t.columns and target_col in t.columns]
    for term_cell, target in pd.concat(tables)[["Term", target_col]].itertuples(index=False):
        group = next((g for pattern, g in TARGET_TO_GROUP if re.search(pattern, str(target).lower())), None)
        if not group:
            continue
        for term in slur_terms(term_cell):
            if term in mention_words:
                continue
            if keep_slur(term):
                lex[group]["slurs"][term] = "wikipedia"
            elif term not in lex[group]["ambiguous_slurs"]:
                lex[group]["ambiguous_slurs"].append(term)

h = pd.read_csv(LEXICONS / "hurtlex/hurtlex_EN_1.2.tsv", sep="\t")
identity = {"gay", "gays", "homosexual", "homosexuals", "lesbian", "lesbians", "queer", "bisexual", "transgender"}
for term in h[(h.category == "om") & (h.level == "conservative")].lemma.str.lower().unique():
    if term in identity:
        continue
    if keep_slur(term):
        lex["gay"]["slurs"][term] = "hurtlex"
    else:
        lex["gay"]["ambiguous_slurs"].append(term)

hurtful = set(h[h.level == "conservative"].lemma.str.lower())
# single SeeGULL words rated offensive whose meaning depends on the lost context of the rating
NOT_OFFENSIVE_ALONE = {"communicate", "educate", "fast", "hot", "proud", "learn", "generous", "child"}
for name in GROUPS:
    offensive[name] |= {w for w in lex[name]["attributes"] if w in hurtful}
    lex[name]["offensive_attributes"] = sorted((offensive[name] & set(lex[name]["attributes"])) - NOT_OFFENSIVE_ALONE)
json.dump(lex, open(DERIVED / "lexicon.json", "w"), indent=1, ensure_ascii=False)
summary = pd.DataFrame({n: {"mentions": len(v["mentions"]), "attributes": len(v["attributes"]),
                            "offensive": len(v["offensive_attributes"]),
                            "from seegull": sum("seegull" in s for s in v["attributes"].values()),
                            "from stereoset": sum("stereoset" in s for s in v["attributes"].values()),
                            "curated": sum("curated" in s for s in v["attributes"].values()),
                            "slurs kept": len(v["slurs"]), "slurs dropped as everyday words": len(v["ambiguous_slurs"])}
                        for n, v in lex.items()}).T
print(summary.to_string())
for g in ["italian", "russian", "mexican", "arab", "german", "muslim", "native_american", "jewish", "gay"]:
    print(f"{g} offensive: " + ", ".join(lex[g]["offensive_attributes"][:40]))
