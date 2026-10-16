"""Build DERIVED/lexicon_social.json for the social groups of social_groups.py, in lexicon.json's format, so that
group_stereotypes.py can scan them with GROUP_SET=social. Pejorative labels take the place that slurs have for the
other groups; the pejorative stereotype words make up the offensive index.
"""
import json

import pandas as pd

from paths import DERIVED
from social_groups import PEJORATIVES, SOCIAL_GROUPS

lex = {}
for name, (category, mentions, sources, pejorative_words) in SOCIAL_GROUPS.items():
    pejoratives = PEJORATIVES[name]
    attributes = {}
    for source, words in sources.items():
        for w in words:
            attributes.setdefault(w, []).append(source)
    assert not set(pejoratives) & set(attributes), name  # labels are counted apart from stereotype words
    lex[name] = {"category": category, "mentions": mentions, "attributes": attributes,
                 "offensive_attributes": sorted(pejorative_words & set(attributes)),
                 "slurs": {t: "curated" for t in pejoratives}, "ambiguous_slurs": []}
json.dump(lex, open(DERIVED / "lexicon_social.json", "w"), indent=1, ensure_ascii=False)
print(pd.DataFrame({n: {"mentions": len(v["mentions"]), "stereotype words": len(v["attributes"]),
                        "from bbq": sum("bbq" in s for s in v["attributes"].values()),
                        "pejorative words": len(v["offensive_attributes"]), "pejorative labels": len(v["slurs"])}
                    for n, v in lex.items()}).T.to_string())
