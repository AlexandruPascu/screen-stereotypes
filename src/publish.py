"""Word-level result tables are published without slurs or insulting labels: any word in the slur lists (lexicon.json)
or among study 11's insulting labels (lexicon_social.json) is dropped before writing."""
import json

from paths import DERIVED


def without_slurs(df, column):
    terms = set()
    for name in ["lexicon.json", "lexicon_social.json"]:
        for v in json.load(open(DERIVED / name)).values():
            terms |= {t.lower() for t in list(v["slurs"]) + list(v["ambiguous_slurs"])}
    return df[~df[column].str.lower().isin(terms)]
