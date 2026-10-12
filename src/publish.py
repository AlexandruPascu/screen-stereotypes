"""Word-level result tables are published without slurs: any word in the slur lists is dropped before writing."""
import json

from paths import DERIVED


def without_slurs(df, column):
    lex = json.load(open(DERIVED / "lexicon.json"))
    slurs = {t for v in lex.values() for t in list(v["slurs"]) + v["ambiguous_slurs"]}
    return df[~df[column].str.lower().isin(slurs)]
