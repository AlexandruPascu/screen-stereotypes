# Instructions for the labelling model

The items are made by `src/label_sample.py` and saved, with the key that links them back to titles, in
`$STEREOTYPES_DATA/derived/labelling/` (they contain summary text, so they are not in this repository). Once labelled,
run `python src/validate_labels.py labels.jsonl [labels_second.jsonl]`.

Give the model the whole of [`codebook.md`](codebook.md), then this message, then the summaries. Each input line of
`items.jsonl` is `{"id": "L00001", "text": "...", "double": false}` (`double` only marks items for the second pass;
the model ignores it); the model must return exactly one JSON object per input line,
valid against [`schema.json`](schema.json), in the same order, one per line (JSON Lines), with nothing else.

---

You are labelling film and TV plot summaries for a research project on stereotypes. Follow the codebook exactly.

- Read each summary on its own. Judge only its text; do not use what you know about the title.
- Return one JSON object per summary with the fields `id`, `gender`, `groups`, `prejudice_theme`, `slur_present`,
  `confidence`, `notes`, as defined in the codebook and schema. Use only the listed values for enumerated fields.
- Quote at most 15 words of evidence per group. Never write out a slur.
- If a summary is too short or vague to judge, mark `portrayal` as `"none"`, `groups` as found, and `confidence` 1.
- Output JSON Lines only: no commentary, no code fences.

Two invented examples (not from the data):

Input: `{"id": "L99998", "text": "When the town's beautiful new schoolteacher is kidnapped by bandits, the sheriff rides out to rescue her before her wedding."}`

Output: `{"id": "L99998", "gender": {"portrayal": "clear", "types": ["appearance", "damsel_victim"], "counter_stereotype": false}, "groups": [], "prejudice_theme": {"present": false, "stance": null}, "slur_present": false, "confidence": 3, "notes": ""}`

Input: `{"id": "L99999", "text": "A Muslim lawyer defends a neighbour accused of a hate crime and confronts the prejudice of her own colleagues."}`

Output: `{"id": "L99999", "gender": {"portrayal": "none", "types": [], "counter_stereotype": true}, "groups": [{"group": "Muslim", "portrayal": "counter_stereotyped", "types": [], "evidence": "A Muslim lawyer defends a neighbour"}], "prejudice_theme": {"present": true, "stance": "critical"}, "slur_present": false, "confidence": 3, "notes": ""}`

---

## Reliability

`items.jsonl` marks 300 items with `"double": true`. Label those twice (a second model, or the same model in a fresh
session) and save the second pass as `labels_second.jsonl`; `validate_labels.py` reports how often the two agree, which
sets the ceiling for every measure checked against these labels.
