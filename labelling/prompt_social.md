# Instructions for the labelling model: social groups

The items are made by `src/label_sample_social.py` and saved, with the key that links them back to titles, in
`$STEREOTYPES_DATA/derived/labelling_social/` (they contain summary text, so they are not in this repository). Check a
labels file with `python src/check_labels.py <file> --social [--double]`.

Give the model the whole of [`codebook_social.md`](codebook_social.md), then this message, then the summaries. Each
input line of `items.jsonl` is `{"id": "S00001", "text": "...", "double": false}` (`double` only marks items for the
second pass; the model ignores it); the model must return exactly one JSON object per input line, valid against
[`schema_social.json`](schema_social.json), in the same order, one per line (JSON Lines), with nothing else.

---

You are labelling film and TV plot summaries for a research project on stereotypes about immigrants and refugees,
disabled people, people with a mental illness, older people, poor and working-class people, and homeless people.
Follow the codebook exactly.

- Read each summary on its own. Judge only its text; do not use what you know about the title.
- Return one JSON object per summary with the fields `id`, `social_groups`, `pejorative_label`,
  `discrimination_theme`, `confidence`, `notes`, as defined in the codebook and schema. Use only the listed values.
- Quote at most 15 words of evidence per group. Never write out a pejorative label or a slur.
- Most summaries mention none of the six groups: then `social_groups` is `[]`, `pejorative_label` is `false` and
  `discrimination_theme` is `{"present": false, "stance": null}`.
- Output JSON Lines only: no commentary, no code fences.

Three invented examples (not from the data):

Input: `{"id": "S99997", "text": "A detective hunts a killer who escaped from a psychiatric hospital and preys on nurses."}`

Output: `{"id": "S99997", "social_groups": [{"group": "mental_illness", "portrayal": "stereotyped", "types": ["danger_threat"], "evidence": "a killer who escaped from a psychiatric hospital"}], "pejorative_label": false, "discrimination_theme": {"present": false, "stance": null}, "confidence": 3, "notes": ""}`

Input: `{"id": "S99998", "text": "A retired schoolteacher in her seventies trains for her first marathon while caring for her grandson."}`

Output: `{"id": "S99998", "social_groups": [{"group": "older_people", "portrayal": "counter_stereotyped", "types": [], "evidence": "A retired schoolteacher in her seventies trains for her first marathon"}], "pejorative_label": false, "discrimination_theme": {"present": false, "stance": null}, "confidence": 2, "notes": ""}`

Input: `{"id": "S99999", "text": "Two refugee brothers fight their deportation while a local journalist exposes the town's hostility."}`

Output: `{"id": "S99999", "social_groups": [{"group": "immigrant_refugee", "portrayal": "neutral", "types": [], "evidence": "Two refugee brothers fight their deportation"}], "pejorative_label": false, "discrimination_theme": {"present": true, "stance": "critical"}, "confidence": 3, "notes": ""}`

---

## Reliability

`items.jsonl` marks about 10% of items with `"double": true`. Label those twice (a second model, or the same model in
a fresh session) and save the second pass as `labels_second.jsonl`; the agreement between the two passes sets the
ceiling for every measure checked against these labels.
