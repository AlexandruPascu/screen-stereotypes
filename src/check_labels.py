"""Format and coverage check for a labels file, for the labelling model to run on its own output (writes nothing).

Usage: check_labels.py labels.jsonl [--double]
  --double  check a second-pass file: it should cover exactly the items marked "double"
"""
import json
import sys

from jsonschema import Draft202012Validator

from paths import DERIVED, RESULTS

validator = Draft202012Validator(json.load(open(RESULTS.parent / "labelling/schema.json")))
items = [json.loads(line) for line in open(DERIVED / "labelling/items.jsonl")]
expected = [i["id"] for i in items if i["double"]] if "--double" in sys.argv else [i["id"] for i in items]
seen, problems = [], []
for n, line in enumerate(open(sys.argv[1]), 1):
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError as e:
        problems.append(f"line {n}: not JSON ({e.msg})")
        continue
    for error in validator.iter_errors(record):
        problems.append(f"line {n} ({record.get('id')}): {error.message[:120]}")
    seen.append(record.get("id"))
dupes = sorted({i for i in seen if seen.count(i) > 1})
missing = [i for i in expected if i not in set(seen)]
extra = sorted(set(seen) - set(expected))
print(f"records {len(seen):,} of {len(expected):,} expected; format problems {len(problems):,}; duplicates {len(dupes):,}; "
      f"missing {len(missing):,}; unexpected ids {len(extra):,}")
for p in problems[:20]:
    print("  " + p)
if missing:
    print(f"  first missing ids: {', '.join(missing[:10])}")
sys.exit(1 if problems or dupes or missing or extra else 0)
