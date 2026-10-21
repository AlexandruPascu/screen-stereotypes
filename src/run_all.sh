#!/usr/bin/env bash
# Runs the whole pipeline in order, or from a given step onward:  bash run_all.sh [first-step]
# Needs STEREOTYPES_DATA (the local data folder) and PYTHON (defaults to python3). Logs go to $STEREOTYPES_DATA/logs/.
set -euo pipefail
cd "$(dirname "$0")"
: "${STEREOTYPES_DATA:?set STEREOTYPES_DATA to the data folder}"
PY=${PYTHON:-python3}
LOGS="$STEREOTYPES_DATA/logs"
mkdir -p "$LOGS"

# agency filters its word table against both lexicons, so it runs after stereotype_lexicon and social_lexicon
STEPS=(build_roles tmdb_titles occupation_gender tmdb_episode_credits writer_gender wiki_fetch build_texts parse_texts
       stereotype_lexicon group_stereotypes group_stereotypes_offensive social_lexicon group_stereotypes_social
       group_stereotypes_social_offensive agency ratings embeddings plot_scores classifier figures)
start=${1:-${STEPS[0]}}
running=false
for step in "${STEPS[@]}"; do
  [[ $step == "$start" ]] && running=true
  $running || continue
  echo "== $(date -Iseconds) $step"
  case $step in
    parse_texts) rm -rf "$STEREOTYPES_DATA/derived/parse"   # the parts are resumable; a new text table needs a clean parse
                 $PY -u parse_texts.py 13 > "$LOGS/$step.log" 2>&1 ;;
    group_stereotypes_offensive) INDEX=offensive $PY -u group_stereotypes.py > "$LOGS/$step.log" 2>&1 ;;
    group_stereotypes_social) GROUP_SET=social $PY -u group_stereotypes.py > "$LOGS/$step.log" 2>&1 ;;
    group_stereotypes_social_offensive) GROUP_SET=social INDEX=offensive $PY -u group_stereotypes.py \
                                          > "$LOGS/$step.log" 2>&1 ;;
    *) $PY -u "$step.py" > "$LOGS/$step.log" 2>&1 ;;
  esac
  grep -v Warning "$LOGS/$step.log" | tail -3
done
echo "== $(date -Iseconds) all done"
