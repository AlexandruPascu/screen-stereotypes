# Stereotypes on screen

How film and TV cast, write about and label women and social groups, measured across IMDb's full title catalogue,
TMDb's TV and film records and three million plot summaries.

This expands [Stefano Li Pira](https://github.com/stefanolipira)'s 2022 initiative to measure stereotypes in TV
series, which he and Alexandru Pascu first explored with a BERT classifier (kept in [`2022-bert/`](2022-bert/), tag
`v2022`). The follow-up replaces the classifier with measures that can be checked against the data: who is cast in
which job, who writes the episode, what characters do in plot summaries, and which words attach to which groups.

No raw data lives in this repository. The scripts read local copies of the sources listed below and write aggregate
tables and figures to `results/`.

*Content note: the word lists and result tables contain stereotypes about groups of people, collected in order to
measure them. Slurs are used for counting but left out of every published table.*

## Findings so far

1. **Who plays whom.** In English-language film and TV from 2010 to 2026, women play 87% of nurses, 83% of
   receptionists and 74% of secretaries, but 26% of doctors, 20% of detectives, 15% of police officers and 7% of
   soldiers, against 41% of all named roles. Before 1970 almost no detective, judge or lawyer was played by a woman;
   now 20–30% are. (`results/occupations/`)
2. **Who writes matters a little.** Within the same series, episodes written only by women cast about 3 percentage
   points more women as guest stars than episodes written only by men; a woman director adds less than one point.
   (`results/writers/`)
3. **How summaries describe women.** Women's share of the people mentioned in plot summaries rose from 28% (1950s) to
   46% (2020s), and they now drive the action about as often as men. They are still described by their looks 1.5
   times as often as their share would predict (3 times in the 1950s), while investigating, rescuing and violence are
   close to parity after starting far below it. (`results/agency/`)
4. **Ratings barely move.** Within a series, none of these measures changes an episode's IMDb rating by more than
   0.02 points; episodes that describe women by their looks, as victims, or pair groups with stereotype words attract
   2–5% more votes. (`results/ratings/`)
5. **Groups and slurs.** Slurs in credited character names fell from 42 to 6 per 100,000 roles between the 1920s and
   the 2020s. Characters credited by a group label ("Mexican bandit", "Russian agent") are given criminal jobs far
   more often than other characters. Stereotype words have not faded from summaries that mention a group: the pairing
   is as strong now as in the 1950s, with Native Americans as the clear exception. (`results/groups/`,
   `results/groups_offensive/` for the pejorative words only)

## Analyses

| # | Question | Script | Output |
| --- | --- | --- | --- |
| 1 | Which jobs are characters given, by gender, 1920–2026? | `build_roles.py`, `occupation_gender.py` | `results/occupations/` |
| 2 | Does the writer's gender change who is cast in an episode? | `tmdb_episode_credits.py`, `writer_gender.py` | `results/writers/` |
| 3 | What do women and men do in plot summaries, and how are they described? | `build_texts.py`, `parse_texts.py`, `agency.py` | `results/agency/` |
| 4 | Do audiences rate episodes differently when casting or writing differs? | `ratings.py` | `results/ratings/` |
| 5 | Which stereotype words and slurs attach to which groups? | `stereotype_lexicon.py`, `group_stereotypes.py` | `results/groups/` |

Set `STEREOTYPES_DATA` to the local data folder, then run from `src/`: `build_roles.py`, `tmdb_titles.py`,
`occupation_gender.py`; `tmdb_episode_credits.py`, `writer_gender.py`; `build_texts.py`, `parse_texts.py`,
`agency.py`; `stereotype_lexicon.py`, `group_stereotypes.py` (and again with `INDEX=offensive`); `ratings.py`.
Python 3.12 with pandas, pyarrow, duckdb, statsmodels, spacy (`en_core_web_sm`), matplotlib, lxml, wordfreq.

## Method notes

- **Gender** comes from IMDb's actor/actress credit category, TMDb's gender field (0 unknown, 1 woman, 2 man,
  3 non-binary) and, in summaries, from pronouns, gendered nouns and the title's own cast list (a character name that
  belongs only to roles played by actresses counts as a woman). Non-binary identities are rarely recorded.
- **Jobs** are recognised in character names ("Nurse", "Detective Morales", "Dr. Patel") with a hand-checked word
  list: the last job word wins ("police detective" → detective), job words that are also surnames count only when used
  as a job ("the baker", not "John Baker"), and gendered job words (waitress, policeman) are kept out of the main
  measure. The main scope is English-language productions, since an English list misses feminine job forms in other
  languages.
- **Casting skew** is the log odds of a woman being cast in a job against women's share of all named fictional roles
  in the same period, so a general rise in women's roles is not mistaken for a change in a job's image.
- **Within-series comparisons** (analyses 2 and 4) use series and year fixed effects with standard errors clustered by
  series: they compare episodes of the same show.
- **Summaries** are joined to IMDb episodes through TMDb only for series whose season and episode numbering is shown
  to line up with IMDb's (episode titles or air years agree); 15,107 series that do not line up are kept apart.
- **Group stereotypes** pair group mentions (matched with their capitalisation, so "Polish" counts and "polish" does
  not) with stereotype words from SeeGULL, StereoSet and short curated lists; lift is how much more often those words
  appear when the group is mentioned. The offensive-only index keeps the words SeeGULL's raters scored as offensive,
  words in HurtLex, and a short pejorative list. Slurs come from Wikipedia's lists and HurtLex, minus terms that are
  also common names or everyday words. Both are validated against viewer tags (titles tagged "racism" contain 9 times
  as many slurs). Word-level result tables leave slurs out.

## Caveats

- Credits and summaries are written by IMDb and TMDb contributors, not by the productions.
- The full stereotype word lists mix pejorative stereotypes with neutral topic words ("mexico", "reservation"), which
  is why the offensive-only index exists; lift is an association measure, not a judgement on any title.
- Ratings come from IMDb's self-selected voters.

## Data and terms

- IMDb non-commercial datasets (2026-10-07): personal, non-commercial use; not republished here.
  *Information courtesy of IMDb ([https://www.imdb.com](https://www.imdb.com)). Used with permission.*
- TMDb API (fetched 2026-10-07 to 09). Cached copies are refreshed or deleted within six months; none are shared.

  ![TMDB](docs/tmdb-logo.svg)

  This product uses the TMDB API but is not endorsed or certified by TMDB.
- Plot text and keywords collected for the 2022 project, and IMDb's 2017 plot list: private copies, not republished.
- SeeGULL (Google, CC BY 4.0), StereoSet (CC BY-SA 4.0), HurtLex (CC BY-NC-SA 4.0), Wikipedia lists of ethnic and
  religious slurs (CC BY-SA 4.0). The lexicon built from them is not included; it keeps their licences.

## Licence

Code: MIT (see `LICENSE`; the 2022 notebook is © Alexandru Pascu and Stefano Li Pira). Results tables and figures:
CC BY 4.0, with the source attributions above.
