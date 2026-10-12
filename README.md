# Stereotypes on screen

How film and TV cast, write about and label women and social groups, measured across IMDb's full title catalogue,
TMDb's TV and film records and three million plot summaries, 1920–2026.

This expands [Stefano Li Pira](https://github.com/stefanolipira)'s 2022 initiative to measure stereotypes in TV
series, which he and Alexandru Pascu first explored with a BERT classifier (kept in [`2022-bert/`](2022-bert/), tag
`v2022`). The follow-up replaces the classifier with five measures that can be checked against the data.

*Content note: the word lists and result tables contain stereotypes about groups of people, collected in order to
measure them. Slurs are counted but left out of every published table.*

## At a glance

| Study | Main finding |
| --- | --- |
| [1. Who plays whom](#1-who-plays-whom) | Detectives, judges and lawyers went from almost never women to 20–30% women; nurses, nannies and receptionists are still 83–90% women. |
| [2. Who writes the episode](#2-who-writes-the-episode) | Episodes written by women cast about 3 points more women as guest stars than episodes of the same show written by men. |
| [3. What characters do in plot summaries](#3-what-characters-do-in-plot-summaries) | Women went from 28% to 46% of the people summaries talk about, but are still described by their looks 1.5 times as often as their share. |
| [4. Do audiences care?](#4-do-audiences-care) | Hardly: none of these measures moves an episode's IMDb rating by more than 0.02 points. |
| [5. Groups, stereotype words and slurs](#5-groups-stereotype-words-and-slurs) | Slurs in character names fell sevenfold, but pejorative stereotype words still cluster around group mentions as much as in the 1950s. |

## 1. Who plays whom

Jobs in credited character names ("Nurse", "Detective Morales", "Dr. Patel"), and whether an actor or an actress
played them. English-language film and TV: 2.5 million named fictional roles (of 7 million in all languages).

<img src="results/occupations/skew_by_job_english.png" alt="Casting skew by job, 2010–26 vs 1920–69" width="640">

- In 2010–26 women play 90% of nannies, 87% of nurses, 83% of receptionists and 74% of secretaries, but 26% of doctors,
  20% of detectives, 15% of police officers, 7% of soldiers and 3% of gangsters, against 39% of all named roles.
- The biggest changes are in professional and authority jobs: before 1970 women played 0–2% of detectives, judges,
  lawyers, doctors and professors; now they play 19–30%. Reporters went from 6% to 49%, bartenders from 0% to 28%.
- Jobs coded female barely moved: nurses are 87% women today (98% before 1970). Change came from women entering male
  roles on screen, not men entering female ones.

![Women's share of on-screen jobs by decade](results/occupations/women_share_by_decade_english.png)

Tables: [`results/occupations/`](results/occupations/).

## 2. Who writes the episode

TMDb lists writers, directors and guest cast, each with a gender, for 118,128 episodes of 8,156 scripted series.
Guest stars are cast per episode, which is where an episode's writers have most say.

![Guest casting by writers' gender](results/writers/guest_share_by_writers.png)

- Comparing episodes of the same series, those written only by women cast 2.9 percentage points more women among
  guest stars than those written only by men (95% interval 2.5–3.3); mixed teams 1.1 points more. The gap holds for
  English-language series (3.1) and for episodes aired since 2010 (2.6).
- A woman director adds 0.6 points on top of the writers.
- Between shows the gap is larger: series created only by women average 43% women among guest stars, those created
  only by men 35%, though such shows differ in many other ways too.
- Writers' gender makes no clear difference to whether guests get counter-stereotypical jobs (a woman detective, a
  male nurse).

Tables: [`results/writers/`](results/writers/).

## 3. What characters do in plot summaries

Three million English summaries (IMDb and TMDb) parsed into who-does-what. A character's gender comes from pronouns,
gendered words, or the title's own cast list.

![How summaries write about women vs men](results/agency/categories_by_decade.png)

![Words most tied to women and men](results/agency/distinctive_words.png)

- Women's share of the people summaries talk about rose from 28% (1950s) to 46% (2020s), and they are now the subject
  of the action as often as their share would predict.
- Appearance is the most persistent gap: women are described by their looks ("beautiful", "pretty", "attractive",
  "sexy") 3 times as often as their share in the 1950s and still 1.5 times today.
- Investigating, rescuing and committing violence started far below women's share (0.4–0.6) and are now close to it
  (0.8–1.0); romance-driven verbs fell from 1.4 to parity; emotional verbs ("cry", "worry", "panic") stay at 1.2.
- What happens to characters differs sharply: women are kidnapped, raped, abducted, assaulted and stalked; men are
  arrested, shot, wounded, convicted and jailed.

Tables: [`results/agency/`](results/agency/).

## 4. Do audiences care?

Each measure compared across episodes of the same series, with air-year effects, for 887,000 rated IMDb episodes.
This is the question the 2022 project asked with its classifier.

![Effects on ratings and votes](results/ratings/effects.png)

- None of the measures changes an episode's rating by more than 0.02 points out of 10: casting more women, writing
  women as the ones who act, or pairing groups with stereotypes all leave ratings essentially unchanged.
- Some get more attention: episodes whose summary describes a woman by her looks get 5% more votes, those with a woman
  as a victim or with a group stereotype 2–3% more.
- The 2022 null result therefore stands, now with measures that do capture what they claim to.

Table: [`results/ratings/models.csv`](results/ratings/models.csv).

## 5. Groups, stereotype words and slurs

Thirty groups (nationalities, ethnic groups, religions, sexual orientation and gender identity), each with a list of
stereotype words from published stereotype datasets and curated lists, and slurs from Wikipedia and HurtLex. The
pejorative index keeps only words rated offensive. The measures pass a check against viewers' own tags: titles tagged
"racism" contain slurs 9 times as often as other titles, those tagged "racial slur" 17 times as often, and titles
tagged "Italian stereotype" mention Italians 70 times as often.

![Slurs in character names by decade](results/groups/slurs_in_character_names.png)

- Characters credited with a slur in their name fell from 42 per 100,000 roles in the 1920s to 6 in the 2020s.

![Pejorative stereotype pairing by decade](results/groups_offensive/pejorative_pairing_by_decade.png)

- Pejorative stereotype words have not faded from summaries that mention a group: across groups they appear about 3
  times as often next to a group mention in every decade since the 1950s (2.9 then, 3.1 in the 2020s).
- Native Americans are the clear decline (15 times in the 1950s, 1.6 now, as "savage" and "warpath" disappeared).
  Germans (Nazi plots, 6 → 11) and Italians (mafia, 2 → 7) rose; Muslims have been tied to extremism words at about 5
  times since 2000 (3 times in the 1990s).

![Criminal jobs for group-labelled characters](results/groups/criminal_jobs_by_group_label.png)

- Characters credited by a group label ("Mexican bandit", "Russian agent") are cast as criminals far more often than
  other characters with a job (4.7%): Latino 40%, Native American 25%, Russian 21%, Asian 20%. French, Italian and
  gay-labelled characters are instead given service jobs (maid, waiter, hairdresser) about three times as often.

Tables: [`results/groups/`](results/groups/), [`results/groups_offensive/`](results/groups_offensive/).

## How it was measured

- **Gender** comes from IMDb's actor/actress credit category, TMDb's gender field and, in summaries, from pronouns,
  gendered nouns and the title's own cast list (a character name that belongs only to roles played by actresses counts
  as a woman). Non-binary identities are rarely recorded in either source.
- **Jobs** are recognised in character names with a hand-checked word list: the last job word wins ("police
  detective" → detective), words that are also surnames count only when used as a job ("the baker", not "John
  Baker"), and gendered job words (waitress, policeman) are kept apart. English-language productions only, since an
  English list misses feminine job forms in other languages.
- **Casting skew** compares women's share in a job with women's share of all named fictional roles in the same period,
  so the general rise in women's roles is not mistaken for a change in a job's image.
- **Within-series comparisons** (studies 2 and 4) use series and year fixed effects with standard errors clustered by
  series.
- **Summaries** are joined to IMDb episodes through TMDb only for series whose season and episode numbering lines up
  with IMDb's (episode titles or air years agree); 15,107 series that do not are kept apart.
- **Group mentions** are matched with their capitalisation ("Polish" counts, "polish the silver" does not). Stereotype
  words come from SeeGULL, StereoSet and short curated lists; the pejorative index keeps SeeGULL words its raters
  scored offensive, words in HurtLex and a short pejorative list. Slurs that are also common names or everyday words
  are dropped.

## Caveats

- Credits and summaries are written by IMDb and TMDb contributors, not by the productions.
- The full stereotype word lists mix pejorative words with neutral topic words ("mexico", "reservation"), which is why
  the pejorative index exists; both measure association, not a judgement on any single title.
- Ratings come from IMDb's self-selected voters.

## Reproduce

Set `STEREOTYPES_DATA` to a local folder holding the sources below, then run from `src/`: `build_roles.py`,
`tmdb_titles.py`, `occupation_gender.py`; `tmdb_episode_credits.py`, `writer_gender.py`; `build_texts.py`,
`parse_texts.py`, `agency.py`; `stereotype_lexicon.py`, `group_stereotypes.py` (and again with `INDEX=offensive`);
`ratings.py`; `figures.py`. Python 3.12 with pandas, pyarrow, duckdb, statsmodels, spacy (`en_core_web_sm`),
matplotlib, lxml and wordfreq. No raw data is stored in this repository.

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
