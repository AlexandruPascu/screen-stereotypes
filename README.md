# Stereotypes on screen

How film and TV cast, write about and label women and social groups, measured across IMDb's full title catalogue,
TMDb's TV and film records and three million plot summaries (IMDb, TMDb and Wikipedia), 1920–2026.

This expands [Stefano Li Pira](https://github.com/stefanolipira)'s 2022 initiative to measure stereotypes in TV
series, which he and Alexandru Pascu first explored with a BERT classifier (kept in [`2022-bert/`](2022-bert/), tag
`v2022`). The follow-up replaces the classifier with measures that can be checked against the data, then rebuilds the
classifier properly and adds the word-embedding method from the project's original reading list.

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
| [6. The language of plot summaries](#6-the-language-of-plot-summaries) | Word embeddings trained on the summaries track the census and on-screen casting; gender-coded wording has halved since the 1950s, while terrorism words moved back towards Islam after 2001. |
| [7. A stereotype classifier, done properly](#7-a-stereotype-classifier-done-properly) | Without the 2022 length shortcut, predicting viewers' stereotype tags from a summary reaches AUC 0.71–0.76, not 0.95; the tagged share is falling slowly. |

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

Three million English summaries (IMDb, TMDb and Wikipedia) parsed into who-does-what. A character's gender comes from pronouns,
gendered words, or the title's own cast list.

![How summaries write about women vs men](results/agency/categories_by_decade.png)

![Words most tied to women and men](results/agency/distinctive_words.png)

- Women's share of the people summaries talk about rose from 28–29% (1950s–60s) to 46% (2020s), and they are now the
  subject of the action as often as their share would predict.
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

- No measure has a clear effect on an episode's rating larger than 0.02 points out of 10: casting more women, writing
  women as the ones who act, or pairing groups with stereotypes all leave ratings essentially unchanged.
- Some get more attention: episodes whose summary describes a woman by her looks get 5% more votes, those with a woman
  as a victim or with a group stereotype 2–3% more.
- The 2022 null result therefore stands, now with measures that do capture what they claim to.

Table: [`results/ratings/models.csv`](results/ratings/models.csv).

## 5. Groups, stereotype words and slurs

Thirty groups (nationalities, ethnic groups, religions, sexual orientation and gender identity), each with a list of
stereotype words from published stereotype datasets and curated lists, and slurs from Wikipedia and HurtLex. The
pejorative index keeps only words rated offensive. The measures pass a check against viewers' own tags: titles tagged
"racism" contain slurs 11 times as often as other titles, those tagged "racial slur" 17 times as often, and titles
tagged "Italian stereotype" mention Italians 70 times as often.

![Slurs in character names by decade](results/groups/slurs_in_character_names.png)

- Characters credited with a slur in their name fell from 42 per 100,000 roles in the 1920s to 6 in the 2020s.

![Pejorative stereotype pairing by decade](results/groups_offensive/pejorative_pairing_by_decade.png)

- Pejorative stereotype words have not faded from summaries that mention a group: across groups they appear about 3
  times as often next to a group mention in every decade since the 1950s (2.8 then, 3.1 in the 2020s).
- Native Americans are the clear decline (15 times in the 1950s, 1.5 now, as "savage" and "warpath" disappeared).
  Germans (Nazi plots, 6 → 10) and Italians (mafia, 2 → 6.5) rose; Muslims have been tied to extremism words at about 5
  times since 2000 (3 times in the 1990s).

![Criminal jobs for group-labelled characters](results/groups/criminal_jobs_by_group_label.png)

- Characters credited by a group label ("Mexican bandit", "Russian agent") are cast as criminals far more often than
  other characters with a job (4.7%): Latino 40%, Native American 25%, Russian 21%, Asian 20%. French, Italian and
  gay-labelled characters are instead given service jobs (maid, waiter, hairdresser) about three times as often.

Tables: [`results/groups/`](results/groups/), [`results/groups_offensive/`](results/groups_offensive/).

## 6. The language of plot summaries

The method from the project's original reading list (Garg et al. 2018, *PNAS*): word embeddings trained on the
summaries themselves (198 million words), with Garg et al.'s published word lists for women and men, occupations,
adjectives, surnames by group, and religion and terrorism words. Early decades are small, so one model is trained on all
text with every job and adjective word tagged by decade ("nurse_1960s"), and the gender and group words shared across
decades (temporal referencing); three training runs, averaged.

![Job words in the summaries' language vs women's share of the job on screen](results/embeddings/language_vs_casting.png)

- The measure passes the paper's own test: how close a job word sits to women's words tracks women's real share of
  that job in the US census (correlation 0.52–0.69 by decade), and tracks women's share of that job on screen even more
  closely (0.80–0.91). The language of summaries, the casting and the workforce tell the same story.

![Gender and group bias of word sets by decade](results/embeddings/bias_by_decade.png)

- Job words moved from clearly nearer men's words in the 1950s to slightly nearer women's words in the 2020s;
  intelligence words did the same. Appearance words sit nearest women's words in every decade.
- "Outsider" adjectives (alien, devious, inscrutable…) sat closer to Asian and Chinese surnames than ordinary
  personality adjectives did until the 1980s; since the 1990s the gap is close to zero. Intelligence words sit further
  from Hispanic than from White surnames in every decade.
- Terrorism words sat nearer Islam than Christianity words in the 1950s, were almost neutral in the 1990s, and moved
  back after 2001 to their 1950s level.

![Gender-coded wording around women and men, by decade](results/embeddings/plot_gender_congruence_by_decade.png)

- Per summary, using general-English word vectors (GloVe): the words used for women are more feminine-coded, and those
  for men more masculine-coded, than the reverse. That gap has halved for films and fallen by two thirds for TV since
  the 1950s. Westerns have the most gender-coded wording, animation the least.

Tables: [`results/embeddings/`](results/embeddings/).

## 7. A stereotype classifier, done properly

The 2022 notebook's question, rebuilt: from a summary alone, can a model tell which titles viewers tagged with a
stereotype or prejudice keyword ("racist remark", "Italian stereotype", "homophobia"…)? 158,004 tagged titles, 7,222 of
them with such a keyword.

![Classifier calibration and scores by decade](results/classifier/classifier.png)

- The 2022 shortcut is gone: untagged comparison titles are matched on medium, decade, number of keywords and text
  length, so a model that sees only the length scores AUC 0.50, pure chance.
- With whole series kept on one side of each cross-validation split, word-vector and word-count models reach AUC
  0.71 (films 0.73, TV 0.67), and 0.68–0.74 across short, medium and long summaries.
  A 2021 sentence-embedding model (`all-mpnet-base-v2`) used as frozen features does better: AUC 0.76 (films 0.79,
  TV 0.72). A small BERT (`MiniLM`) fine-tuned on the CPU did not converge in its one run (AUC 0.53) and is not used;
  newer open models are the next comparison. The 2022 model's 0.95 came mostly from text length; viewer tags are
  noisy labels, so 0.7–0.76 is what a summary can honestly tell so far.
- Calibrated on held-out folds, the scores are real probabilities. Applied to every summary (with the word-vector
  model; the sentence model is too slow on a CPU for three million texts), the share likely to be tagged peaks for 1970s
  films and falls slowly after; for TV it falls from 5.4% in the 1950s to 4.2% now.
- The words that weigh most are the expected ones ("racist", "racism", "homophobic", "discrimination", "prejudice",
  "interracial"); words found in fewer than 20 different films or series are left out so the model cannot learn
  which shows were tagged.

Tables: [`results/classifier/`](results/classifier/).

**Next: a careful reading.** Viewer tags and word lists are proxies. A labelling model will read 3,000 sampled summaries
against a written codebook ([`labelling/`](labelling/)): 1,400 at random to estimate how common each stereotype really
is, 1,600 drawn from what each measure flags to check its precision, and 300 labelled twice to measure how reliable the
labels themselves are. `src/validate_labels.py` then scores every measure in this repository against that reading.

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
- **Wikipedia episode summaries** come from season and episode-list pages ({{Episode list}}), traced to the series'
  IMDb id through Wikidata and kept only for series whose episode titles agree with IMDb's.

## Caveats

- Credits and summaries are written by IMDb and TMDb contributors, not by the productions.
- The full stereotype word lists mix pejorative words with neutral topic words ("mexico", "reservation"), which is why
  the pejorative index exists; both measure association, not a judgement on any single title.
- Ratings come from IMDb's self-selected voters.
- Garg et al.'s Black surname list overlaps heavily with common White surnames, so the embeddings give no usable
  Black–White contrast; surnames also stand in for groups only roughly.

## Reproduce

Set `STEREOTYPES_DATA` to a local folder holding the sources below and `PYTHON` to an environment with pandas, pyarrow,
duckdb, statsmodels, scikit-learn, gensim, spacy (`en_core_web_sm`), torch, sentence-transformers, transformers,
mwparserfromhell, matplotlib, lxml and wordfreq, then run `bash src/run_all.sh` (or `bash src/run_all.sh <step>` to
start from a step). `src/transformer.py` runs separately (CPU, about two hours). No raw data is stored in this
repository.

## Data and terms

- IMDb non-commercial datasets (2026-10-07): personal, non-commercial use; not republished here.
  *Information courtesy of IMDb ([https://www.imdb.com](https://www.imdb.com)). Used with permission.*
- TMDb API (fetched 2026-10-07 to 09). Cached copies are refreshed or deleted within six months; none are shared.

  ![TMDB](docs/tmdb-logo.svg)

  This product uses the TMDB API but is not endorsed or certified by TMDB.
- Plot text and keywords collected for the 2022 project, and IMDb's 2017 plot list: private copies, not republished.
- Wikipedia episode summaries (CC BY-SA 4.0) and Wikidata links (CC0), fetched through the official APIs.
- Garg et al. (2018) word lists and their census occupation shares by decade (from IPUMS), from the authors'
  repository; GloVe word vectors (PDDL); `all-mpnet-base-v2` (Apache-2.0) and `MiniLM-L12-H384` (MIT).
- SeeGULL (Google, CC BY 4.0), StereoSet (CC BY-SA 4.0), HurtLex (CC BY-NC-SA 4.0), Wikipedia lists of ethnic and
  religious slurs (CC BY-SA 4.0). The lexicon built from them is not included; it keeps their licences.

## Licence

Code: MIT (see `LICENSE`; the 2022 notebook is © Alexandru Pascu and Stefano Li Pira). Results tables and figures:
CC BY 4.0, with the source attributions above.
