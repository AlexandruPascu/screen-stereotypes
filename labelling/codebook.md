# Labelling codebook (v1)

Labels for a sample of film and TV plot summaries, used to check every measure in this project against a careful
reading (studies 3, 5, 6 and 7) and to train better models later. One label record per summary.

## What to judge

Judge **only the summary text in front of you**. Do not use what you know about the film or show, its cast, or its
reputation: the measures being checked only ever see the summary, so the labels must too. If the summary is too thin to
judge, say so with low confidence rather than guessing.

A **stereotype** is a widely held, oversimplified belief that a group of people shares a trait, role or behaviour. A
summary *portrays* a stereotype when it frames a character through that belief as if it were natural or unremarkable. A
summary that is *about* prejudice (a story of racism, sexism, homophobia) is not stereotyped for that reason alone: mark
the theme separately, and only mark a portrayal when the summary itself reproduces the stereotype uncritically.

## Fields

### 1. `gender`

- `portrayal`: `"none"`, `"mild"` or `"clear"` — does the summary frame a woman or a man through a gender stereotype?
  `mild` = present but not central; `clear` = the stereotype defines the character's role in the summary.
- `types`: every type that applies (empty list when `portrayal` is `"none"`):

| type | applies when |
| --- | --- |
| `appearance` | a woman is introduced or defined by her looks ("a beautiful blonde") when looks play no role in the plot |
| `sexualization` | a woman is framed as an object of desire or a temptress rather than as a person with aims |
| `domestic_role` | a woman's identity is reduced to wife, mother or homemaker; or a man's to breadwinner |
| `damsel_victim` | a woman exists in the plot mainly to be endangered, kidnapped or rescued by a man |
| `emotional_irrational` | a woman is shown as hysterical or irrational in contrast with calm, rational men |
| `romance_defined` | a woman's goals are reduced to winning or keeping a man |
| `occupational_marking` | a job is marked by gender as unusual ("a female surgeon") or assigned by stereotype ("the pretty secretary") |
| `male_dominance_norm` | violence, dominance or emotional coldness is presented as what makes a man a man |
| `incompetent_man_at_home` | a man is a bumbling, helpless figure in domestic or caring tasks, played for laughs |
| `other` | another gender stereotype; explain in `notes` |

- `counter_stereotype`: `true` when the summary clearly gives a woman or man a role against the stereotype and treats
  it as normal (a woman leads the investigation; a father is the main carer), otherwise `false`.

### 2. `groups`

One entry for **each** group the summary mentions or clearly identifies: a nationality, ethnic group, religion, sexual
orientation or gender identity (not women or men as such, which belong under `gender`). Empty list when there is none.

- `group`: the group, in plain words ("Italian", "Muslim", "Native American", "gay men", "trans woman").
- `portrayal`: `"neutral"`, `"stereotyped"` or `"counter_stereotyped"`.
- `types`: when stereotyped, every type that applies:

| type | applies when |
| --- | --- |
| `criminality` | the group is tied to crime, gangs, the mafia, drugs or cartels |
| `terrorism_extremism` | the group is tied to terrorism, fanaticism or extremism |
| `danger_threat` | the group is framed as dangerous or menacing in general |
| `savage_primitive` | the group is framed as savage, primitive, backward or uncivilised |
| `exotic_other` | the group is framed as mysterious, foreign or exotic rather than as people |
| `servile_role` | the group appears only as servants, maids, cooks, gardeners or helpers |
| `greed_money` | the group is tied to greed, money-grubbing or cheating |
| `hypersexual` | the group is framed as sexually excessive or predatory |
| `caricature` | the group is a comic accent, costume or cliché played for laughs |
| `victim_only` | the group exists in the plot only to suffer, with no agency |
| `other` | another stereotype; explain in `notes` |

- `evidence`: the shortest quote from the summary (at most 15 words) that supports the judgement.

### 3. `prejudice_theme`

- `present`: `true` when the story is *about* prejudice or discrimination (racism, sexism, homophobia, antisemitism,
  islamophobia, xenophobia, a hate crime, segregation), whatever its stance.
- `stance`: when present, `"critical"` (the story opposes or exposes the prejudice), `"neutral"` or `"endorsing"`;
  otherwise `null`.

### 4. `slur_present`

`true` when the summary itself contains an ethnic, religious, sexual-orientation or gender-identity slur. Never repeat
the slur anywhere in the record.

### 5. `confidence` and `notes`

- `confidence`: `1` (unsure, thin text), `2` (fairly sure) or `3` (sure).
- `notes`: optional, at most 20 words; required when a type is `other`.

## Rules for hard cases

- A stereotype a character voices and the story rejects is a `prejudice_theme` with stance `critical`, not a
  stereotyped portrayal.
- Historical settings: judge how the summary frames people, not whether the period was prejudiced.
- One mention of a nationality as plain fact ("an Italian chef") is `neutral` unless the summary leans on the cliché.
- Comedy is not an excuse: a caricature played for laughs is still `caricature`.
- When unsure between `mild` and `none`, choose `none` and set `confidence` to 1.
