# Labelling codebook: social groups (v2.0)

A second, separate pass over film and TV plot summaries, for six groups the first codebook ([`codebook.md`](codebook.md))
left out: **immigrants and refugees, disabled people, people with a mental illness, older people, poor and
working-class people, and homeless people.** Gender, nationality, ethnicity, religion and sexuality are not labelled
in this pass. One label record per summary.

## What to judge

The same principles as the first codebook:

- Judge **only the summary text in front of you**. Do not use what you know about the film or show.
- A **stereotype** is a widely held, oversimplified belief that a group of people shares a trait, role or behaviour.
  A summary *portrays* a stereotype when it frames a character through that belief as if it were natural or
  unremarkable. Positive stereotypes (the wise old man, the noble poor, the inspiring disabled hero) count too.
- A summary that is *about* discrimination is not stereotyped for that reason alone: mark the theme separately, and
  mark a portrayal only when the summary itself reproduces the stereotype uncritically.

## The six groups

| `group` | who counts |
| --- | --- |
| `immigrant_refugee` | people who have moved to another country to live, whatever their papers: immigrants, migrants, refugees, asylum seekers, exiles fleeing persecution, expatriates, and families described as immigrant ("the son of Polish immigrants"). Tourists, travellers and visitors do not count. |
| `disability` | people with a lasting physical, sensory, intellectual or developmental disability: wheelchair users, amputees, blind, deaf or mute people, people with autism, Down syndrome, dwarfism or a learning disability, and people marked by a disfigurement. A short-lived injury does not count unless the summary makes it lasting or central. |
| `mental_illness` | people the summary presents as having a mental illness: a named condition (schizophrenia, bipolar disorder, depression, PTSD, an eating disorder...), a breakdown, psychiatric treatment, or a stay in a psychiatric hospital. Anger, grief or eccentricity is not mental illness. "Crazy", "mad" or "insane" used loosely ("a crazy plan", "drives him mad") is not a mention. |
| `older_people` | people described as old, elderly or retired, or aged about 65 or over ("an old man", "a retired couple", "a 78-year-old widow"). A parent or grandparent counts only when the summary marks their age. |
| `poverty_class` | people framed by being poor or working class: poor families, peasants, sharecroppers, beggars, people on welfare, a working-class neighbourhood's residents. "Poor" in the sense of pity ("the poor girl") is not a mention, nor is a job on its own (a waitress, a miner) unless the summary frames the person by class. |
| `homelessness` | people without a home: homeless people, people living on the streets, vagrants, hobos, street children, residents of a homeless shelter. A wandering stranger or "drifter" counts only when the summary says or clearly implies they have no home. |

## Fields

### 1. `social_groups`

One entry for **each** of the six groups the summary mentions or clearly identifies. Empty list when there is none.
One entry per group even if several of its members appear.

- `group`: one of the six values above.
- `portrayal`: `"neutral"`, `"stereotyped"` or `"counter_stereotyped"`.
- `types`: when stereotyped, every type that applies (empty list otherwise):

| type | applies when |
| --- | --- |
| `danger_threat` | the group is framed as violent, menacing or dangerous: the escaped mental patient on a killing spree, the dangerous vagrant |
| `criminality` | the group is tied to crime, drugs, gangs or theft as if typical: criminal immigrants, thieving poor |
| `illegitimacy` | immigrants framed as illegal, invaders, job-stealers or a burden on the country |
| `burden_dependence` | the group is framed as helpless, dependent or a burden on family or society |
| `pity_tragedy` | the person exists to be pitied; their condition is a tragedy that defines them |
| `inspiration` | a disabled, ill, old or poor person exists to inspire others or to be overcome ("overcoming" as the whole story) |
| `cure_or_death` | the story resolves the condition by a miracle cure or by the person's death, as if no life were possible with it |
| `villain_marker` | a disability, disfigurement or mental illness marks or explains a villain |
| `decline_incompetence` | older or disabled people framed as senile, frail, forgetful, confused, incapable or out of touch |
| `moral_failing` | poverty or homelessness framed as laziness, drunkenness, addiction, dirt or ignorance |
| `comic_figure` | a stock comic type: the grumpy old man, the pot-smoking granny, the comic tramp, the hillbilly fool |
| `wise_or_noble` | a positive stock type: the wise old mentor, the noble poor, the magical homeless sage, the savant |
| `victim_only` | the group exists in the plot only to suffer, with no aims of its own |
| `other` | another stereotype; explain in `notes` |

- `evidence`: the shortest quote from the summary (at most 15 words) that supports the judgement. Never quote a
  pejorative label; paraphrase instead.

### 2. `pejorative_label`

`true` when the summary itself calls a member of these groups by an insulting label: for example the common slurs
for disabled people (named in the copy given to the labelling model, left out of this public copy), "lunatic",
"madman", "maniac", "psycho", "nutcase", "psychopath" or "sociopath" used as a label,
"geezer", "old hag", "bum", "tramp" or "wino" for a homeless person, "white trash", "trailer trash", "redneck",
"hillbilly", "illegal alien" or "illegals". Words that brand a person as mad and menacing in one go (lunatic,
madman, maniac, psycho, psychopath) count whether or not the summary says anything else about mental illness. Never
repeat the label anywhere in the record; write "pejorative: <group>" in `notes` instead (for example
"pejorative: mental_illness").

An outdated name used descriptively in an older summary ("handicapped", "hobo", or an old clinical term for
intellectual disability in a 1960s synopsis) is not a pejorative label: set `pejorative_label` to `false` and write "outdated term" in `notes`.

### 3. `discrimination_theme`

- `present`: `true` when the story is *about* discrimination against one of these groups: hostility to immigrants or
  refugees, ableism, stigma against mental illness, ageism, classism or the treatment of homeless people, whatever its
  stance.
- `stance`: when present, `"critical"`, `"neutral"` or `"endorsing"`; otherwise `null`.

### 4. `confidence` and `notes`

- `confidence`: `1` (unsure, thin text), `2` (fairly sure) or `3` (sure).
- `notes`: optional, at most 20 words; required when a type is `other` and when `pejorative_label` is `true`.

## Rules for hard cases

These are settled in advance; apply them without stopping to ask.

- **Settings are not people.** A nursing home, homeless shelter, refugee camp, slum or psychiatric hospital is a
  setting. Add a group entry only when the summary identifies people as members ("residents of a nursing home",
  "a woman committed to an asylum", "refugees in the camp").
- **Individual versus group.** One member of a group doing something bad is not a stereotype by itself. Mark
  `stereotyped` when the summary attaches the group or the condition to the trait ("a schizophrenic killer", "a
  senile old fool", "drunken bums", "illegal immigrants smuggling drugs") or when the condition is offered as the
  explanation of the behaviour.
- **Villains labelled mad.** A killer described as a maniac, madman or psycho: set `pejorative_label` to `true`. Add
  a `mental_illness` entry (stereotyped, `danger_threat`) only when the summary presents the killer as mentally ill
  (a diagnosis, an escape from a psychiatric hospital, "psychotic", "deranged by illness").
- **Mad scientists, fools and eccentrics** are stock characters, not mental illness, unless the summary says
  otherwise.
- **Comedy is not an excuse:** a stock comic type played for laughs is `comic_figure`.
- **Surprise is not a counter-stereotype.** When the joke or hook is that an old, disabled or poor person does
  something unexpected ("a feisty 80-year-old robs a bank"), that is `comic_figure` unless the summary treats it as
  ordinary. `counter_stereotyped` is for a member of the group shown with ordinary competence and aims, without the
  hook depending on their group.
- **Historical settings:** judge how the summary frames people, not whether the period was poor or harsh. Medieval
  peasants doing ordinary things are `neutral`.
- **Overlap with the first pass:** a Mexican immigrant is labelled here as `immigrant_refugee` only; the nationality
  was labelled in the first pass.
- **Two groups at once:** a homeless veteran with PTSD gets two entries (`homelessness`, `mental_illness`).
- **When unsure** between `neutral` and `stereotyped`, choose `neutral` and set `confidence` to 1.
