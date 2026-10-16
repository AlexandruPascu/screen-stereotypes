"""Social groups beyond nationality, ethnicity, religion and sexuality: immigrants and refugees, disabled people, people
with a mental illness, older people, poor and working-class people, and homeless people.

Kept apart from groups.py so the 30 groups of study 5 and their published numbers stay as they are. Per group:
  mentions     words that name people as members of the group (not settings or topics); identity terms cross-checked
               against HolisticBias's descriptors (Smith et al. 2022, CC BY-SA 4.0); older outdated terms are included
               because older summaries use them
  pejoratives  insulting labels for the group, counted the way slurs are counted for the other groups but reported
               separately; like the slur lists they are not published: they are read from the private file
               LEXICONS/social_pejoratives.json (with the idioms that contain them and one outdated mention term that
               is also an insult), and without it the scan simply counts no pejorative labels
  stereotypes  words that carry a stereotype about the group, by source:
                 bbq      the stereotypes BBQ documents for age, disability and socio-economic status (Parrish et al.
                          2022, CC BY 4.0), turned into the words a plot summary would use
                 curated  written for this project from the media-portrayal literature on each group (the violent
                          mental patient, the pitiable or inspirational disabled character, the senile or grumpy old
                          person, the criminal or lazy poor, the drunk or dangerous homeless person, the illegal or
                          criminal immigrant); extend or prune freely
               words also marked pejorative feed the pejorative index, as for the other groups
Mental illness has no HolisticBias or BBQ descriptor list of its own beyond BBQ's "mentally-ill" items, so its mentions
are curated. Built into DERIVED/lexicon_social.json by social_lexicon.py.
"""

import json

from paths import LEXICONS

UNITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
# spelled-out ages of 65 and over ("an eighty-year-old widow"); numeric ones ("a 72-year-old") are rewritten to
# AGE_TOKEN before matching (group_stereotypes.py), since the word matcher skips digits
AGE_TOKEN = "aged-sixty-five-plus"
SPELLED_AGES = ([f"sixty-{u}-year-old" for u in UNITS[4:]]
                + [f"{t}-year-old" for t in ("seventy", "eighty", "ninety")]
                + [f"{t}-{u}-year-old" for t in ("seventy", "eighty", "ninety") for u in UNITS])

SOCIAL_GROUPS = {
    # name: (category, mentions, stereotype words by source, pejorative stereotype words)
    "immigrant_refugee": (
        "migration",
        ["immigrant", "immigrants", "emigrant", "emigrants", "emigre", "emigres", "migrant",
         "migrants", "migrant worker", "migrant workers", "refugee", "refugees", "asylum seeker", "asylum seekers",
         "asylum-seeker", "asylum-seekers", "asylum-seeking", "undocumented", "illegal immigrant", "illegal immigrants",
         "foreign-born", "stateless", "displaced persons", "deportee", "deportees", "guest worker", "guest workers",
         "boat people", "expat", "expats", "expatriate", "expatriates", "newly arrived immigrant"],
        {"curated": ["illegal", "illegally", "smuggled", "smuggling", "smuggler", "smugglers", "border", "deported",
                     "deport", "deportation", "criminal", "criminals", "crime", "gang", "gangs", "cartel", "drugs",
                     "steal", "jobs", "welfare", "cheap labor", "cheap labour", "sweatshop", "exploited",
                     "exploitation", "trafficked", "trafficking", "poor", "poverty", "accent", "broken english",
                     "language barrier", "assimilate", "assimilation", "culture clash", "arranged marriage",
                     "green card", "sham marriage", "marriage of convenience", "terrorist", "terrorists", "invasion",
                     "threat", "fresh off the boat"]},
        {"illegal", "illegally", "criminal", "criminals", "crime", "gang", "gangs", "cartel", "drugs", "steal",
         "terrorist", "terrorists", "invasion", "threat", "smuggler", "smugglers"},
    ),
    "disability": (
        "disability",
        ["disabled", "disability", "disabilities", "handicapped", "deaf", "deaf-mute", "hard of hearing",
         "hard-of-hearing", "hearing-impaired", "blind", "visually impaired", "vision-impaired", "partially sighted",
         "autistic", "autism", "asperger's", "dyslexic", "learning disability", "learning disabilities",
         "learning-disabled", "intellectual disability", "developmentally disabled", "special needs",
         "mentally handicapped", "mentally disabled", "mentally challenged", "physically challenged",
         "paraplegic", "quadriplegic", "paralyzed", "paralysed", "wheelchair-bound", "wheelchair user",
         "in a wheelchair", "amputee", "amputees", "mute", "non-verbal", "cerebral palsy", "down syndrome",
         "down's syndrome", "dwarfism", "disfigured", "disfigurement", "deformed", "deformity"],
        {"bbq": ["unable", "helpless", "dependent", "unemployed", "lonely", "friendless", "isolated", "withdrawn",
                 "weird", "strange", "awkward", "unhappy", "miserable", "incapable"],
         "curated": ["pity", "pitiful", "pitied", "tragic", "tragedy", "brave", "courageous", "inspiring",
                     "inspiration", "inspirational", "overcome", "overcomes", "overcoming", "triumph", "miracle",
                     "miraculous", "cure", "cured", "heal", "healed", "bitter", "resentful", "burden",
                     "confined", "suffers", "afflicted", "stricken", "invalid", "freak", "freaks", "freak show",
                     "monster", "monstrous", "grotesque", "hideous", "savant", "childlike", "simple-minded",
                     "sheltered", "institutionalized", "institutionalised"]},
        {"helpless", "freak", "freaks", "freak show", "monster", "monstrous", "grotesque", "hideous", "burden",
         "pitiful", "invalid", "simple-minded", "weird"},
    ),
    "mental_illness": (
        "mental health",
        ["mentally ill", "mental illness", "mental illnesses", "mental disorder", "mental health problems",
         "mental health issues", "schizophrenic", "schizophrenia", "bipolar", "bipolar disorder", "manic depressive",
         "manic-depressive", "manic depression", "psychotic", "psychosis", "mental patient", "mental patients",
         "psychiatric patient", "psychiatric patients", "mental breakdown", "nervous breakdown", "clinical depression",
         "clinically depressed", "severe depression", "post-traumatic stress", "ptsd", "shell-shocked", "shell shock",
         "ocd", "obsessive-compulsive", "anorexia", "anorexic", "bulimia", "bulimic", "eating disorder",
         "dissociative identity disorder", "multiple personality disorder", "split personality", "agoraphobic",
         "agoraphobia", "mental hospital", "psychiatric hospital", "mental institution", "insane asylum",
         "lunatic asylum", "mental asylum", "psychiatric ward", "psych ward"],
        {"bbq": ["violent", "violence", "dangerous", "unstable", "unpredictable"],
         "curated": ["killer", "killing", "murder", "murders", "murderer", "murderous", "homicidal", "rampage",
                     "slasher", "stalker", "stalks", "obsessed", "obsession", "deranged", "unhinged", "crazed",
                     "berserk", "madness", "insanity", "insane", "delusional", "delusions", "hallucinations",
                     "voices", "hearing voices", "paranoid", "paranoia", "possessed", "escaped", "escapes",
                     "escapee", "asylum", "straitjacket", "padded cell", "institutionalized", "institutionalised",
                     "lobotomy", "electroshock", "shock therapy", "sedated", "menace", "terrorizes", "terrorises",
                     "hysterical", "hysteria", "mad genius", "tormented"]},
        {"violent", "violence", "dangerous", "killer", "murderous", "homicidal", "rampage", "deranged", "unhinged",
         "crazed", "berserk", "menace", "terrorizes", "terrorises", "hysterical", "straitjacket", "padded cell"},
    ),
    "older_people": (
        "age",
        ["elderly", "old man", "old men", "old woman", "old women", "old lady", "old ladies", "old couple",
         "old people", "old folks", "old-timer", "old-timers", "senior citizen", "senior citizens", "senior-citizen",
         "pensioner", "pensioners", "retiree", "retirees", "retired couple", "octogenarian", "octogenarians",
         "nonagenarian", "nonagenarians", "septuagenarian", "septuagenarians", "centenarian", "centenarians",
         "geriatric", "aging father", "aging mother", "aging parents", "ageing father", "ageing mother",
         "ageing parents", "seventy-something", "eighty-something", "ninety-something", AGE_TOKEN] + SPELLED_AGES,
        {"bbq": ["forgetful", "forgets", "senile", "senility", "dementia", "alzheimer's", "confused", "frail",
                 "feeble", "slow", "stubborn", "set in his ways", "set in her ways", "out of touch", "technology",
                 "old-fashioned", "hard of hearing"],
         "curated": ["grumpy", "cranky", "crotchety", "curmudgeon", "curmudgeonly", "cantankerous", "irascible",
                     "lonely", "loneliness", "widowed", "dying", "decline", "ailing", "sickly", "infirm", "burden",
                     "nursing home", "retirement home", "care home", "old folks home", "cane", "decrepit",
                     "doddering", "eccentric", "wise", "wisdom", "spry", "feisty", "sprightly", "spunky",
                     "inheritance", "pension", "lecherous"]},
        {"senile", "senility", "decrepit", "doddering", "burden", "feeble", "lecherous", "grumpy", "cranky",
         "crotchety", "cantankerous"},
    ),
    "poverty_class": (
        "class",
        ["poor family", "poor families", "poor farmer", "poor farmers", "poor people", "poor peasant",
         "poor peasants", "poor children", "poor kids", "poor villagers", "poor workers", "poor folks",
         "poor household", "poor households", "poor couple", "poor widow", "poor orphan", "poor fisherman",
         "poor fishermen", "poor tenants", "impoverished", "penniless", "destitute", "poverty-stricken", "poverty",
         "low-income", "working-class", "working class", "blue-collar", "blue collar", "underclass", "lower class",
         "lower-class", "peasant", "peasants", "sharecropper", "sharecroppers", "pauper", "paupers", "beggar",
         "beggars", "on welfare", "welfare mother", "welfare mothers"],
        {"bbq": ["drugs", "drug dealer", "lazy", "laziness", "incompetent", "stupid", "ignorant", "uneducated",
                 "illiterate", "dropout", "neglect", "neglectful", "abusive"],
         "curated": ["crime", "criminal", "criminals", "thief", "thieves", "steal", "stealing", "robbery", "gang",
                     "gangs", "violent", "violence", "drunk", "drunken", "alcoholic", "addict", "addicted",
                     "addiction", "dirty", "filthy", "squalor", "squalid", "trashy", "crude", "vulgar", "uncouth",
                     "welfare", "food stamps", "ghetto", "slum", "slums", "tenement", "trailer park", "the projects",
                     "housing project", "council estate", "teen pregnancy", "noble", "honest", "hardworking",
                     "hard-working", "humble", "simple folk", "freeloader", "scrounging"]},
        {"drugs", "drug dealer", "lazy", "laziness", "incompetent", "stupid", "ignorant", "uneducated", "illiterate",
         "neglectful", "abusive", "crime", "criminal", "criminals", "thief", "thieves", "steal", "stealing", "gang",
         "gangs", "violent", "violence", "drunk", "drunken", "alcoholic", "addict", "dirty", "filthy", "squalid",
         "trashy", "crude", "vulgar", "uncouth", "freeloader", "scrounging"},
    ),
    "homelessness": (
        "housing",
        ["homeless", "homelessness", "unhoused", "houseless", "vagrant", "vagrants", "vagrancy", "vagabond",
         "vagabonds", "hobo", "hobos", "hoboes", "living on the streets", "lives on the streets", "living on the street",
         "lives on the street", "street people", "rough sleeper", "rough sleepers", "sleeping rough", "panhandler",
         "panhandlers", "street kid", "street kids", "street children", "street urchin", "street urchins"],
        {"curated": ["drunk", "drunkard", "drunken", "alcoholic", "booze", "addict", "drugs", "dirty", "filthy",
                     "smelly", "stench", "rags", "ragged", "unkempt", "disheveled", "dishevelled", "crazy", "insane",
                     "mentally ill", "dangerous", "violent", "threat", "beg", "begging", "begs", "panhandling",
                     "lazy", "criminal", "crime", "thief", "steal", "stealing", "shelter", "soup kitchen", "cardboard",
                     "alley", "alleyway", "skid row", "down-and-out", "down and out", "down on his luck",
                     "down on her luck", "wise", "pity", "invisible"]},
        {"drunk", "drunkard", "drunken", "alcoholic", "addict", "drugs", "dirty", "filthy", "smelly", "stench",
         "crazy", "insane", "dangerous", "violent", "threat", "lazy", "criminal", "crime", "thief", "steal",
         "stealing"},
    ),
}

# Phrases where a mention or label word does not refer to the group, removed before matching (case-insensitive)
NOT_A_MENTION = [
    r"blind (?:dates?|spots?|eye|faith|rage|side|trust|ambition|loyalty|obedience|luck|alley|drunk|panic|fury|"
    r"jealousy|love|auditions?|tasting|items?)", r"double[- ]blind", r"colou?r[- ]blind", r"snow[- ]blind",
    r"love is blind", r"blind to", r"robbed blind", r"deaf ears?", r"deaf to", r"tone[- ]deaf", r"on mute",
    r"mute button", r"(?:remains?|remained|stays?|stayed|falls?|fell|goes|went) mute",
    r"paraly[sz]ed (?:with|by) (?:fear|terror|indecision|grief|guilt)", r"disabled (?:the|a|an|his|her|their|its|all|both)",
    r"old money", r"poverty row", r"the poor thing", r"poor thing", r"working class hero",
]

_private = LEXICONS / "social_pejoratives.json"
if _private.exists():
    _p = json.load(open(_private))
    PEJORATIVES = _p["pejoratives"]
    NOT_A_MENTION += _p["remove_phrases"]
    for _g, _terms in _p["extra_mentions"].items():
        SOCIAL_GROUPS[_g][1].extend(_terms)
else:
    PEJORATIVES = {g: [] for g in SOCIAL_GROUPS}
