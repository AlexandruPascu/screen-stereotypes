"""Occupations recognised in character names, e.g. "Nurse", "Detective Morales", "Dr. Patel", "Bartender #2".

Only words that name the job without naming a gender are used for the main measure, so the gender comes from who
was cast, not from the word. Gendered job words (waitress, policeman, chairman...) are listed separately: their own
trend is a different question (does the script choose a gendered word at all?).

Each occupation: (name, group, regex alternatives matched as whole words in the lower-cased character name).
Groups follow the broad US Census / SOC occupation families so they can later be compared with workforce data.
"""

import re

OCCUPATIONS = [
    # health
    ("doctor", "health", [r"doctor", r"physician", r"medic"]),
    ("surgeon", "health", [r"surgeon"]),
    ("nurse", "health", [r"nurse"]),
    ("paramedic", "health", [r"paramedic", r"emt"]),
    ("psychiatrist", "health", [r"psychiatrist", r"psychologist", r"therapist", r"shrink"]),
    ("dentist", "health", [r"dentist"]),
    ("pharmacist", "health", [r"pharmacist"]),
    ("coroner", "health", [r"coroner", r"medical examiner", r"pathologist"]),
    # science, tech, engineering
    ("scientist", "science", [r"scientist", r"researcher", r"chemist", r"physicist", r"biologist"]),
    ("engineer", "science", [r"engineer"]),
    ("programmer", "science", [r"programmer", r"hacker", r"developer", r"computer technician", r"it guy"]),
    ("astronaut", "science", [r"astronaut"]),
    ("architect", "science", [r"architect"]),
    # education
    ("teacher", "education", [r"teacher", r"tutor"]),
    ("professor", "education", [r"professor", r"lecturer"]),
    ("principal", "education", [r"principal", r"headteacher"]),
    ("librarian", "education", [r"librarian"]),
    # legal
    ("lawyer", "legal", [r"lawyer", r"attorney", r"solicitor", r"barrister", r"defense counsel", r"public defender"]),
    ("prosecutor", "legal", [r"prosecutor", r"district attorney", r"\bda\b"]),
    ("judge", "legal", [r"judge", r"magistrate", r"justice of the peace"]),
    # protective services
    ("detective", "protective", [r"detective", r"det\.?", r"inspector", r"investigator", r"private eye"]),
    ("police officer", "protective", [r"police officer", r"police", r"cop", r"patrol officer", r"constable", r"trooper",
                                      r"deputy", r"sheriff", r"officer"]),
    ("federal agent", "protective", [r"fbi agent", r"cia agent", r"federal agent", r"special agent", r"agent"]),
    ("firefighter", "protective", [r"firefighter"]),
    ("security guard", "protective", [r"security guard", r"guard", r"bodyguard", r"bouncer", r"security"]),
    ("soldier", "military", [r"soldier", r"marine", r"private", r"corporal", r"infantryman", r"coast guard"]),
    ("sergeant", "ambiguous", [r"sergeant", r"sgt\.?"]),  # army or police: kept apart rather than guessed
    ("pilot", "transport", [r"pilot", r"aviator"]),
    ("flight attendant", "transport", [r"flight attendant", r"cabin crew"]),
    # business, office, management
    ("ceo", "management", [r"ceo", r"chief executive", r"executive", r"tycoon", r"entrepreneur"]),
    ("boss", "management", [r"boss", r"manager", r"supervisor", r"foreman"]),
    ("banker", "business", [r"banker", r"accountant", r"stockbroker", r"broker", r"financier", r"investor"]),
    ("secretary", "office", [r"secretary"]),
    ("receptionist", "office", [r"receptionist"]),
    ("assistant", "office", [r"assistant", r"aide", r"intern"]),
    ("clerk", "office", [r"clerk", r"cashier", r"teller"]),
    # politics
    ("politician", "politics", [r"politician", r"senator", r"mayor", r"governor", r"congressperson", r"minister of",
                                r"prime minister", r"councillor", r"councilor", r"ambassador", r"diplomat"]),
    # media and arts
    ("reporter", "media", [r"reporter", r"journalist", r"news anchor", r"anchor", r"newscaster", r"correspondent"]),
    ("photographer", "media", [r"photographer", r"cameraman"]),
    ("singer", "arts", [r"singer"]),
    ("dancer", "arts", [r"dancer"]),
    ("model", "arts", [r"model"]),
    ("artist", "arts", [r"artist", r"painter", r"sculptor"]),
    ("writer", "arts", [r"writer", r"author", r"novelist", r"poet"]),
    # food, hospitality, personal service
    ("bartender", "hospitality", [r"bartender", r"barkeep", r"barista"]),
    ("chef", "hospitality", [r"chef", r"cook", r"baker"]),
    ("housekeeper", "domestic", [r"housekeeper", r"maid", r"servant"]),
    ("janitor", "domestic", [r"janitor", r"cleaner", r"custodian", r"caretaker"]),
    ("nanny", "domestic", [r"nanny", r"babysitter", r"au pair"]),
    ("hairdresser", "personal", [r"hairdresser", r"hair stylist", r"stylist", r"barber", r"beautician"]),
    # trades and transport
    ("mechanic", "trades", [r"mechanic", r"plumber", r"electrician", r"carpenter", r"builder", r"construction worker"]),
    ("driver", "transport", [r"driver", r"chauffeur", r"cabbie", r"trucker"]),
    ("farmer", "agriculture", [r"farmer", r"rancher"]),
    # religion
    ("clergy", "religion", [r"priest", r"pastor", r"reverend", r"preacher", r"vicar", r"rabbi", r"imam", r"chaplain"]),
    # crime and sex work (stereotyped roles rather than census occupations)
    ("gangster", "crime", [r"gangster", r"mobster", r"thug", r"henchman", r"hitman", r"hit man", r"goon"]),
    ("thief", "crime", [r"thief", r"burglar", r"robber", r"pickpocket", r"mugger"]),
    ("drug dealer", "crime", [r"drug dealer", r"dope dealer", r"pusher"]),
    ("terrorist", "crime", [r"terrorist"]),
    ("killer", "crime", [r"killer", r"murderer", r"assassin"]),
    ("prisoner", "crime", [r"prisoner", r"inmate", r"convict"]),
    ("sex worker", "sex work", [r"prostitute", r"hooker", r"call girl", r"escort", r"streetwalker", r"sex worker"]),
    ("stripper", "sex work", [r"stripper", r"exotic dancer"]),
]

# Job words that carry a gender themselves. Tracked separately; never counted in OCCUPATIONS.
GENDERED = {
    "waiter": "m", "waitress": "f", "policeman": "m", "policewoman": "f", "fireman": "m", "businessman": "m",
    "businesswoman": "f", "salesman": "m", "saleswoman": "f", "chairman": "m", "chairwoman": "f", "congressman": "m",
    "congresswoman": "f", "stewardess": "f", "steward": "m", "hostess": "f", "headmaster": "m", "headmistress": "f",
    "landlord": "m", "landlady": "f", "seamstress": "f", "governess": "f", "butler": "m", "doorman": "m",
    "cameraman": "m", "barman": "m", "cleaning lady": "f", "maid of honor": "f", "barmaid": "f", "nun": "f", "monk": "m", "priestess": "f", "housewife": "f",
    "milkman": "m", "postman": "m", "mailman": "m", "fisherman": "m", "anchorman": "m", "anchorwoman": "f",
}

# Words that would otherwise match but are not jobs in context.
NOT_A_JOB = [
    r"model citizen", r"role model", r"agent orange", r"agent of", r"double agent", r"doc ock", r"guard dog",
    r"secretary bird", r"principal's", r"private parts",
    r"executive producer", r"guardian", r"security camera", r"officer's wife", r"officer's daughter",
    r"cook's", r"the doctor's", r"doctor's wife", r"doctor's daughter", r"doctor's son", r"nurse's",
    r"real estate agent", r"estate agent", r"travel agent", r"talent agent", r"insurance agent", r"literary agent",
    r"booking agent", r"ticket agent", r"press agent", r"sports agent", r"casting agent", r"principal dancer",
    r"art dealer", r"car dealer", r"card dealer", r"[a-z]+'s agent", r"civil servant", r"maid of hono[u]?r",
    r"chef de gare", r"public servant",
]

# Job words that are also common surnames: counted only alone, as "the <job>", or after a modifier that is not a
# first name ("head chef", "village baker"), never as "<first name> <word>" (John Baker, Tom Baker).
SURNAME_PRONE = {"baker", "cook", "carpenter", "barber", "farmer", "singer", "painter", "porter", "builder", "foreman",
                 "guard", "principal"}
MODIFIERS = {"the", "a", "an", "head", "chief", "pastry", "sous", "army", "ship's", "ship", "school", "prison", "camp",
             "restaurant", "hotel", "village", "castle", "palace", "royal", "master", "assistant", "line", "fry",
             "short-order", "cafeteria", "kitchen", "pizza", "sushi", "town", "local", "old", "young", "first", "second",
             "third", "fourth", "night", "day", "security", "lounge", "wedding", "opera", "folk", "pop", "lead", "backup",
             "jazz", "blues", "country", "nightclub", "club", "bar", "street", "house", "portrait", "sign", "court",
             "tenant", "dirt", "dairy", "pig", "chicken", "cattle", "rice", "tea", "fat", "drunk", "female", "male",
             "temple", "museum", "bank", "border", "gate", "body", "coast", "national", "castle", "tower", "jail",
             "home", "hairdresser", "beauty", "prison", "vice", "deputy", "new", "school's", "acting"}
POLICE_OFFICER_CONTEXT = r"police|patrol|prison|parole|probation|correctional|corrections|traffic|duty|arresting|" \
                         r"uniformed|desk|beat|transit|highway|state|county|city|nypd|lapd|cpd|metro|motorcycle|k-9|" \
                         r"customs|immigration|border|court"

# Context words that move a match to another occupation (applied after the main match).
MILITARY_CONTEXT = (r"army|navy|naval|military|nazi|ss|wehrmacht|soviet|confederate|union|royal|air force|luftwaffe|"
                    r"u-boat|german|japanese|british|french|italian|russian|american|enemy|korean|chinese|roman|"
                    r"turkish|polish|spanish|imperial|rebel|cavalry|infantry|artillery")
CRIME_CONTEXT = r"mob|mafia|crime|gang|cartel|yakuza|triad|drug"


def alternatives():
    """All occupation words as one regex alternation, longest first so multi-word terms win."""
    words = sorted({w for _, _, ws in OCCUPATIONS for w in ws}, key=len, reverse=True)
    return "|".join(words)


_WORDS = sorted(((w, name) for name, _, ws in OCCUPATIONS for w in ws), key=lambda x: len(x[0]), reverse=True)
_ANY = re.compile(r"\b(" + "|".join(w for w, _ in _WORDS) + r")\b")
_TO_NAME = [(re.compile(r"^(?:" + w + r")$"), name) for w, name in _WORDS]
_NOT = re.compile(r"\b(" + "|".join(NOT_A_JOB) + r")\b")
_MIL = re.compile(r"\b(" + MILITARY_CONTEXT + r")\b")
_CRIME = re.compile(r"\b(" + CRIME_CONTEXT + r")\b")
_DR = re.compile(r"^(dr\.?|doctor)\s+[a-z]")
GROUP = {name: group for name, group, _ in OCCUPATIONS}


def occupation(name):
    """The occupation in a normalised character name, or None. The last job word wins ("police detective" ->
    detective, "doctor's assistant" -> assistant); a longer phrase wins over a shorter one ending at the same place."""
    if _NOT.search(name):
        name = _NOT.sub(" ", name)
    best = None
    for m in _ANY.finditer(name):
        if best is None or m.end() > best.end() or (m.end() == best.end() and len(m.group()) > len(best.group())):
            best = m
    if best is None:
        return "dr. title" if re.match(r"^dr\.?\s+[a-z]", name) else None
    word = best.group()
    found = next(n for p, n in _TO_NAME if p.match(word))
    if word in SURNAME_PRONE:
        m = re.match(r"^([a-z'.\-]+) " + re.escape(word) + r"$", name)
        if m and m.group(1) not in MODIFIERS:
            return None
    if word == "officer" and not (name.startswith("officer") or re.search(r"\b(" + POLICE_OFFICER_CONTEXT + r")\b", name)
                                  or _MIL.search(name)):
        return None
    if found in ("soldier", "sergeant") and re.search(r"\b(det\.?|detective|police|cop|constable|desk)\b", name):
        return "detective" if re.search(r"\b(det\.?|detective)\b", name) else "police officer"
    if found in ("police officer", "security guard", "sergeant") and _MIL.search(name):
        return "soldier"
    if found in ("boss", "ceo") and _CRIME.search(name):
        return "gangster"
    return found


GROUP["dr. title"] = "title"
