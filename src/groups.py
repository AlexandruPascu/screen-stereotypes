"""Groups, the words that mention them, and the stereotype words attached to them.

Mentions are curated here. Stereotype words come from three places, each kept with its source:
  seegull    Google's SeeGULL (CC BY 4.0): (identity, attribute) pairs rated as stereotypes by annotators; we keep
             pairs with >= 3 stereotype votes and reduce each attribute phrase to its content words
             ("love pizza" -> pizza, "hard drinking" -> drinking).
  stereoset  StereoSet dev set (CC BY-SA 4.0): the word that turns a sentence into the stereotype for a target.
  curated    our own short lists for groups those sources miss or cover thinly (written for this project, documented
             here; extend or prune them freely).
Slurs come from Wikipedia's lists of ethnic and religious slurs (CC BY-SA 4.0), mapped to groups by their "targets"
column, plus HurtLex's homosexuality category (CC BY-NC-SA 4.0). They are built into DERIVED/lexicon.json by
stereotype_lexicon.py rather than written out here.
"""

GROUPS = {
    # nationality or ethnicity: (category, mention terms, SeeGULL identities, StereoSet targets, Wikipedia targets, curated)
    "italian": ("nationality", ["italian", "italians", "italian-american", "italian-americans", "sicilian", "sicilians"],
                ["Italian"], ["Italy"], ["Italian"],
                ["mafia", "mob", "mobster", "godfather", "pizza", "pasta", "spaghetti", "meatballs", "wine", "passionate",
                 "hot-tempered", "gangster", "capo", "consigliere", "omerta", "vendetta"]),
    "irish": ("nationality", ["irish", "irishman", "irishmen", "irishwoman", "irish-american"], ["Irish"], [], ["Irish"],
              ["drunk", "drunken", "whiskey", "pub", "brawl", "fighting", "potato", "leprechaun", "catholic", "cop",
               "temper", "ira"]),
    "mexican": ("nationality", ["mexican", "mexicans", "mexican-american", "mexican-americans", "chicano", "chicana"],
                ["Mexican"], [], ["Mexican"],
                ["cartel", "illegal", "border", "drug", "smuggler", "gardener", "maid", "bandit", "sombrero", "tequila",
                 "lazy", "gang", "immigrant", "undocumented"]),
    "russian": ("nationality", ["russian", "russians", "soviet", "soviets"], ["Russian"], ["Russian"], ["Russian"],
                ["vodka", "spy", "kgb", "mobster", "mafia", "communist", "cold", "ruthless", "oligarch", "drunk"]),
    "german": ("nationality", ["german", "germans"], ["German"], [], ["German"],
               ["nazi", "nazis", "efficient", "strict", "beer", "cold", "discipline", "punctual", "humorless"]),
    "french": ("nationality", ["french", "frenchman", "frenchmen", "frenchwoman"], ["French"], [], ["French"],
               ["wine", "romantic", "rude", "arrogant", "snob", "cheese", "seductive", "baguette", "chic", "affair"]),
    "british": ("nationality", ["british", "brit", "brits", "englishman", "englishmen", "englishwoman"],
                ["British", "English"], ["Britain"], ["British", "English"],
                ["tea", "posh", "polite", "reserved", "stiff", "aristocrat", "butler", "snobbish"]),
    "polish": ("nationality", ["polish"], ["Polish"], [], ["Polish"], ["plumber", "stupid", "laborer"]),
    "greek": ("nationality", ["greek", "greeks"], ["Greek"], [], ["Greek"], ["wedding", "family", "restaurant"]),
    "chinese": ("nationality", ["chinese", "chinese-american", "chinese-americans"], ["Chinese"], [], ["Chinese"],
                ["kung", "restaurant", "laundry", "takeout", "triad", "math", "communist", "opium", "dragon", "spy"]),
    "japanese": ("nationality", ["japanese", "japanese-american"], ["Japanese"], ["Japanese"], ["Japanese"],
                 ["samurai", "ninja", "geisha", "yakuza", "honor", "polite", "workaholic", "tourist", "camera"]),
    "korean": ("nationality", ["korean", "koreans", "korean-american"], ["Korean", "South Korean", "North Korean"], [],
               ["Korean"], ["store", "grocer", "kimchi", "nail"]),
    "south_asian": ("ethnicity", ["pakistani", "pakistanis", "bangladeshi", "sri lankan", "desi", "south asian"],
                    ["Indian", "Pakistani", "Bangladeshi"], ["Bangladesh", "Bengali"], ["South Asian", "Pakistani"],
                    ["arranged", "marriage", "curry", "doctor", "engineer", "shopkeeper", "taxi", "cab", "convenience",
                     "call", "spiritual", "guru"]),
    "arab": ("ethnicity", ["arab", "arabs", "arabian", "saudi", "saudis", "iraqi", "iraqis", "syrian", "syrians",
                           "lebanese", "palestinian", "palestinians", "yemeni", "jordanian"],
             ["Saudi Arabian", "Iraqi", "Syrian", "Lebanese", "Yemeni", "Jordanian", "Palestinian"],
             ["Arab", "Saudi Arabian", "Iraq", "Syria", "Yemen", "Jordan", "Lebanon"], ["Arab"],
             ["terrorist", "terrorists", "terrorism", "bomb", "bomber", "sheik", "sheikh", "oil", "desert", "camel",
              "harem", "jihad", "extremist", "veil", "fanatic"]),
    "iranian": ("nationality", ["iranian", "iranians", "persian", "persians"], ["Iranian"], ["Iranian", "Persian people"],
                ["Iranian", "Persian"], ["terrorist", "nuclear", "regime", "carpet", "fanatic"]),
    "turkish": ("nationality", ["turkish", "turks"], ["Turkish"], [], ["Turkish"], ["kebab", "barber"]),
    "african": ("ethnicity", ["african", "africans", "nigerian", "nigerians", "ghanaian", "kenyan", "ethiopian",
                              "somali", "somalis", "congolese", "sudanese", "ugandan"],
                ["Nigerian", "Ghanaian", "Kenyan", "Ethiopian", "Somali", "Congolese", "Sudanese", "Ugandan"],
                ["African", "Ethiopia", "Ethiopian", "Somalia", "Ghanaian", "Cameroon", "Eritrean", "Sierra Leon"],
                ["African", "Nigerian", "Somali"],
                ["tribe", "tribal", "warlord", "pirate", "poverty", "starving", "famine", "safari", "jungle", "witch",
                 "scam", "prince"]),
    "black": ("ethnicity", ["african american", "african-american", "african americans", "african-americans",
                            "black man", "black men", "black woman", "black women", "black people", "black family",
                            "black community", "black teen", "black teenager", "black kid", "black youth", "black girl",
                            "black boy", "black cop", "black neighborhood"], [], [], ["Black", "African American"],
              ["gang", "gangs", "ghetto", "hood", "drug", "drugs", "dealer", "thug", "crime", "criminal", "prison",
               "athlete", "basketball", "rapper", "hip-hop", "welfare", "slave", "slavery", "poverty"]),
    "latino": ("ethnicity", ["latino", "latinos", "latina", "latinas", "latinx", "hispanic", "hispanics", "puerto rican",
                             "puerto ricans", "cuban", "cubans", "dominican", "colombian", "colombians", "salvadoran"],
               ["Puerto Rican", "Cuban", "Dominican", "Colombian", "Salvadoran"], ["Hispanic", "Columbian", "Ecuador"],
               ["Hispanic", "Latino", "Puerto Rican", "Cuban"],
               ["gang", "cartel", "drug", "illegal", "maid", "housekeeper", "gardener", "spicy", "fiery", "passionate",
                "immigrant", "undocumented", "telenovela"]),
    "asian": ("ethnicity", ["asian", "asians", "asian-american", "asian-americans", "east asian", "vietnamese",
                            "filipino", "filipina", "thai", "cambodian", "laotian", "hmong"],
              ["Vietnamese", "Filipino", "Thai", "Cambodian", "Laotian"], ["Vietnam"], ["Asian", "East Asian", "Vietnamese"],
              ["kung", "karate", "martial", "math", "nerd", "geisha", "submissive", "exotic", "restaurant", "massage",
               "nail", "tiger", "model minority", "accent"]),
    "native_american": ("ethnicity", ["native american", "native americans", "american indian", "american indians",
                                      "apache", "comanche", "cherokee", "sioux", "navajo", "cheyenne", "mohawk",
                                      "indian chief", "indian tribe", "red indian", "indigenous"],
                        [], [], ["Native American"],
                        ["savage", "savages", "scalp", "scalped", "warpath", "tomahawk", "warrior", "chief", "tribe",
                         "reservation", "casino", "shaman", "spirit", "mystic", "alcoholic", "noble"]),
    "roma": ("ethnicity", ["gypsy", "gypsies", "romani"], [], [], ["Romani"],
             ["thief", "steal", "curse", "fortune", "teller", "caravan", "nomadic", "con", "palm"]),
    "white": ("ethnicity", ["white man", "white men", "white woman", "white women", "white people", "white family",
                            "white teen", "caucasian", "caucasians", "redneck", "rednecks", "hillbilly", "hillbillies"],
              [], [], ["White"], ["racist", "privileged", "trailer", "inbred", "rural", "suburban", "supremacist"]),
    # religion
    "jewish": ("religion", ["jew", "jews", "jewish", "hasidic", "orthodox jew", "orthodox jews"], [], [],
               ["Jewish", "Jews"], ["money", "greedy", "banker", "lawyer", "neurotic", "mother", "kosher", "rabbi",
                                    "holocaust", "rich", "stingy", "cheap"]),
    "muslim": ("religion", ["muslim", "muslims", "moslem", "islamic", "islamist"], [], ["Muslim"], ["Muslim"],
               ["terrorist", "terrorists", "terrorism", "jihad", "extremist", "radical", "bomb", "veil", "hijab",
                "burqa", "fanatic", "oppressed", "honor"]),
    "christian": ("religion", ["christians", "devout christian", "christian family", "christian school",
                               "christian fundamentalist", "born-again christian", "catholic", "catholics", "evangelical",
                               "evangelicals", "protestant", "mormon", "mormons", "baptist"], [], ["Bible"], ["Christian", "Catholic"],
                  ["fanatic", "hypocrite", "conservative", "zealot", "cult", "preacher", "abuse", "guilt", "nun",
                   "priest", "bigot"]),
    "hindu": ("religion", ["hindu", "hindus", "brahmin", "brahmins"], [], ["Brahmin"], ["Hindu"],
              ["caste", "cow", "guru", "spiritual", "superstitious", "temple", "arranged"]),
    # sexuality and gender identity
    "gay": ("sexuality", ["gay", "gays", "homosexual", "homosexuals", "gay man", "gay men", "queer"], [], [], [],
            ["flamboyant", "fashion", "hairdresser", "designer", "hiv", "promiscuous", "effeminate", "closeted",
             "closet", "drag", "musical", "best friend", "camp"]),
    "lesbian": ("sexuality", ["lesbian", "lesbians"], [], [], [],
                ["butch", "tomboy", "man-hating", "predatory", "seduce", "feminist"]),
    "trans": ("gender identity", ["transgender", "transsexual", "trans woman", "trans women", "trans man", "trans men",
                                  "transvestite", "cross-dresser"], [], [], [],
              ["deceive", "deception", "secret", "freak", "drag", "prostitute", "murdered", "surgery", "disguise"]),
}

# Words written with a capital letter when they name the group ("Italian", "Native American"); other mention terms
# ("gay", "transgender", "black man", "indigenous") are matched lower-case or with a sentence-initial capital.
LOWERCASE_OK = {"gay", "gays", "homosexual", "homosexuals", "gay man", "gay men", "queer", "lesbian", "lesbians",
                "transgender", "transsexual", "trans woman", "trans women", "trans man", "trans men", "transvestite",
                "cross-dresser", "indigenous", "redneck", "rednecks", "hillbilly", "hillbillies", "caucasian",
                "caucasians"}
LOWERCASE_OK |= {f"{c} {n}" for c in ("black", "white") for n in ("man", "men", "woman", "women", "people", "family",
                 "community", "teen", "teenager", "kid", "youth", "girl", "boy", "cop", "neighborhood")}
ALWAYS_CAPITAL = {"american", "americans", "indian", "indians", "native", "asian", "african", "east", "south", "sri", "christian",
                  "lankan", "puerto", "rican", "ricans", "orthodox", "red", "african-american", "african-americans"}


def cased_variants(term, category_proper):
    """How a mention term is written in running text."""
    if term in LOWERCASE_OK or not category_proper:
        return {term, term[0].upper() + term[1:]}
    words = term.split()
    proper = {w for w in words if len(words) == 1 or w in ALWAYS_CAPITAL or w in SINGLE}
    return {" ".join("-".join(p[:1].upper() + p[1:] for p in w.split("-")) if w in proper else w for w in words)}


# Mentions that are not about the group in context ("indian" is left out on purpose: in westerns it means Native
# Americans and elsewhere people from India; "french fries", "black market"...).
NOT_A_MENTION = [r"french fries", r"french toast", r"french kiss", r"german shepherd", r"swiss cheese",
                 r"turkish delight", r"chinese food", r"chinese restaurant", r"chinese takeout", r"gay paree",
                 r"roman catholic church", r"white house", r"white collar", r"honky[- ]tonks?",
                 r"fresh off the boat"]


SINGLE = {t for g in GROUPS.values() for t in g[1] if " " not in t}
