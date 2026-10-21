"""What characters do in plot summaries, by gender: who acts, who is acted upon, and how each is described.

Input: the parse tuples (parse_texts.py). A mention's gender comes from, in order:
  pronouns (she/her, he/him), gendered nouns and honorifics (mother, wife, Mrs., Queen...), and for names the title's
  own cast list: a first or last name that, among that film's or series' credited characters, belongs only to roles
  played by actresses (or only by actors).
Outputs (aggregates only) to results/agency/:
  agency_by_decade.csv     how often women and men are the subject of an active verb vs passive subject / object
  distinctive_words.csv    verbs and descriptions most associated with women vs men (log-odds ratio with an
                           informative Dirichlet prior, Monroe et al. 2008), per era
  categories_by_decade.csv women's share of mentions in curated word categories (appearance, intellect, power,
                           violence done, violence suffered...), against women's share of all gendered mentions
"""
import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from publish import without_slurs
from paths import DERIVED, IMDB, RESULTS

OUT = RESULTS / "agency"
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit = '8GB'; SET preserve_insertion_order = false")

FEMALE = """she her herself woman women girl girls lady ladies mother mothers mom moms mum mommy mama wife wives daughter
daughters sister sisters aunt aunts grandmother grandmothers grandma granny niece nieces girlfriend girlfriends bride
brides fiancee fiancée widow widows queen queens princess princesses heroine madam mrs mrs. miss ms ms. nun nuns
stepmother stepdaughter stepsister housewife housewives matriarch landlady businesswoman policewoman schoolgirl
schoolgirls duchess empress countess baroness goddess heiress mistress hostess waitress stewardess actress""".split()
MALE = """he him himself man men boy boys guy guys gentleman gentlemen father fathers dad dads daddy papa husband husbands
son sons brother brothers uncle uncles grandfather grandfathers grandpa granddad nephew nephews boyfriend boyfriends
groom grooms fiance fiancé widower king kings prince princes sir mr mr. monk monks stepfather stepson stepbrother
patriarch landlord businessman policeman schoolboy schoolboys lad lads duke emperor baron waiter""".split()
NOT_NAMES = """the a an young old little big dr dr. doctor mr mr. mrs mrs. ms ms. miss detective det det. officer agent
captain capt capt. sergeant sgt sgt. lieutenant lt lt. professor prof judge nurse father mother sister brother uncle
aunt king queen prince princess lord lady sir madam mister man woman boy girl guy kid the""".split()
CATEGORIES = {
    "appearance (desc)": ("desc", "beautiful pretty attractive sexy gorgeous stunning lovely cute hot glamorous seductive "
                                  "blonde slim thin fat ugly handsome elegant"),
    "intellect (desc)": ("desc", "smart brilliant intelligent genius clever wise talented gifted"),
    "power (desc)": ("desc", "powerful rich wealthy ruthless dangerous strong ambitious influential famous"),
    "emotion (subj)": ("subj", "cry weep sob worry panic fear despair grieve mourn"),
    "violence done (subj)": ("subj", "kill murder shoot stab attack beat fight strangle poison"),
    "violence suffered (pass)": ("pass", "kill murder kidnap rape attack abduct assault stalk strangle stab shoot beat "
                                         "torture"),
    "rescue done (subj)": ("subj", "rescue save protect defend"),
    "rescued (pass)": ("pass", "rescue save protect"),
    "investigation (subj)": ("subj", "investigate solve uncover track discover suspect interrogate"),
    "romance (subj)": ("subj", "marry date kiss seduce flirt fall love"),
    "career (subj)": ("subj", "work hire promote lead manage run build launch negotiate win"),
    "care (subj)": ("subj", "care cook clean babysit nurse feed comfort raise"),
}

con.execute(f"""CREATE TEMP TABLE gendered_words AS
    SELECT unnest({FEMALE}) AS word, true AS female UNION ALL SELECT unnest({MALE}), false""")
# title -> work (the film or series whose cast list names the characters)
con.execute(f"""CREATE TEMP TABLE title_work AS
    SELECT tconst AS title, parentTconst AS work FROM read_csv('{IMDB / "title.episode.tsv.gz"}', delim='\\t',
        header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','parentTconst':'VARCHAR',
        'seasonNumber':'VARCHAR','episodeNumber':'VARCHAR'}})""")
con.execute(f"""CREATE TEMP TABLE name_gender AS
    SELECT work, token, bool_and(female) AS female FROM (
        SELECT work, female, unnest([split_part(name, ' ', 1), split_part(name, ' ', -1)]) AS token
        FROM '{DERIVED / "roles.parquet"}')
    WHERE length(token) > 1 AND token NOT IN (SELECT unnest({NOT_NAMES}))
    GROUP BY work, token HAVING bool_and(female) = bool_or(female)""")
print(f"name -> gender pairs: {con.execute('SELECT COUNT(*) FROM name_gender').fetchone()[0]:,}", flush=True)
con.execute(f"""CREATE TEMP TABLE meta AS
    SELECT tconst AS title, TRY_CAST(startYear AS INT) AS year, titleType FROM read_csv('{IMDB / "title.basics.tsv.gz"}',
        delim='\\t', header=true, quote='', escape='', nullstr='\\N', columns={{'tconst':'VARCHAR','titleType':'VARCHAR',
        'primaryTitle':'VARCHAR','originalTitle':'VARCHAR','isAdult':'VARCHAR','startYear':'VARCHAR','endYear':'VARCHAR',
        'runtimeMinutes':'VARCHAR','genres':'VARCHAR'}})
    UNION ALL SELECT 'tmdb:' || series || ':' || season || ':' || episode, year, 'tvEpisode'
    FROM '{DERIVED / "tmdb_episodes.parquet"}'""")
con.execute(f"""CREATE TEMP TABLE tmdb_series_imdb AS SELECT tmdb_id, imdb FROM '{DERIVED / "tmdb_titles.parquet"}'
    WHERE kind = 'series'""")

con.execute(f"""CREATE TEMP TABLE mentions AS
    WITH p AS (SELECT *, TRY_CAST(CASE WHEN title LIKE 'tmdb:%' THEN split_part(title, ':', 2) END AS INT) AS tmdb_sid
               FROM '{DERIVED / "parse"}/part-*.parquet'),
    w AS (SELECT p.*, COALESCE(tw.work, CASE WHEN p.title LIKE 'tmdb:%' THEN ts.imdb END, p.title) AS work,
                 CASE WHEN p.mention_kind = 'name' THEN p.mention END AS name_first,
                 CASE WHEN p.mention_kind = 'name' THEN p.last END AS name_last
          FROM p LEFT JOIN title_work tw ON tw.title = p.title
          LEFT JOIN tmdb_series_imdb ts ON ts.tmdb_id = p.tmdb_sid)
    SELECT w.title, w.role, w.lemma, w.mention_kind,
           COALESCE(gw.female, nf.female, nl.female) AS female, m.year, m.titleType
    FROM w LEFT JOIN gendered_words gw ON gw.word = w.mention
    LEFT JOIN name_gender nf ON nf.work = w.work AND nf.token = w.name_first
    LEFT JOIN name_gender nl ON nl.work = w.work AND nl.token = w.name_last
    LEFT JOIN meta m ON m.title = w.title""")
cov = con.execute("""SELECT mention_kind, COUNT(*) n, COUNT(female) gendered, AVG(female::INT) women
    FROM mentions GROUP BY ALL ORDER BY 1""").df()
print(cov.to_string(index=False), flush=True)
# spaCy lemmatises "raped" as "rap" (a passive "rape" is never parsed), so "rap" is read as "rape"
con.execute("""CREATE TEMP TABLE g AS SELECT * REPLACE (CASE WHEN lemma = 'rap' THEN 'rape' ELSE lemma END AS lemma),
    CASE WHEN year < 1950 THEN 1940 ELSE CAST(floor(year / 10) * 10 AS INT) END AS decade,
    CASE WHEN year < 1980 THEN '1920-79' WHEN year < 2000 THEN '1980-99' WHEN year < 2010 THEN '2000-09' ELSE '2010-26' END
    AS era, CASE WHEN titleType IN ('movie', 'tvMovie') THEN 'film' ELSE 'tv' END AS medium
    FROM mentions WHERE female IS NOT NULL AND year BETWEEN 1920 AND 2026""")

# gendered who-does-what tuples for the embedding scores (private)
con.execute(f"""COPY (SELECT title, role, lemma, female, year, decade, medium FROM g)
    TO '{DERIVED / "gendered_tuples.parquet"}' (FORMAT parquet)""")

# per-title measures for the ratings models (private: keyed by title, no text)
con.execute(f"""COPY (SELECT title, COUNT(*) FILTER (WHERE female) AS women_mentions,
        COUNT(*) FILTER (WHERE NOT female) AS men_mentions,
        COUNT(*) FILTER (WHERE female AND role = 'subj') AS women_active,
        COUNT(*) FILTER (WHERE NOT female AND role = 'subj') AS men_active,
        COUNT(*) FILTER (WHERE female AND role = 'desc' AND lemma IN (SELECT unnest({CATEGORIES["appearance (desc)"][1].split()}))) AS women_appearance,
        COUNT(*) FILTER (WHERE female AND role = 'pass' AND lemma IN (SELECT unnest({CATEGORIES["violence suffered (pass)"][1].split()}))) AS women_victim
    FROM g GROUP BY title) TO '{DERIVED / "title_gender_text.parquet"}' (FORMAT parquet)""")

agency = con.execute("""SELECT decade, female, role, COUNT(*) n FROM g WHERE role IN ('subj', 'pass', 'obj')
    GROUP BY ALL ORDER BY ALL""").df()
a = agency.pivot_table(index=["decade", "female"], columns="role", values="n").reset_index()
a["active_share"] = a.subj / (a.subj + a["pass"] + a.obj)
a.to_csv(OUT / "agency_by_decade.csv", index=False)
women_subj = agency[agency.role == "subj"].pivot(index="decade", columns="female", values="n")
women_all = agency.groupby(["decade", "female"]).n.sum().unstack()
summary = pd.DataFrame({"women_share_of_active_subjects": women_subj[True] / women_subj.sum(axis=1),
                        "women_share_of_all_mentions": women_all[True] / women_all.sum(axis=1)})
summary.to_csv(OUT / "women_share_of_subjects_by_decade.csv")
print(a.pivot(index="decade", columns="female", values="active_share").round(3).rename(
    columns={False: "men: share active", True: "women: share active"}).join(summary.round(3)).to_string(), flush=True)


def fightin_words(counts, prior_scale=0.01):
    """Monroe et al. (2008) log-odds ratio with an informative Dirichlet prior; z > 0 = associated with women."""
    f, m = counts["f"].to_numpy(float), counts["m"].to_numpy(float)
    prior = (f + m) / (f + m).sum() * (f.sum() + m.sum()) * prior_scale + 0.01
    a0 = prior.sum()
    lf = np.log((f + prior) / (f.sum() + a0 - f - prior))
    lm = np.log((m + prior) / (m.sum() + a0 - m - prior))
    delta = lf - lm
    var = 1 / (f + prior) + 1 / (m + prior)
    return delta / np.sqrt(var)


rows = []
for role in ["subj", "pass", "obj", "desc"]:
    for era in ["1920-79", "1980-99", "2000-09", "2010-26", "all"]:
        q = f"SELECT lemma, SUM(female::INT) f, SUM((NOT female)::INT) m FROM g WHERE role = '{role}'"
        q += "" if era == "all" else f" AND era = '{era}'"
        c = con.execute(q + " GROUP BY lemma HAVING COUNT(*) >= 30").df()
        c = c[~c.lemma.isin(FEMALE + MALE + ["woman", "man", "girl", "boy", "lady", "guy"])]  # gendered by definition
        c["z"] = fightin_words(c)
        c["role"], c["era"] = role, era
        rows.append(c)
words = pd.concat(rows)
without_slurs(words, "lemma").sort_values(["role", "era", "lemma"]).to_csv(OUT / "distinctive_words.csv",
                                                                     index=False)
for role in ["subj", "pass", "desc"]:
    d = words[(words.role == role) & (words.era == "all")].sort_values("z")
    print(f"{role}: most women-associated: " + ", ".join(d.tail(20).lemma[::-1]) + "\n      most men-associated: "
          + ", ".join(d.head(20).lemma), flush=True)

cat_rows = []
base = con.execute("SELECT decade, AVG(female::INT) AS women_all FROM g GROUP BY decade").df()
for name, (role, wordlist) in CATEGORIES.items():
    c = con.execute(f"""SELECT decade, COUNT(*) n, AVG(female::INT) women FROM g WHERE role = '{role}'
        AND lemma IN (SELECT unnest({wordlist.split()})) GROUP BY decade""").df()
    c["category"] = name
    cat_rows.append(c)
cats = pd.concat(cat_rows).merge(base, on="decade")
cats["ratio_to_baseline"] = cats.women / cats.women_all
cats.to_csv(OUT / "categories_by_decade.csv", index=False)
print(cats[cats.decade >= 1950].pivot(index="category", columns="decade", values="ratio_to_baseline").round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
for name in ["appearance (desc)", "intellect (desc)", "power (desc)", "violence done (subj)", "violence suffered (pass)",
             "investigation (subj)", "care (subj)", "emotion (subj)"]:
    d = cats[(cats.category == name) & (cats.decade >= 1950) & (cats.n >= 200)].sort_values("decade")
    ax.plot(d.decade, d.ratio_to_baseline, marker="o", ms=3, label=name)
ax.axhline(1, color="black", lw=1)
ax.set_yscale("log")
ax.set_ylabel("women's share in the category ÷ women's share of all mentions")
ax.set_title("How plot summaries write about women vs men (1 = in line with how often women are mentioned)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "categories_by_decade.png", dpi=150)
