"""The scored first seasons of finished scripted series (score_series.py) with their TMDb record, for makers.py and
series_reception.py: one row per series with the classifier means, the start year, the first network, the share of
women among the creators, the scored summaries' mean log length and, on request, the women's share of the scored
episodes' credited cast (IMDb principals, actor/actress; episodes with at least two credited actors, averaged).
"""
import duckdb
import numpy as np
import pandas as pd

from paths import DERIVED, IMDB


def first_seasons(cast=False):
    ss = pd.read_parquet(DERIVED / "series_classifier_scores.parquet")
    tm = pd.read_parquet(DERIVED / "tmdb_titles.parquet")
    ss = ss.merge(tm[tm.kind == "series"].drop_duplicates("imdb").rename(columns={"imdb": "series"}), on="series",
                  how="left")
    ss["start"] = pd.to_datetime(ss.first_air_date, errors="coerce").dt.year
    ss["network"] = ss.networks.fillna("").str.split("|").str[0].replace("", np.nan)
    ss["series_cluster"] = ss.network.fillna(ss.series)
    ss["genre_decade"] = ss.tmdb_genres.fillna("none") + "|" + (ss.start // 10 * 10).astype("Int64").astype(str)
    ss["genre_decade_language"] = ss.genre_decade + "|" + ss.language.fillna("none")
    known = (ss.creators_women + ss.creators_men) > 0
    ss["creators_women_share"] = (ss.creators_women / (ss.creators_women + ss.creators_men)).where(known)
    eps = pd.read_parquet(DERIVED / "series_embeddings/manifest.parquet")  # the episodes score_series.py scored
    ss = ss.merge(eps.assign(log_words=np.log(eps.words)).groupby("series").log_words.mean().reset_index(),
                  on="series", how="left")
    if cast:
        con = duckdb.connect()
        con.register("scored", eps[["series", "title"]])
        cols = ", ".join(f"'{c}': 'VARCHAR'" for c in ["tconst", "ordering", "nconst", "category", "job", "characters"])
        c = con.execute(f"""
            WITH c AS (SELECT tconst, AVG((category = 'actress')::INT) AS women, COUNT(*) AS n
                       FROM read_csv('{IMDB / "title.principals.tsv.gz"}', delim='\\t', header=true, quote='',
                                     escape='', nullstr='\\N', columns={{{cols}}})
                       WHERE category IN ('actor', 'actress') AND tconst IN (SELECT title FROM scored)
                       GROUP BY tconst)
            SELECT s.series, AVG(c.women) AS cast_women FROM scored s JOIN c ON c.tconst = s.title
            WHERE c.n >= 2 GROUP BY s.series""").df()
        ss = ss.merge(c, on="series", how="left")
    return ss
