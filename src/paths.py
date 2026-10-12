"""Where the data lives. Raw and derived data stay outside the repo (IMDb and TMDb terms); only aggregates go in results/.

Set STEREOTYPES_DATA to the folder holding imdb/, tmdb/, gcp-2022/, lexicons/ and derived/.
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("STEREOTYPES_DATA", Path.home() / "stereotypes-data")).expanduser()
IMDB = DATA / "imdb/2026-10-07"
TMDB_DB = DATA / "tmdb/tmdb.sqlite"
LEXICONS = DATA / "lexicons"
DERIVED = DATA / "derived"
RESULTS = Path(__file__).resolve().parent.parent / "results"
