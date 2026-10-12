"""Per-episode writers, directors and guest cast from the TMDb season responses, with TMDb's gender codes
(0 unknown, 1 woman, 2 man, 3 non-binary). Guest stars are the people cast for that episode only; series regulars are
not listed per episode, so guest casting is where an episode's writers have the most say.

Outputs (private, derived from TMDb): DERIVED/tmdb_episodes.parquet (one row per episode, counts by gender) and
DERIVED/tmdb_guest_roles.parquet (one row per guest credit, with the character name).
"""
import json
import sqlite3
import zlib

import pandas as pd

from paths import DERIVED, TMDB_DB

WRITING = {"Writer", "Teleplay", "Story", "Screenplay", "Screenstory", "Story Editor", "Head Writer"}
db = sqlite3.connect(f"file:{TMDB_DB}?mode=ro", uri=True)
series = {}
for path, body in db.execute("SELECT path, body FROM responses WHERE path LIKE '/tv/%?append%' AND status = 200"):
    d = json.loads(zlib.decompress(body))
    series[d["id"]] = (d.get("type"), d.get("original_language"), (d.get("origin_country") or [None])[0],
                       (d.get("external_ids") or {}).get("imdb_id"),
                       "|".join(g["name"] for g in d.get("genres") or []),
                       "|".join(str(p.get("gender")) for p in d.get("created_by") or []))

episodes, guests = [], []
for path, body in db.execute("SELECT path, body FROM responses WHERE path LIKE '/tv/%/season/%' AND status = 200"):
    sid = int(path.split("/")[2])
    for e in json.loads(zlib.decompress(body)).get("episodes") or []:
        count = {f"{role}_{g}": 0 for role in ("writers", "directors", "guests") for g in (0, 1, 2, 3)}
        seen = set()
        for c in e.get("crew") or []:
            role = "writers" if c.get("job") in WRITING else "directors" if c.get("job") == "Director" else None
            if role and (role, c.get("id")) not in seen:
                seen.add((role, c.get("id")))
                count[f"{role}_{c.get('gender') or 0}"] += 1
        for g in e.get("guest_stars") or []:
            count[f"guests_{g.get('gender') or 0}"] += 1
            guests.append((sid, e.get("season_number"), e.get("episode_number"), g.get("id"), g.get("gender") or 0,
                           (g.get("character") or "").strip(), g.get("order")))
        air = e.get("air_date") or ""
        episodes.append((sid, e.get("season_number"), e.get("episode_number"), int(air[:4]) if air[:4].isdigit() else None,
                         e.get("vote_average"), e.get("vote_count"), len((e.get("overview") or "").split()), *count.values()))

cols = ["series", "season", "episode", "year", "vote_average", "vote_count", "overview_words"] + list(count)
ep = pd.DataFrame(episodes, columns=cols)
meta = pd.DataFrame.from_dict(series, orient="index", columns=["tv_type", "language", "country", "imdb", "tmdb_genres",
                                                              "creator_genders"]).rename_axis("series").reset_index()
ep = ep.merge(meta, on="series", how="left")
ep.to_parquet(DERIVED / "tmdb_episodes.parquet", index=False)
pd.DataFrame(guests, columns=["series", "season", "episode", "person", "gender", "character", "order"]).to_parquet(
    DERIVED / "tmdb_guest_roles.parquet", index=False)
has_w = ep[[f"writers_{g}" for g in (1, 2, 3)]].sum(axis=1) > 0
has_g = ep[[f"guests_{g}" for g in (1, 2, 3)]].sum(axis=1) > 0
print(f"episodes {len(ep):,}; with a writer of known gender {has_w.sum():,}; with a guest of known gender {has_g.sum():,}; "
      f"both {(has_w & has_g).sum():,}; guest credits {len(guests):,}")
