import csv
import pandas as pd
from tqdm import tqdm

from src.config import (
    MB_RELEASE_GROUP_ARCHIVE,
    MB_RELEASE_GROUPS_CSV,
    SPOTIFY_CLEAN,
)
from src.utils.musicbrainz_tar import iter_json_dump
from src.utils.text import normalize_text, primary_artist


def parse_artist_credit(record):
    names, mbids = [], []
    for credit in record.get("artist-credit") or []:
        if not isinstance(credit, dict):
            continue
        artist = credit.get("artist") or {}
        if artist.get("name"):
            names.append(str(artist["name"]))
        if artist.get("id"):
            mbids.append(str(artist["id"]))
    return names, mbids


def spotify_filters():
    if not SPOTIFY_CLEAN.exists():
        return None, None
    df = pd.read_csv(SPOTIFY_CLEAN, usecols=["track_name", "artists"])
    titles = {normalize_text(v) for v in df["track_name"].dropna() if normalize_text(v)}
    artists = {
        normalize_text(primary_artist(v))
        for v in df["artists"].dropna()
        if normalize_text(primary_artist(v))
    }
    return titles, artists


def main():
    if not MB_RELEASE_GROUP_ARCHIVE.exists():
        print("release-group.tar.xz absent : étape ignorée.")
        return

    titles, artists_filter = spotify_filters()
    MB_RELEASE_GROUPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "release_group_mbid", "release_group_title", "primary_type",
        "secondary_types", "first_release_date", "artist_names", "artist_mbids",
        "disambiguation",
    ]
    scanned = written = 0

    with MB_RELEASE_GROUPS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in tqdm(
            iter_json_dump(MB_RELEASE_GROUP_ARCHIVE, "release-group"),
            desc="MusicBrainz release groups",
        ):
            scanned += 1
            title = record.get("title") or ""
            if titles is not None and normalize_text(title) not in titles:
                continue
            artist_names, artist_mbids = parse_artist_credit(record)
            primary = artist_names[0] if artist_names else ""
            if artists_filter is not None and normalize_text(primary) not in artists_filter:
                continue
            writer.writerow({
                "release_group_mbid": record.get("id"),
                "release_group_title": title,
                "primary_type": record.get("primary-type"),
                "secondary_types": "|".join(record.get("secondary-types") or []),
                "first_release_date": record.get("first-release-date"),
                "artist_names": "|".join(artist_names),
                "artist_mbids": "|".join(artist_mbids),
                "disambiguation": record.get("disambiguation"),
            })
            written += 1

    print(f"{scanned:,} release groups scannés ; {written:,} candidats conservés.")
    print("Sortie :", MB_RELEASE_GROUPS_CSV)


if __name__ == "__main__":
    main()
