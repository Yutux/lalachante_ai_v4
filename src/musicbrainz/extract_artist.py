import csv
import pandas as pd
from tqdm import tqdm
from src.config import MB_ARTIST_ARCHIVE, MB_ARTISTS_CSV, SPOTIFY_CLEAN
from src.utils.musicbrainz_tar import iter_json_dump
from src.utils.text import normalize_text, primary_artist


def nested_id(value):
    return value.get("id") if isinstance(value, dict) else None


def nested_name(value):
    return value.get("name") if isinstance(value, dict) else None


def spotify_artist_names():
    if not SPOTIFY_CLEAN.exists():
        return None
    df = pd.read_csv(SPOTIFY_CLEAN, usecols=["artists"])
    return {
        normalize_text(primary_artist(value))
        for value in df["artists"].dropna()
        if normalize_text(primary_artist(value))
    }


def main():
    relevant = spotify_artist_names()
    if relevant is not None:
        print(f"Préfiltre Spotify : {len(relevant):,} artistes distincts")

    MB_ARTISTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "artist_mbid", "artist_name", "artist_sort_name", "artist_type",
        "artist_type_id", "gender", "country", "area_mbid", "area_name",
        "begin_area_mbid", "begin_area_name", "end_area_mbid", "end_area_name",
        "begin", "end", "ended", "disambiguation", "isnis", "ipis",
    ]
    scanned = written = 0

    with MB_ARTISTS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in tqdm(iter_json_dump(MB_ARTIST_ARCHIVE, "artist"), desc="MusicBrainz artists"):
            scanned += 1
            name = record.get("name") or ""
            if relevant is not None and normalize_text(name) not in relevant:
                continue
            life = record.get("life-span") or {}
            writer.writerow({
                "artist_mbid": record.get("id"),
                "artist_name": name,
                "artist_sort_name": record.get("sort-name"),
                "artist_type": record.get("type"),
                "artist_type_id": record.get("type-id"),
                "gender": record.get("gender"),
                "country": record.get("country"),
                "area_mbid": nested_id(record.get("area")),
                "area_name": nested_name(record.get("area")),
                "begin_area_mbid": nested_id(record.get("begin-area")),
                "begin_area_name": nested_name(record.get("begin-area")),
                "end_area_mbid": nested_id(record.get("end-area")),
                "end_area_name": nested_name(record.get("end-area")),
                "begin": life.get("begin"),
                "end": life.get("end"),
                "ended": life.get("ended"),
                "disambiguation": record.get("disambiguation"),
                "isnis": "|".join(record.get("isnis") or []),
                "ipis": "|".join(record.get("ipis") or []),
            })
            written += 1

    print(f"{scanned:,} artistes scannés ; {written:,} conservés.")
    print("Sortie :", MB_ARTISTS_CSV)


if __name__ == "__main__":
    main()
