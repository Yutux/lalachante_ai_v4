import csv
import pandas as pd
from tqdm import tqdm
from src.config import MB_RECORDING_ARCHIVE, MB_RECORDINGS_CSV, SPOTIFY_CLEAN
from src.utils.musicbrainz_tar import iter_json_dump
from src.utils.text import normalize_text, primary_artist


def parse_artist_credit(record):
    credits = record.get("artist-credit") or []
    names, mbids, parts = [], [], []
    for credit in credits:
        if not isinstance(credit, dict):
            continue
        credited_name = credit.get("name")
        artist = credit.get("artist") or {}
        canonical_name = artist.get("name")
        mbid = artist.get("id")
        if canonical_name:
            names.append(str(canonical_name))
        elif credited_name:
            names.append(str(credited_name))
        if mbid:
            mbids.append(str(mbid))
        if credited_name:
            parts.append(str(credited_name))
        if credit.get("joinphrase"):
            parts.append(str(credit["joinphrase"]))
    credit_name = "".join(parts).strip() or "; ".join(names)
    return {
        "artist_credit_name": credit_name,
        "artist_names": "|".join(names),
        "artist_mbids": "|".join(mbids),
        "primary_artist_name": names[0] if names else "",
        "primary_artist_mbid": mbids[0] if mbids else "",
    }


def load_spotify_filters():
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
    spotify_titles, spotify_artists = load_spotify_filters()
    if spotify_titles is not None:
        print(f"Préfiltre : {len(spotify_titles):,} titres et {len(spotify_artists):,} artistes Spotify")

    MB_RECORDINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "recording_mbid", "recording_name", "recording_length_ms", "video",
        "first_release_date", "disambiguation", "artist_credit_name", "artist_names",
        "artist_mbids", "primary_artist_name", "primary_artist_mbid", "isrcs",
    ]
    scanned = written = 0

    with MB_RECORDINGS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in tqdm(iter_json_dump(MB_RECORDING_ARCHIVE, "recording"), desc="MusicBrainz recordings"):
            scanned += 1
            title = record.get("title") or ""
            if spotify_titles is not None and normalize_text(title) not in spotify_titles:
                continue
            credit = parse_artist_credit(record)
            if spotify_artists is not None and normalize_text(credit["primary_artist_name"]) not in spotify_artists:
                continue
            writer.writerow({
                "recording_mbid": record.get("id"),
                "recording_name": title,
                "recording_length_ms": record.get("length"),
                "video": record.get("video"),
                "first_release_date": record.get("first-release-date"),
                "disambiguation": record.get("disambiguation"),
                **credit,
                "isrcs": "|".join(record.get("isrcs") or []),
            })
            written += 1

    print(f"{scanned:,} recordings scannés ; {written:,} candidats conservés.")
    print("Sortie :", MB_RECORDINGS_CSV)


if __name__ == "__main__":
    main()
