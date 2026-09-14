import csv
import shutil
import pandas as pd
from tqdm import tqdm

from src.config import SPOTIFY_MB_MATCHED, MB_ARTIST_ARCHIVE, INTERMEDIATE_DIR
from src.utils.musicbrainz_tar import iter_json_dump

SELECTED_ARTISTS_CSV = INTERMEDIATE_DIR / "musicbrainz_selected_artists.csv"
BACKUP_MATCHED_CSV = INTERMEDIATE_DIR / "spotify_musicbrainz_matched_before_artist_metadata.csv"


def nested_id(value):
    return value.get("id") if isinstance(value, dict) else None


def nested_name(value):
    return value.get("name") if isinstance(value, dict) else None


def collect_needed_mbids(df):
    needed = set()
    if "primary_artist_mbid" in df.columns:
        for value in df["primary_artist_mbid"].dropna():
            text = str(value).strip()
            if text and text.lower() != "nan":
                needed.add(text)
    if "artist_mbids" in df.columns:
        for value in df["artist_mbids"].dropna():
            for mbid in str(value).split("|"):
                mbid = mbid.strip()
                if mbid and mbid.lower() != "nan":
                    needed.add(mbid)
    return needed


def extract_selected_artists(needed_mbids):
    fields = [
        "artist_mbid", "artist_name", "artist_sort_name", "artist_type",
        "country", "area_mbid", "area_name", "begin_area_mbid",
        "begin_area_name", "end_area_mbid", "end_area_name",
        "begin", "end", "ended", "disambiguation",
    ]
    found = 0
    SELECTED_ARTISTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with SELECTED_ARTISTS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for record in tqdm(
            iter_json_dump(MB_ARTIST_ARCHIVE, "artist"),
            desc="Recherche artistes MusicBrainz",
        ):
            mbid = str(record.get("id") or "")
            if mbid not in needed_mbids:
                continue

            life = record.get("life-span") or {}

            writer.writerow({
                "artist_mbid": mbid,
                "artist_name": record.get("name"),
                "artist_sort_name": record.get("sort-name"),
                "artist_type": record.get("type"),
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
            })
            found += 1
            if found >= len(needed_mbids):
                break
    return found


def main():
    if not SPOTIFY_MB_MATCHED.exists():
        raise FileNotFoundError(f"Fichier absent : {SPOTIFY_MB_MATCHED}")
    if not MB_ARTIST_ARCHIVE.exists():
        raise FileNotFoundError(f"Archive absente : {MB_ARTIST_ARCHIVE}")

    df = pd.read_csv(SPOTIFY_MB_MATCHED, low_memory=False)
    needed = collect_needed_mbids(df)
    print(f"Artist MBID distincts à rechercher : {len(needed):,}")

    found = extract_selected_artists(needed)
    print(f"Artistes trouvés dans le dump : {found:,}")

    artists = pd.read_csv(SELECTED_ARTISTS_CSV, low_memory=False)
    artists = (
        artists.drop_duplicates(subset=["artist_mbid"], keep="first")
        .rename(columns={
            "artist_mbid": "primary_artist_mbid",
            "artist_name": "mb_artist_name",
            "artist_type": "mb_artist_type",
            "country": "artist_country",
            "area_mbid": "artist_area_mbid",
            "area_name": "artist_area_name",
            "begin_area_mbid": "artist_begin_area_mbid",
            "begin_area_name": "artist_begin_area_name",
        })
    )

    metadata = [
        "primary_artist_mbid", "mb_artist_name", "mb_artist_type",
        "artist_country", "artist_area_mbid", "artist_area_name",
        "artist_begin_area_mbid", "artist_begin_area_name",
    ]
    artists = artists[[c for c in metadata if c in artists.columns]]

    df = df.drop(
        columns=[c for c in metadata if c != "primary_artist_mbid" and c in df.columns],
        errors="ignore",
    )

    df = df.merge(artists, how="left", on="primary_artist_mbid", validate="m:1")

    if not BACKUP_MATCHED_CSV.exists():
        shutil.copy2(SPOTIFY_MB_MATCHED, BACKUP_MATCHED_CSV)

    df.to_csv(SPOTIFY_MB_MATCHED, index=False)

    country_cov = df["artist_country"].notna().mean() * 100 if "artist_country" in df.columns else 0
    area_cov = df["artist_area_name"].notna().mean() * 100 if "artist_area_name" in df.columns else 0

    print("\n--- Enrichissement artiste terminé ---")
    print(f"Couverture pays : {country_cov:.2f}%")
    print(f"Couverture zone : {area_cov:.2f}%")
    print("Fichier mis à jour :", SPOTIFY_MB_MATCHED)
    print("Sauvegarde :", BACKUP_MATCHED_CSV)


if __name__ == "__main__":
    main()