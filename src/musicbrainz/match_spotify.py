import pandas as pd

from src.config import (
    SPOTIFY_CLEAN,
    MB_RECORDINGS_CSV,
    MB_ARTISTS_CSV,
    SPOTIFY_MB_MATCHED,
    MAPPING_CONFIDENCE_THRESHOLD,
)
from src.musicbrainz.mapping_api import (
    map_dataframe,
    make_lookup_key,
)
from src.utils.text import (
    normalize_text,
    primary_artist,
)


# ============================================================
# LOCAL EXACT MATCHING
# ============================================================

def prepare_spotify(df):
    result = df.copy()

    result["_title_norm"] = (
        result["track_name"]
        .fillna("")
        .map(normalize_text)
    )

    result["_artist_norm"] = (
        result["artists"]
        .fillna("")
        .map(
            lambda value: normalize_text(
                primary_artist(value)
            )
        )
    )

    return result


def prepare_recordings(df):
    result = df.copy()

    result["_title_norm"] = (
        result["recording_name"]
        .fillna("")
        .map(normalize_text)
    )

    result["_artist_norm"] = (
        result["primary_artist_name"]
        .fillna("")
        .map(normalize_text)
    )

    result["_has_date"] = (
        result["first_release_date"]
        .notna()
        .astype(int)
    )

    # Plusieurs MB recordings peuvent partager
    # exactement titre + artiste.
    # Pour la passe exacte, on privilégie celui
    # qui possède une date de première sortie.
    result = (
        result
        .sort_values(
            "_has_date",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "_title_norm",
                "_artist_norm",
            ],
            keep="first",
        )
    )

    return result


def local_exact_match(
    spotify,
    recordings,
):
    recording_columns = [
        "_title_norm",
        "_artist_norm",
        "recording_mbid",
        "recording_name",
        "first_release_date",
        "primary_artist_name",
        "primary_artist_mbid",
        "artist_mbids",
        "isrcs",
    ]

    recording_columns = [
        column
        for column in recording_columns
        if column in recordings.columns
    ]

    merged = spotify.merge(
        recordings[
            recording_columns
        ],
        how="left",
        on=[
            "_title_norm",
            "_artist_norm",
        ],
        validate="m:1",
    )

    merged["musicbrainz_match_method"] = (
        "unmatched"
    )

    exact_mask = (
        merged["recording_mbid"]
        .notna()
    )

    merged.loc[
        exact_mask,
        "musicbrainz_match_method",
    ] = "exact_local"

    merged[
        "mapping_confidence"
    ] = pd.NA

    merged.loc[
        exact_mask,
        "mapping_confidence",
    ] = 1.0

    return merged


# ============================================================
# API FALLBACK
# ============================================================

def apply_api_mapping(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    unmatched_mask = (
        merged["recording_mbid"]
        .isna()
    )

    unmatched = merged.loc[
        unmatched_mask
    ].copy()

    if unmatched.empty:
        print(
            "Aucun morceau à envoyer "
            "à ListenBrainz."
        )
        return merged

    print(
        f"\nPasse API ListenBrainz : "
        f"{len(unmatched):,} lignes non matchées."
    )

    cache = map_dataframe(
        unmatched,
        track_column="track_name",
        artist_column="artists",
        release_column=(
            "album_name"
            if "album_name"
            in unmatched.columns
            else None
        ),
    )

    # Recréation de la même clé pour joindre
    # chaque morceau Spotify au cache.
    merged["_lookup_key"] = merged.apply(
        lambda row: make_lookup_key(
            row.get(
                "track_name",
                "",
            ),
            row.get(
                "artists",
                "",
            ),
            (
                row.get(
                    "album_name",
                    "",
                )
                if "album_name"
                in merged.columns
                else ""
            ),
        ),
        axis=1,
    )

    api_columns = [
        "lookup_key",
        "recording_mbid",
        "recording_name",
        "artist_credit_name",
        "artist_mbids",
        "primary_artist_mbid",
        "release_mbid",
        "release_name",
        "mapping_confidence",
        "mapping_status",
    ]

    api_columns = [
        column
        for column in api_columns
        if column in cache.columns
    ]

    api = cache[
        api_columns
    ].copy()

    rename = {
        column: f"api_{column}"
        for column in api.columns
        if column != "lookup_key"
    }

    api = api.rename(
        columns=rename
    )

    merged = merged.merge(
        api,
        how="left",
        left_on="_lookup_key",
        right_on="lookup_key",
        validate="m:1",
    )

    confidence = pd.to_numeric(
        merged.get(
            "api_mapping_confidence"
        ),
        errors="coerce",
    )

    api_recording = merged.get(
        "api_recording_mbid"
    )

    accepted = (
        merged["recording_mbid"]
        .isna()
        & api_recording
        .fillna("")
        .astype(str)
        .ne("")
        & confidence.ge(
            MAPPING_CONFIDENCE_THRESHOLD
        )
    )

    # Champs issus du mapping API.
    for target, source in [
        (
            "recording_mbid",
            "api_recording_mbid",
        ),
        (
            "recording_name",
            "api_recording_name",
        ),
        (
            "artist_mbids",
            "api_artist_mbids",
        ),
        (
            "primary_artist_mbid",
            "api_primary_artist_mbid",
        ),
    ]:
        if source in merged.columns:
            merged.loc[
                accepted,
                target,
            ] = merged.loc[
                accepted,
                source,
            ]

    # Conserver aussi release_mbid / release_name
    # s'ils n'existaient pas dans le matching local.
    for target, source in [
        (
            "release_mbid",
            "api_release_mbid",
        ),
        (
            "release_name",
            "api_release_name",
        ),
    ]:
        if source in merged.columns:
            if target not in merged.columns:
                merged[target] = pd.NA

            merged.loc[
                accepted,
                target,
            ] = merged.loc[
                accepted,
                source,
            ]

    merged.loc[
        accepted,
        "mapping_confidence",
    ] = confidence.loc[
        accepted
    ]

    merged.loc[
        accepted,
        "musicbrainz_match_method",
    ] = "listenbrainz_mapping_api"

    # Les réponses API qui ont trouvé un MBID
    # mais n'atteignent pas le seuil sont explicitement
    # marquées afin d'être analysables dans le mémoire.
    low_confidence = (
        merged["recording_mbid"]
        .isna()
        & api_recording
        .fillna("")
        .astype(str)
        .ne("")
        & confidence.notna()
        & confidence.lt(
            MAPPING_CONFIDENCE_THRESHOLD
        )
    )

    merged.loc[
        low_confidence,
        "musicbrainz_match_method",
    ] = "api_rejected_low_confidence"

    return merged


# ============================================================
# ARTIST METADATA
# ============================================================

def add_artist_metadata(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    if not MB_ARTISTS_CSV.exists():
        return merged

    artists = pd.read_csv(
        MB_ARTISTS_CSV,
        low_memory=False,
    )

    if (
        "artist_mbid"
        not in artists.columns
    ):
        return merged

    artist_columns = [
        "artist_mbid",
        "artist_name",
        "country",
        "area_mbid",
        "area_name",
        "begin_area_mbid",
        "begin_area_name",
    ]

    artist_columns = [
        column
        for column in artist_columns
        if column in artists.columns
    ]

    artists = (
        artists[
            artist_columns
        ]
        .drop_duplicates(
            subset=[
                "artist_mbid"
            ],
            keep="first",
        )
        .rename(
            columns={
                "artist_mbid": (
                    "primary_artist_mbid"
                ),
                "artist_name": (
                    "mb_artist_name"
                ),
                "country": (
                    "artist_country"
                ),
                "area_mbid": (
                    "artist_area_mbid"
                ),
                "area_name": (
                    "artist_area_name"
                ),
                "begin_area_mbid": (
                    "artist_begin_area_mbid"
                ),
                "begin_area_name": (
                    "artist_begin_area_name"
                ),
            }
        )
    )

    # Supprimer d'anciennes colonnes de métadonnées
    # si le fichier est relancé sur un dataset déjà enrichi.
    duplicate_metadata = [
        column
        for column in artists.columns
        if (
            column
            != "primary_artist_mbid"
            and column in merged.columns
        )
    ]

    merged = merged.drop(
        columns=duplicate_metadata,
        errors="ignore",
    )

    return merged.merge(
        artists,
        how="left",
        on="primary_artist_mbid",
        validate="m:1",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    for path in [
        SPOTIFY_CLEAN,
        MB_RECORDINGS_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier requis absent : "
                f"{path}"
            )

    spotify = pd.read_csv(
        SPOTIFY_CLEAN,
        low_memory=False,
    )

    recordings = pd.read_csv(
        MB_RECORDINGS_CSV,
        low_memory=False,
    )

    print(
        "Morceaux Spotify :",
        len(spotify),
    )

    spotify = prepare_spotify(
        spotify
    )

    recordings = prepare_recordings(
        recordings
    )

    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------
    merged = local_exact_match(
        spotify,
        recordings,
    )

    exact_count = (
        merged["recording_mbid"]
        .notna()
        .sum()
    )

    print(
        f"\nPasse 1 - exact local : "
        f"{exact_count:,} "
        f"({exact_count / len(merged) * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # PASS 2
    # --------------------------------------------------------
    merged = apply_api_mapping(
        merged
    )

    # --------------------------------------------------------
    # ARTIST METADATA
    # --------------------------------------------------------
    merged = add_artist_metadata(
        merged
    )

    # Première année connue.
    if "first_release_date" in merged.columns:
        merged["release_year"] = (
            pd.to_numeric(
                merged[
                    "first_release_date"
                ]
                .astype("string")
                .str.slice(0, 4),
                errors="coerce",
            )
            .astype("Int64")
        )

    # Nettoyage des colonnes techniques temporaires/API.
    technical_columns = [
        column
        for column in merged.columns
        if (
            column.startswith("api_")
            or column
            in {
                "_title_norm",
                "_artist_norm",
                "_lookup_key",
                "lookup_key",
            }
        )
    ]

    merged = merged.drop(
        columns=technical_columns,
        errors="ignore",
    )

    SPOTIFY_MB_MATCHED.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        SPOTIFY_MB_MATCHED,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    total_matched = (
        merged["recording_mbid"]
        .notna()
        .sum()
    )

    api_accepted = (
        merged[
            "musicbrainz_match_method"
        ]
        .eq(
            "listenbrainz_mapping_api"
        )
        .sum()
    )

    rejected = (
        merged[
            "musicbrainz_match_method"
        ]
        .eq(
            "api_rejected_low_confidence"
        )
        .sum()
    )

    print(
        "\n--- Matching MusicBrainz V2 ---"
    )

    print(
        f"Morceaux Spotify : "
        f"{len(merged):,}"
    )

    print(
        f"Exact local : "
        f"{exact_count:,}"
    )

    print(
        f"API acceptés "
        f"(confidence >= "
        f"{MAPPING_CONFIDENCE_THRESHOLD:.2f}) : "
        f"{api_accepted:,}"
    )

    print(
        f"API rejetés pour faible confiance : "
        f"{rejected:,}"
    )

    print(
        f"Total matchés : "
        f"{total_matched:,}"
    )

    print(
        f"Taux final : "
        f"{total_matched / len(merged) * 100:.2f}%"
    )

    print(
        "\nSortie :",
        SPOTIFY_MB_MATCHED,
    )


if __name__ == "__main__":
    main()
