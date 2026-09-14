import os
import time
from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import (
    LISTENBRAINZ_MAPPING_URL,
    MAPPING_BATCH_SIZE,
    MAPPING_BATCH_PAUSE_SECONDS,
    MAPPING_MAX_RETRIES,
    MAPPING_TIMEOUT_SECONDS,
    MAPPING_CACHE_CSV,
)
from src.utils.text import (
    normalize_text,
    primary_artist,
)


load_dotenv()


# ============================================================
# TOKEN / HEADERS
# ============================================================

def get_listenbrainz_token() -> str:
    token = os.getenv(
        "LISTENBRAINZ_TOKEN",
        "",
    ).strip()

    if not token:
        raise RuntimeError(
            "\nLISTENBRAINZ_TOKEN manquant.\n\n"
            "Le endpoint ListenBrainz /metadata/lookup/ "
            "nécessite maintenant une authentification.\n\n"
            "Crée un fichier .env à la racine du projet :\n\n"
            "LISTENBRAINZ_TOKEN=ton_token_ici\n"
        )

    return token


def request_headers() -> dict:
    return {
        "Authorization": (
            f"Token {get_listenbrainz_token()}"
        ),
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": (
            "LalachanteMasterPrototype/1.0 "
            "(academic music recommendation prototype)"
        ),
    }


# ============================================================
# CONFIDENCE
# ============================================================

def similarity(a, b) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    return SequenceMatcher(
        None,
        a_norm,
        b_norm,
    ).ratio()


def mapping_confidence(
    input_track,
    input_artist,
    mapped_track,
    mapped_artist,
) -> float:
    """
    ListenBrainz renvoie un MBID mais pas un score de confiance
    dans la réponse documentée du endpoint /metadata/lookup/.

    On calcule donc un score transparent et reproductible :
      60 % similarité du titre
      40 % similarité de l'artiste.
    """
    title_score = similarity(
        input_track,
        mapped_track,
    )

    artist_score = similarity(
        primary_artist(input_artist),
        primary_artist(mapped_artist),
    )

    return (
        0.60 * title_score
        + 0.40 * artist_score
    )


# ============================================================
# CACHE
# ============================================================

CACHE_COLUMNS = [
    "lookup_key",
    "input_track_name",
    "input_artist_name",
    "input_release_name",
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


def make_lookup_key(
    track_name,
    artist_name,
    release_name=None,
):
    return "|||".join(
        [
            normalize_text(track_name),
            normalize_text(
                primary_artist(artist_name)
            ),
            normalize_text(
                release_name or ""
            ),
        ]
    )


def load_mapping_cache() -> pd.DataFrame:
    if not MAPPING_CACHE_CSV.exists():
        return pd.DataFrame(
            columns=CACHE_COLUMNS
        )

    cache = pd.read_csv(
        MAPPING_CACHE_CSV,
        dtype=str,
        keep_default_na=False,
    )

    for column in CACHE_COLUMNS:
        if column not in cache.columns:
            cache[column] = ""

    return cache[CACHE_COLUMNS]


def append_cache(rows: list[dict]):
    if not rows:
        return

    MAPPING_CACHE_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(rows)

    for column in CACHE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[CACHE_COLUMNS]

    frame.to_csv(
        MAPPING_CACHE_CSV,
        mode="a",
        header=not MAPPING_CACHE_CSV.exists(),
        index=False,
    )


# ============================================================
# LISTENBRAINZ LOOKUP LENGTH LIMIT
# ============================================================

MAX_LOOKUP_CHARACTERS = 249


def fit_lookup_to_limit(item: dict) -> dict:
    """Return a copy that respects ListenBrainz's <250-char limit.

    ListenBrainz validates the total number of characters in
    artist_name + recording_name + release_name. We keep the title and
    artist as much as possible and shorten/drop release_name first.
    """
    safe = {
        "recording_name": str(item.get("recording_name", "") or "").strip(),
        "artist_name": str(item.get("artist_name", "") or "").strip(),
    }

    release = str(item.get("release_name", "") or "").strip()
    if release:
        safe["release_name"] = release

    def total_length(value):
        return sum(
            len(str(value.get(key, "") or ""))
            for key in ("artist_name", "recording_name", "release_name")
        )

    if total_length(safe) <= MAX_LOOKUP_CHARACTERS:
        return safe

    # release_name is optional and is the first thing we reduce.
    base_length = len(safe["artist_name"]) + len(safe["recording_name"])
    if base_length < MAX_LOOKUP_CHARACTERS:
        remaining = MAX_LOOKUP_CHARACTERS - base_length
        if release and remaining > 0:
            safe["release_name"] = release[:remaining]
        else:
            safe.pop("release_name", None)
        return safe

    # Very unusual case: title + artist alone exceed the limit.
    safe.pop("release_name", None)
    artist = safe["artist_name"][:90]
    title_budget = max(1, MAX_LOOKUP_CHARACTERS - len(artist))
    title = safe["recording_name"][:title_budget]
    safe["artist_name"] = artist
    safe["recording_name"] = title

    # Final guard, keeping the payload strictly below 250 chars.
    while total_length(safe) > MAX_LOOKUP_CHARACTERS and safe["recording_name"]:
        safe["recording_name"] = safe["recording_name"][:-1]

    return safe


# ============================================================
# RESPONSE PARSING
# ============================================================

def _extract_result_list(data, expected_count):
    """
    L'API a connu plusieurs représentations selon les endpoints /
    versions. Cette fonction accepte les formes usuelles :
      - liste de mappings
      - {"recordings": [...]}
      - {"payload": [...]}
      - mapping unique
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(
            data.get("recordings"),
            list,
        ):
            return data["recordings"]

        if isinstance(
            data.get("payload"),
            list,
        ):
            return data["payload"]

        # Réponse d'un lookup unique.
        if expected_count == 1:
            return [data]

    raise ValueError(
        "Format de réponse ListenBrainz inattendu : "
        f"{type(data).__name__}"
    )


def _normalize_api_result(
    request_item: dict,
    api_result,
) -> dict:
    track = request_item["recording_name"]
    artist = request_item["artist_name"]
    release = request_item.get(
        "release_name",
        "",
    )

    lookup_key = make_lookup_key(
        track,
        artist,
        release,
    )

    # Aucun match.
    if (
        not api_result
        or not isinstance(api_result, dict)
        or not api_result.get(
            "recording_mbid"
        )
    ):
        return {
            "lookup_key": lookup_key,
            "input_track_name": track,
            "input_artist_name": artist,
            "input_release_name": release,
            "recording_mbid": "",
            "recording_name": "",
            "artist_credit_name": "",
            "artist_mbids": "",
            "primary_artist_mbid": "",
            "release_mbid": "",
            "release_name": "",
            "mapping_confidence": 0.0,
            "mapping_status": "unmatched",
        }

    artist_mbids = (
        api_result.get("artist_mbids")
        or []
    )

    if isinstance(artist_mbids, str):
        artist_mbids = [artist_mbids]

    mapped_track = (
        api_result.get("recording_name")
        or ""
    )

    mapped_artist = (
        api_result.get("artist_credit_name")
        or ""
    )

    confidence = mapping_confidence(
        track,
        artist,
        mapped_track,
        mapped_artist,
    )

    return {
        "lookup_key": lookup_key,
        "input_track_name": track,
        "input_artist_name": artist,
        "input_release_name": release,
        "recording_mbid": (
            api_result.get("recording_mbid")
            or ""
        ),
        "recording_name": mapped_track,
        "artist_credit_name": mapped_artist,
        "artist_mbids": "|".join(
            str(value)
            for value in artist_mbids
            if value
        ),
        "primary_artist_mbid": (
            str(artist_mbids[0])
            if artist_mbids
            else ""
        ),
        "release_mbid": (
            api_result.get("release_mbid")
            or ""
        ),
        "release_name": (
            api_result.get("release_name")
            or ""
        ),
        "mapping_confidence": round(
            confidence,
            6,
        ),
        "mapping_status": "mapped",
    }


# ============================================================
# HTTP
# ============================================================

def lookup_batch(
    recordings: list[dict],
) -> list[dict]:
    if not recordings:
        return []

    safe_recordings = [
        fit_lookup_to_limit(item)
        for item in recordings
    ]

    payload = {
        "recordings": safe_recordings
    }

    delay = 2.0

    for attempt in range(
        1,
        MAPPING_MAX_RETRIES + 1,
    ):
        response = requests.post(
            LISTENBRAINZ_MAPPING_URL,
            json=payload,
            headers=request_headers(),
            timeout=MAPPING_TIMEOUT_SECONDS,
        )

        if response.status_code == 429:
            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            wait = (
                float(retry_after)
                if retry_after
                else delay
            )

            print(
                f"ListenBrainz rate limit : "
                f"attente {wait:.1f}s..."
            )

            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        if (
            500
            <= response.status_code
            < 600
        ):
            print(
                f"ListenBrainz erreur "
                f"{response.status_code} "
                f"(tentative {attempt}/"
                f"{MAPPING_MAX_RETRIES})."
            )

            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if response.status_code == 401:
            raise RuntimeError(
                "ListenBrainz a refusé le token "
                "(401 Unauthorized). Vérifie "
                "LISTENBRAINZ_TOKEN dans .env."
            )

        if response.status_code == 400:
            # Do not stop tens of thousands of lookups because one item
            # in a batch is invalid. Split the batch to isolate it.
            if len(recordings) > 1:
                midpoint = len(recordings) // 2
                return (
                    lookup_batch(recordings[:midpoint])
                    + lookup_batch(recordings[midpoint:])
                )

            bad = recordings[0]
            print(
                "\n⚠️ Un morceau a été ignoré après une erreur 400 ListenBrainz:"
            )
            print("   Titre   :", str(bad.get("recording_name", ""))[:120])
            print("   Artiste :", str(bad.get("artist_name", ""))[:120])
            print("   Réponse :", response.text[:300])

            track = bad.get("recording_name", "")
            artist = bad.get("artist_name", "")
            release = bad.get("release_name", "")

            return [{
                "lookup_key": make_lookup_key(track, artist, release),
                "input_track_name": track,
                "input_artist_name": artist,
                "input_release_name": release,
                "recording_mbid": "",
                "recording_name": "",
                "artist_credit_name": "",
                "artist_mbids": "",
                "primary_artist_mbid": "",
                "release_mbid": "",
                "release_name": "",
                "mapping_confidence": 0.0,
                "mapping_status": "api_error_400",
            }]

        response.raise_for_status()

        data = response.json()

        results = _extract_result_list(
            data,
            len(recordings),
        )

        # Le POST lookup est conçu pour retourner
        # une réponse par enregistrement, dans l'ordre.
        # Si l'API renvoie moins d'éléments, on complète
        # avec des non-matchs pour ne pas décaler les lignes.
        if len(results) < len(recordings):
            results = list(results) + [
                None
            ] * (
                len(recordings)
                - len(results)
            )

        normalized = []

        for request_item, api_result in zip(
            recordings,
            results,
        ):
            normalized.append(
                _normalize_api_result(
                    request_item,
                    api_result,
                )
            )

        return normalized

    raise RuntimeError(
        "Échec ListenBrainz après "
        f"{MAPPING_MAX_RETRIES} tentatives."
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def map_dataframe(
    df: pd.DataFrame,
    *,
    track_column="track_name",
    artist_column="artists",
    release_column="album_name",
) -> pd.DataFrame:
    """
    Mappe les lignes Spotify non résolues vers MusicBrainz.

    - utilise le cache existant ;
    - ne requête que les clés encore inconnues ;
    - sauvegarde chaque lot immédiatement ;
    - retourne un DataFrame indexé sur lookup_key.
    """

    cache = load_mapping_cache()

    cached_keys = set(
        cache["lookup_key"]
        .astype(str)
        .tolist()
    )

    unique_requests = {}

    for _, row in df.iterrows():
        track = row.get(
            track_column,
            "",
        )
        artist = row.get(
            artist_column,
            "",
        )

        release = (
            row.get(
                release_column,
                "",
            )
            if release_column
            and release_column in df.columns
            else ""
        )

        key = make_lookup_key(
            track,
            artist,
            release,
        )

        if (
            not key
            or key in cached_keys
            or key in unique_requests
        ):
            continue

        item = {
            "recording_name": str(track),
            "artist_name": str(
                primary_artist(artist)
            ),
        }

        if (
            release
            and str(release).strip()
            and str(release).lower()
            != "nan"
        ):
            item["release_name"] = str(
                release
            )

        unique_requests[key] = item

    requests_to_make = list(
        unique_requests.values()
    )

    print(
        f"ListenBrainz mapping : "
        f"{len(requests_to_make):,} nouvelles "
        f"requêtes uniques."
    )

    for start in range(
        0,
        len(requests_to_make),
        MAPPING_BATCH_SIZE,
    ):
        batch = requests_to_make[
            start:start + MAPPING_BATCH_SIZE
        ]

        mapped_rows = lookup_batch(
            batch
        )

        append_cache(mapped_rows)

        processed = min(
            start + len(batch),
            len(requests_to_make),
        )

        print(
            f"  {processed:,}/"
            f"{len(requests_to_make):,}"
        )

        time.sleep(
            MAPPING_BATCH_PAUSE_SECONDS
        )

    # Recharge le cache final.
    cache = load_mapping_cache()

    # Éviter les doublons si le script a été relancé
    # après une interruption sur un même lot.
    cache = cache.drop_duplicates(
        subset=["lookup_key"],
        keep="last",
    )

    return cache
