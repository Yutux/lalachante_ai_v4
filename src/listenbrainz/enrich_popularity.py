import time
import pandas as pd
from tqdm import tqdm
from src.config import (
    SPOTIFY_MB_MATCHED,
    LB_RECORDING_POPULARITY,
    LB_ARTIST_POPULARITY,
    LISTENBRAINZ_API,
    LB_BATCH_SIZE,
)
from src.utils.http import post_json_with_retry


def unique_values(series):
    return list(dict.fromkeys(
        str(v) for v in series.dropna()
        if str(v).strip() and str(v).lower() != "nan"
    ))


def load_done(path, key):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if key not in df.columns:
        return set()
    return set(df[key].dropna().astype(str))


def append_rows(path, rows):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def enrich_recordings(mbids):
    done = load_done(LB_RECORDING_POPULARITY, "recording_mbid")
    todo = [m for m in mbids if m not in done]
    print(f"Recording MBID : {len(mbids):,} total ; {len(todo):,} à interroger.")
    url = f"{LISTENBRAINZ_API}/popularity/recording"
    batch_list = list(batches(todo, LB_BATCH_SIZE))
    for batch in tqdm(batch_list, desc="ListenBrainz recordings"):
        result = post_json_with_retry(url, {"recording_mbids": batch})
        if not isinstance(result, list):
            raise ValueError("Réponse ListenBrainz inattendue pour recordings.")
        append_rows(LB_RECORDING_POPULARITY, result)
        time.sleep(0.15)


def enrich_artists(mbids):
    done = load_done(LB_ARTIST_POPULARITY, "artist_mbid")
    todo = [m for m in mbids if m not in done]
    print(f"Artist MBID : {len(mbids):,} total ; {len(todo):,} à interroger.")
    url = f"{LISTENBRAINZ_API}/popularity/artist"
    batch_list = list(batches(todo, LB_BATCH_SIZE))
    for batch in tqdm(batch_list, desc="ListenBrainz artists"):
        result = post_json_with_retry(url, {"artist_mbids": batch})
        if not isinstance(result, list):
            raise ValueError("Réponse ListenBrainz inattendue pour artists.")
        append_rows(LB_ARTIST_POPULARITY, result)
        time.sleep(0.15)


def main():
    if not SPOTIFY_MB_MATCHED.exists():
        raise FileNotFoundError(f"Matching MusicBrainz absent : {SPOTIFY_MB_MATCHED}")

    LB_RECORDING_POPULARITY.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SPOTIFY_MB_MATCHED, low_memory=False)
    recording_mbids = unique_values(df["recording_mbid"])
    artist_mbids = unique_values(df["primary_artist_mbid"])
    enrich_recordings(recording_mbids)
    enrich_artists(artist_mbids)
    print("\nEnrichissement ListenBrainz terminé.")
    print("Recordings :", LB_RECORDING_POPULARITY)
    print("Artists :", LB_ARTIST_POPULARITY)


if __name__ == "__main__":
    main()
