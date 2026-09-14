import numpy as np
import pandas as pd
from src.config import (
    SPOTIFY_MB_MATCHED,
    LB_RECORDING_POPULARITY,
    LB_ARTIST_POPULARITY,
    FINAL_DATASET,
)


def safe_log1p(series):
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return np.log1p(numeric.clip(lower=0))


def percentile_rank(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True)


def main():
    if not SPOTIFY_MB_MATCHED.exists():
        raise FileNotFoundError(f"Fichier absent : {SPOTIFY_MB_MATCHED}")

    df = pd.read_csv(SPOTIFY_MB_MATCHED, low_memory=False)

    if LB_RECORDING_POPULARITY.exists():
        rec = pd.read_csv(LB_RECORDING_POPULARITY).drop_duplicates(
            subset=["recording_mbid"], keep="last"
        )
        rec = rec.rename(columns={
            "total_listen_count": "lb_recording_listen_count",
            "total_user_count": "lb_recording_listener_count",
        })
        df = df.merge(rec, how="left", on="recording_mbid", validate="m:1")

    if LB_ARTIST_POPULARITY.exists():
        art = pd.read_csv(LB_ARTIST_POPULARITY).drop_duplicates(
            subset=["artist_mbid"], keep="last"
        )
        art = art.rename(columns={
            "artist_mbid": "primary_artist_mbid",
            "total_listen_count": "lb_artist_listen_count",
            "total_user_count": "lb_artist_listener_count",
        })
        df = df.merge(art, how="left", on="primary_artist_mbid", validate="m:1")

    for source, output in [
        ("lb_recording_listener_count", "lb_recording_listener_percentile"),
        ("lb_recording_listen_count", "lb_recording_listen_percentile"),
        ("lb_artist_listener_count", "lb_artist_listener_percentile"),
        ("lb_artist_listen_count", "lb_artist_listen_percentile"),
    ]:
        if source in df.columns:
            df[output] = percentile_rank(safe_log1p(df[source]))

    FINAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_DATASET, index=False)

    print("\n--- Dataset final ---")
    print("Dimensions :", df.shape)
    print("Sortie :", FINAL_DATASET)

    if "recording_mbid" in df.columns:
        print(f"Couverture MusicBrainz : {df['recording_mbid'].notna().mean() * 100:.1f}%")
    if "lb_recording_listener_count" in df.columns:
        print(
            "Couverture ListenBrainz recording : "
            f"{df['lb_recording_listener_count'].notna().mean() * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
