import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import FINAL_DATASET, SPOTIFY_CLEAN, FEATURE_DATASET, AUDIO_FEATURES


def main():
    source = FINAL_DATASET if FINAL_DATASET.exists() else SPOTIFY_CLEAN
    if not source.exists():
        raise FileNotFoundError("Aucun dataset source disponible.")

    df = pd.read_csv(source, low_memory=False)
    missing = [f for f in AUDIO_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes : {missing}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[AUDIO_FEATURES])
    scaled_df = pd.DataFrame(
        scaled,
        columns=[f"scaled_{feature}" for feature in AUDIO_FEATURES],
        index=df.index,
    )

    metadata = [
        "track_id", "track_name", "artists", "track_genre", "recording_mbid",
        "primary_artist_mbid", "artist_country", "artist_area_name", "release_year",
    ]
    metadata = [c for c in metadata if c in df.columns]
    output = pd.concat([df[metadata].copy(), scaled_df], axis=1)

    FEATURE_DATASET.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(FEATURE_DATASET, index=False)

    print("Dataset de features :", FEATURE_DATASET)
    print("Dimensions :", output.shape)
    print("\nMoyennes :")
    print(scaled_df.mean().round(6))
    print("\nÉcarts-types :")
    print(scaled_df.std().round(6))


if __name__ == "__main__":
    main()
