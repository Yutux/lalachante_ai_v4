import pandas as pd
from src.config import SPOTIFY_RAW, SPOTIFY_CLEAN, AUDIO_FEATURES

NUMERIC_COLUMNS = [
    "popularity", "duration_ms", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]
ZERO_ONE_COLUMNS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]


def main():
    if not SPOTIFY_RAW.exists():
        raise FileNotFoundError(f"Dataset Spotify introuvable : {SPOTIFY_RAW}")

    df = pd.read_csv(SPOTIFY_RAW)
    initial_shape = df.shape
    print("Dimensions initiales :", initial_shape)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required = ["track_id", "track_name", "artists", "track_genre", *AUDIO_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes Spotify manquantes : {missing}")

    df = df.dropna(subset=["track_id", "track_name", "artists", "track_genre"])

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_present = [c for c in NUMERIC_COLUMNS if c in df.columns]
    df = df.dropna(subset=numeric_present)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["track_id"], keep="first")

    for column in ["track_name", "artists", "album_name", "track_genre"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    df["track_genre"] = df["track_genre"].str.lower().str.strip()

    for feature in ZERO_ONE_COLUMNS:
        if feature in df.columns:
            df = df[df[feature].between(0, 1, inclusive="both")]

    if "tempo" in df.columns:
        df = df[df["tempo"] > 0]
    if "popularity" in df.columns:
        df = df[df["popularity"].between(0, 100, inclusive="both")]
    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000

    df = df.reset_index(drop=True)
    SPOTIFY_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPOTIFY_CLEAN, index=False)

    removed = initial_shape[0] - len(df)
    print("\n--- Résultat ---")
    print("Dimensions finales :", df.shape)
    print("Lignes supprimées :", removed)
    print(f"Pourcentage supprimé : {removed / initial_shape[0] * 100:.2f}%")
    print("Sortie :", SPOTIFY_CLEAN)


if __name__ == "__main__":
    main()
