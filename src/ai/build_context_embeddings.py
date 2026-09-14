import re

import pandas as pd
import pycountry
from sentence_transformers import SentenceTransformer

from src.config import (
    FINAL_DATASET,
    SPOTIFY_CLEAN,
)
from src.ai.embedding_store import (
    save_embeddings,
)


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

BATCH_SIZE = 128


def safe_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "none",
        "nan",
        "null",
    }:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text,
    )


def country_label(value):
    code = safe_text(value).upper()

    if not code:
        return ""

    try:
        country = pycountry.countries.get(
            alpha_2=code
        )

        if country:
            return country.name
    except Exception:
        pass

    return code


def build_document(row) -> str:
    """
    Le nombre brut d'auditeurs n'est volontairement PAS inclus
    dans le document envoyé au modèle.
    """
    parts = []

    track = safe_text(
        row.get(
            "track_name",
            "",
        )
    )

    artist = safe_text(
        row.get(
            "artists",
            "",
        )
    )

    genre = safe_text(
        row.get(
            "track_genre",
            "",
        )
    )

    country = country_label(
        row.get(
            "artist_country",
            "",
        )
    )

    area = safe_text(
        row.get(
            "artist_area_name",
            "",
        )
    )

    begin_area = safe_text(
        row.get(
            "artist_begin_area_name",
            "",
        )
    )

    year = safe_text(
        row.get(
            "release_year",
            "",
        )
    )

    if track:
        parts.append(
            f"Track: {track}."
        )

    if artist:
        parts.append(
            f"Artist: {artist}."
        )

    if genre:
        parts.append(
            f"Genre: {genre}."
        )

    if country:
        parts.append(
            f"Artist country: {country}."
        )

    if area:
        parts.append(
            f"Artist area: {area}."
        )

    if (
        begin_area
        and begin_area != area
    ):
        parts.append(
            f"Artist origin area: "
            f"{begin_area}."
        )

    if year:
        parts.append(
            f"Release year: {year}."
        )

    return " ".join(parts)


def main():
    source = (
        FINAL_DATASET
        if FINAL_DATASET.exists()
        else SPOTIFY_CLEAN
    )

    if not source.exists():
        raise FileNotFoundError(
            "Aucun dataset disponible."
        )

    df = pd.read_csv(
        source,
        low_memory=False,
    )

    required = [
        "track_id",
        "track_name",
        "artists",
        "track_genre",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : "
            f"{missing}"
        )

    print(
        "Dataset :",
        source,
    )
    print(
        "Morceaux :",
        f"{len(df):,}",
    )

    if "artist_country" in df.columns:
        coverage = (
            df["artist_country"]
            .notna()
            .mean()
            * 100
        )
        print(
            f"Couverture pays : "
            f"{coverage:.2f}%"
        )

    documents = [
        build_document(row)
        for _, row in df.iterrows()
    ]

    print(
        "\nChargement du modèle :",
        MODEL_NAME,
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "\nCréation des embeddings..."
    )

    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    save_embeddings(
        df,
        embeddings,
        model_name=MODEL_NAME,
    )

    print(
        "\nEmbeddings terminés."
    )
    print(
        "Dimensions :",
        embeddings.shape,
    )


if __name__ == "__main__":
    main()
