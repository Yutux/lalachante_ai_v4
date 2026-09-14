import pandas as pd

from src.config import (
    FINAL_DATASET,
    SPOTIFY_CLEAN,
)
from src.ai.embedding_store import (
    load_embeddings,
)
from src.recommender.engine_ai import (
    fit_feature_space,
    recommend,
    clean_country,
)


def main():
    source = (
        FINAL_DATASET
        if FINAL_DATASET.exists()
        else SPOTIFY_CLEAN
    )

    if not source.exists():
        raise FileNotFoundError(
            "Dataset absent."
        )

    df = pd.read_csv(
        source,
        low_memory=False,
    )

    embeddings, metadata = (
        load_embeddings(df)
    )

    scaler, audio_matrix = (
        fit_feature_space(df)
    )

    genres = (
        df[
            "track_genre"
        ]
        .value_counts()
        .head(8)
        .index
        .tolist()
    )

    rows = []

    for genre_index, genre in enumerate(
        genres
    ):
        candidates = df[
            df[
                "track_genre"
            ] == genre
        ]

        if candidates.empty:
            continue

        reference = (
            candidates.sample(
                1,
                random_state=(
                    42
                    + genre_index
                ),
            )
            .iloc[0]
        )

        reference_country = (
            clean_country(
                reference.get(
                    "artist_country",
                    None,
                )
            )
        )

        for mode in [
            "proche",
            "equilibre",
            "decouverte",
        ]:
            result, diagnostics = (
                recommend(
                    df,
                    audio_matrix,
                    scaler,
                    embeddings,
                    liked_track_ids=[
                        reference[
                            "track_id"
                        ]
                    ],
                    preferred_genres=[
                        genre
                    ],
                    main_genre=genre,
                    discovery_mode=mode,
                    top_n=10,
                )
            )

            if result.empty:
                continue

            country_ratio = None

            if (
                reference_country
                and "artist_country"
                in result.columns
            ):
                country_ratio = (
                    result[
                        "artist_country"
                    ]
                    .map(
                        clean_country
                    )
                    .eq(
                        reference_country
                    )
                    .mean()
                )

            rows.append(
                {
                    "mode": mode,
                    "reference_track": (
                        reference[
                            "track_name"
                        ]
                    ),
                    "reference_artist": (
                        reference[
                            "artists"
                        ]
                    ),
                    "reference_genre": genre,
                    "reference_country": (
                        reference_country
                    ),
                    "mean_audio_similarity": (
                        result[
                            "audio_similarity_pct"
                        ].mean()
                        / 100
                    ),
                    "mean_ai_context_similarity": (
                        result[
                            "ai_context_similarity_pct"
                        ].mean()
                        / 100
                    ),
                    "same_genre_ratio": (
                        result[
                            "track_genre"
                        ]
                        .astype(str)
                        .eq(
                            str(genre)
                        )
                        .mean()
                    ),
                    "same_country_ratio": (
                        country_ratio
                    ),
                    "artist_diversity_ratio": (
                        result[
                            "artists"
                        ].nunique()
                        / len(result)
                    ),
                    "genre_diversity": (
                        result[
                            "track_genre"
                        ].nunique()
                    ),
                }
            )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        print(
            "Aucun résultat d'évaluation."
        )
        return

    pd.set_option(
        "display.max_columns",
        None,
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\n--- Moyennes par mode ---"
    )

    numeric = [
        "mean_audio_similarity",
        "mean_ai_context_similarity",
        "same_genre_ratio",
        "same_country_ratio",
        "artist_diversity_ratio",
        "genre_diversity",
    ]

    print(
        results
        .groupby(
            "mode"
        )[
            numeric
        ]
        .mean()
        .round(4)
        .to_string()
    )

    output = (
        source.parent
        / "evaluation_ai_results.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print(
        "\nSortie :",
        output,
    )


if __name__ == "__main__":
    main()
