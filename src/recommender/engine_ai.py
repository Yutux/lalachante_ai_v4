import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.config import AUDIO_FEATURES


AMBIANCE_PRESETS = {
    "🎉 Soirée": [0.85, 0.85, 0.75, 128, 0.10, 0.10],
    "😌 Chill": [0.55, 0.35, 0.55, 92, 0.50, 0.25],
    "🔥 Motivation": [0.70, 0.90, 0.72, 132, 0.10, 0.10],
    "🏃 Sport": [0.72, 0.92, 0.70, 138, 0.08, 0.08],
    "❤️ Romantique": [0.45, 0.40, 0.65, 90, 0.60, 0.15],
    "📚 Concentration": [0.30, 0.30, 0.45, 88, 0.50, 0.70],
    "🚗 Trajet": [0.65, 0.65, 0.65, 112, 0.25, 0.15],
    "🌙 Mélancolique": [0.35, 0.30, 0.20, 82, 0.65, 0.25],
}


MAX_SAME_ARTIST = 1
MAX_SAME_GENRE_RATIO = 0.60


def clean_country(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if not value or value in {
        "NAN",
        "NONE",
        "NULL",
    }:
        return None

    return value


def fit_feature_space(df):
    scaler = StandardScaler()

    matrix = scaler.fit_transform(
        df[AUDIO_FEATURES]
    )

    return scaler, matrix


# ============================================================
# ONBOARDING
# ============================================================

def _dominant_genre_countries(
    subset: pd.DataFrame,
):
    """
    Déduit depuis le catalogue les pays associés au genre.

    Il n'existe aucune table codée en dur comme :
       k-pop -> KR

    On n'applique le contexte géographique que si les données
    montrent réellement une concentration nette.
    """
    if "artist_country" not in subset.columns:
        return []

    countries = (
        subset["artist_country"]
        .map(clean_country)
        .dropna()
    )

    if len(countries) < 20:
        return []

    distribution = (
        countries
        .value_counts(
            normalize=True
        )
    )

    if distribution.empty:
        return []

    top_country = (
        distribution.index[0]
    )

    top_share = float(
        distribution.iloc[0]
    )

    # Le catalogue doit montrer un signal suffisamment net.
    if top_share < 0.40:
        return []

    selected = []
    cumulative = 0.0

    for country, share in distribution.items():
        selected.append(country)
        cumulative += float(share)

        if cumulative >= 0.80:
            break

        if len(selected) >= 3:
            break

    return selected


def representative_tracks(
    df,
    genre,
    limit=10,
    random_state=42,
):
    subset = (
        df[
            df["track_genre"]
            .astype(str)
            .str.lower()
            .eq(
                str(genre).lower()
            )
        ]
        .drop_duplicates(
            subset=[
                "track_name",
                "artists",
            ]
        )
        .copy()
    )

    if subset.empty:
        return subset

    dominant_countries = (
        _dominant_genre_countries(
            subset
        )
    )

    if dominant_countries:
        country_mask = (
            subset["artist_country"]
            .map(clean_country)
            .isin(
                dominant_countries
            )
        )

        contextual_subset = (
            subset[country_mask]
        )

        # On ne restreint que si on garde assez de choix.
        if len(contextual_subset) >= limit * 3:
            subset = contextual_subset

    if len(subset) > 5000:
        subset = subset.sample(
            5000,
            random_state=random_state,
        )

    n_clusters = min(
        limit,
        len(subset),
    )

    if n_clusters <= 1:
        return subset.head(limit)

    scaler = StandardScaler()

    x = scaler.fit_transform(
        subset[AUDIO_FEATURES]
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(x)
    centers = model.cluster_centers_

    selected_indices = []
    artists_seen = set()

    for cluster_id in range(
        n_clusters
    ):
        positions = np.where(
            labels == cluster_id
        )[0]

        if len(positions) == 0:
            continue

        distances = np.linalg.norm(
            x[positions]
            - centers[cluster_id],
            axis=1,
        )

        ranked_positions = positions[
            np.argsort(distances)
        ]

        chosen = None

        for position in ranked_positions:
            row = subset.iloc[position]

            artist = str(
                row.get(
                    "artists",
                    "",
                )
            ).lower()

            if artist not in artists_seen:
                chosen = (
                    subset.index[
                        position
                    ]
                )
                artists_seen.add(
                    artist
                )
                break

        if chosen is None:
            chosen = (
                subset.index[
                    ranked_positions[0]
                ]
            )

        selected_indices.append(
            chosen
        )

    return (
        subset.loc[
            selected_indices
        ]
        .head(limit)
        .copy()
    )


# ============================================================
# PROFILE
# ============================================================

def liked_rows(
    df,
    liked_track_ids,
):
    liked_ids = {
        str(value)
        for value
        in liked_track_ids
    }

    return df[
        df["track_id"]
        .astype(str)
        .isin(liked_ids)
    ]


def build_audio_profile(
    df,
    liked_track_ids,
    preferred_genres,
    ambiances=None,
):
    liked = liked_rows(
        df,
        liked_track_ids,
    )

    selected_genres = {
        str(value).lower()
        for value
        in preferred_genres
    }

    genre_rows = df[
        df["track_genre"]
        .astype(str)
        .str.lower()
        .isin(selected_genres)
    ]

    if genre_rows.empty:
        genre_vector = (
            df[AUDIO_FEATURES]
            .mean()
            .to_numpy(dtype=float)
        )
    else:
        genre_vector = (
            genre_rows[
                AUDIO_FEATURES
            ]
            .mean()
            .to_numpy(dtype=float)
        )

    if liked.empty:
        user_vector = (
            genre_vector
        )
    else:
        liked_vector = (
            liked[
                AUDIO_FEATURES
            ]
            .mean()
            .to_numpy(dtype=float)
        )

        # Les morceaux explicitement aimés dominent.
        user_vector = (
            0.88 * liked_vector
            + 0.12 * genre_vector
        )

    mood_vectors = [
        np.asarray(
            AMBIANCE_PRESETS[a],
            dtype=float,
        )
        for a in (
            ambiances or []
        )
        if a in AMBIANCE_PRESETS
    ]

    if mood_vectors:
        mood = np.mean(
            mood_vectors,
            axis=0,
        )

        user_vector = (
            0.94 * user_vector
            + 0.06 * mood
        )

    return user_vector


def build_context_profile(
    df,
    context_embeddings,
    liked_track_ids,
):
    liked_ids = {
        str(value)
        for value
        in liked_track_ids
    }

    mask = (
        df["track_id"]
        .astype(str)
        .isin(liked_ids)
        .to_numpy()
    )

    positions = np.flatnonzero(
        mask
    )

    if len(positions) == 0:
        return None

    vectors = np.asarray(
        context_embeddings[
            positions
        ],
        dtype=np.float32,
    )

    profile = vectors.mean(
        axis=0
    )

    norm = np.linalg.norm(
        profile
    )

    if norm == 0:
        return None

    return (
        profile / norm
    ).astype(
        np.float32
    )


def build_country_profile(
    df,
    liked_track_ids,
):
    if "artist_country" not in df.columns:
        return {}, {
            "coverage": 0.0,
            "concentration": 0.0,
        }

    liked = liked_rows(
        df,
        liked_track_ids,
    )

    if liked.empty:
        return {}, {
            "coverage": 0.0,
            "concentration": 0.0,
        }

    countries = [
        clean_country(value)
        for value
        in liked[
            "artist_country"
        ].tolist()
    ]

    known = [
        value
        for value in countries
        if value
    ]

    coverage = (
        len(known)
        / len(liked)
    )

    if not known:
        return {}, {
            "coverage": coverage,
            "concentration": 0.0,
        }

    counts = Counter(
        known
    )

    total = sum(
        counts.values()
    )

    profile = {
        country: (
            count / total
        )
        for country, count
        in counts.items()
    }

    concentration = max(
        profile.values()
    )

    return profile, {
        "coverage": coverage,
        "concentration": (
            concentration
        ),
    }


def build_year_profile(
    df,
    liked_track_ids,
):
    if "release_year" not in df.columns:
        return None, 0.0

    liked = liked_rows(
        df,
        liked_track_ids,
    )

    if liked.empty:
        return None, 0.0

    years = pd.to_numeric(
        liked[
            "release_year"
        ],
        errors="coerce",
    )

    known = years.dropna()

    coverage = (
        len(known)
        / len(liked)
    )

    if known.empty:
        return None, coverage

    return (
        float(
            known.median()
        ),
        coverage,
    )


# ============================================================
# POPULARITY
# ============================================================

def local_popularity(
    df,
):
    """
    Retourne un percentile [0,1] DANS LE PAYS.

    Aucune comparaison brute de population/marché n'est utilisée.
    """
    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    listener_column = None

    for candidate in [
        "lb_recording_listener_count",
        "lb_artist_listener_count",
    ]:
        if candidate in df.columns:
            listener_column = candidate
            break

    if listener_column is None:
        return result

    listeners = pd.to_numeric(
        df[listener_column],
        errors="coerce",
    )

    if "artist_country" not in df.columns:
        return result

    countries = (
        df["artist_country"]
        .map(clean_country)
    )

    valid = (
        listeners.notna()
        & countries.notna()
    )

    if not valid.any():
        return result

    temp = pd.DataFrame(
        {
            "listeners": (
                listeners[valid]
            ),
            "country": (
                countries[valid]
            ),
        }
    )

    ranked = (
        temp
        .groupby("country")[
            "listeners"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    result.loc[
        valid
    ] = ranked

    return result


# ============================================================
# ADAPTIVE WEIGHTS
# ============================================================

def adaptive_weights(
    *,
    liked_count,
    country_evidence,
    year_coverage,
    discovery_mode,
):
    """
    Les poids dépendent des données disponibles dans le profil.

    Pas de 60/15/15 fixe pour tous les utilisateurs.
    """
    profile_strength = min(
        1.0,
        liked_count / 4.0,
    )

    country_strength = (
        country_evidence[
            "coverage"
        ]
        * country_evidence[
            "concentration"
        ]
    )

    raw = {
        "audio": (
            1.15
            + 0.15
            * profile_strength
        ),
        "context": (
            1.00
            + 0.25
            * profile_strength
        ),
        "genre": 0.38,
        "country": (
            0.95
            * country_strength
        ),
        "era": (
            0.25
            * year_coverage
        ),
        "local_popularity": 0.12,
        "novelty": 0.0,
    }

    if discovery_mode == "proche":
        raw["genre"] *= 1.15
        raw["country"] *= 1.20

    elif discovery_mode == "decouverte":
        raw["genre"] *= 0.60
        raw["country"] *= 0.60
        raw["novelty"] = 0.42

    total = sum(
        raw.values()
    )

    if total <= 0:
        return raw

    return {
        key: value / total
        for key, value
        in raw.items()
    }


# ============================================================
# SCORING
# ============================================================

def score_candidates(
    df,
    audio_matrix,
    scaler,
    context_embeddings,
    *,
    liked_track_ids,
    preferred_genres,
    main_genre,
    ambiances,
    discovery_mode,
):
    user_audio = build_audio_profile(
        df,
        liked_track_ids,
        preferred_genres,
        ambiances,
    )

    user_audio_scaled = (
        scaler.transform(
            pd.DataFrame(
                [user_audio],
                columns=AUDIO_FEATURES,
            )
        )
    )

    audio_similarity = (
        cosine_similarity(
            user_audio_scaled,
            audio_matrix,
        )[0]
        + 1
    ) / 2

    context_profile = (
        build_context_profile(
            df,
            context_embeddings,
            liked_track_ids,
        )
    )

    if context_profile is None:
        context_similarity = (
            np.full(
                len(df),
                np.nan,
            )
        )
    else:
        # Les embeddings sont déjà normalisés.
        context_similarity = (
            np.asarray(
                context_embeddings
                @ context_profile,
                dtype=float,
            )
            + 1
        ) / 2

    country_profile, country_evidence = (
        build_country_profile(
            df,
            liked_track_ids,
        )
    )

    target_year, year_coverage = (
        build_year_profile(
            df,
            liked_track_ids,
        )
    )

    weights = adaptive_weights(
        liked_count=len(
            liked_track_ids
        ),
        country_evidence=(
            country_evidence
        ),
        year_coverage=(
            year_coverage
        ),
        discovery_mode=(
            discovery_mode
        ),
    )

    result = df.copy()

    result[
        "_audio_similarity"
    ] = audio_similarity

    result[
        "_context_similarity"
    ] = context_similarity

    genres = (
        result["track_genre"]
        .astype(str)
        .str.lower()
    )

    preferred = {
        str(value).lower()
        for value
        in preferred_genres
    }

    main = str(
        main_genre
    ).lower()

    result[
        "_genre_affinity"
    ] = np.where(
        genres.eq(main),
        1.0,
        np.where(
            genres.isin(
                preferred
            ),
            0.72,
            0.0,
        ),
    )

    result[
        "_country_affinity"
    ] = np.nan

    if (
        country_profile
        and "artist_country"
        in result.columns
    ):
        candidate_country = (
            result[
                "artist_country"
            ]
            .map(
                clean_country
            )
        )

        maximum = max(
            country_profile.values()
        )

        normalized_profile = {
            country: (
                share / maximum
            )
            for country, share
            in country_profile.items()
        }

        result[
            "_country_affinity"
        ] = candidate_country.map(
            normalized_profile
        )

    result[
        "_era_affinity"
    ] = np.nan

    if (
        target_year is not None
        and "release_year"
        in result.columns
    ):
        years = pd.to_numeric(
            result[
                "release_year"
            ],
            errors="coerce",
        )

        result[
            "_era_affinity"
        ] = np.exp(
            -(
                years
                - target_year
            ).abs()
            / 10.0
        )

    result[
        "_local_popularity"
    ] = local_popularity(
        result
    )

    result[
        "_novelty"
    ] = (
        (
            1.0
            - genres.isin(
                preferred
            ).astype(float)
        )
        * result[
            "_context_similarity"
        ].fillna(
            result[
                "_audio_similarity"
            ]
        )
    )

    signal_columns = {
        "audio": (
            "_audio_similarity"
        ),
        "context": (
            "_context_similarity"
        ),
        "genre": (
            "_genre_affinity"
        ),
        "country": (
            "_country_affinity"
        ),
        "era": (
            "_era_affinity"
        ),
        "local_popularity": (
            "_local_popularity"
        ),
        "novelty": (
            "_novelty"
        ),
    }

    numerator = np.zeros(
        len(result),
        dtype=float,
    )

    denominator = np.zeros(
        len(result),
        dtype=float,
    )

    for signal, column in (
        signal_columns.items()
    ):
        weight = weights.get(
            signal,
            0.0,
        )

        if weight <= 0:
            continue

        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        available = (
            values.notna()
        )

        numerator += (
            values
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            * weight
        )

        denominator += (
            available
            .to_numpy(
                dtype=float
            )
            * weight
        )

    denominator = np.where(
        denominator == 0,
        1.0,
        denominator,
    )

    result[
        "_relevance"
    ] = (
        numerator
        / denominator
    )

    diagnostics = {
        "weights": weights,
        "country_profile": (
            country_profile
        ),
        "country_evidence": (
            country_evidence
        ),
        "target_year": (
            target_year
        ),
        "year_coverage": (
            year_coverage
        ),
    }

    return result, diagnostics


# ============================================================
# MMR RE-RANKING
# ============================================================

def rerank_mmr(
    ranked,
    context_embeddings,
    *,
    top_n,
    pool_size=300,
    diversity_strength=0.10,
):
    """
    Maximal Marginal Relevance :
    garde une forte pertinence tout en évitant une liste
    composée de morceaux contextuellement quasi identiques.
    """
    if ranked.empty:
        return ranked

    pool = ranked.head(
        min(
            pool_size,
            len(ranked),
        )
    ).copy()

    pool_positions = (
        pool["_row_position"]
        .astype(int)
        .to_numpy()
    )

    pool_vectors = np.asarray(
        context_embeddings[
            pool_positions
        ],
        dtype=np.float32,
    )

    selected_local = []
    artist_counts = (
        defaultdict(int)
    )
    genre_counts = (
        defaultdict(int)
    )

    max_per_genre = max(
        1,
        math.ceil(
            top_n
            * MAX_SAME_GENRE_RATIO
        ),
    )

    remaining = list(
        range(
            len(pool)
        )
    )

    while (
        remaining
        and len(
            selected_local
        ) < top_n
    ):
        best = None
        best_score = (
            -float("inf")
        )

        for local_index in remaining:
            row = pool.iloc[
                local_index
            ]

            artist = str(
                row.get(
                    "artists",
                    "",
                )
            ).lower()

            genre = str(
                row.get(
                    "track_genre",
                    "",
                )
            ).lower()

            if (
                artist_counts[
                    artist
                ]
                >= MAX_SAME_ARTIST
            ):
                continue

            if (
                genre_counts[
                    genre
                ]
                >= max_per_genre
            ):
                continue

            relevance = float(
                row[
                    "_relevance"
                ]
            )

            redundancy = 0.0

            if selected_local:
                candidate_vector = (
                    pool_vectors[
                        local_index
                    ]
                )

                chosen_vectors = (
                    pool_vectors[
                        selected_local
                    ]
                )

                redundancy = float(
                    np.max(
                        chosen_vectors
                        @ candidate_vector
                    )
                )

                redundancy = (
                    redundancy + 1
                ) / 2

            mmr = (
                relevance
                - diversity_strength
                * redundancy
            )

            if mmr > best_score:
                best_score = mmr
                best = local_index

        if best is None:
            break

        selected_local.append(
            best
        )

        row = pool.iloc[
            best
        ]

        artist = str(
            row.get(
                "artists",
                "",
            )
        ).lower()

        genre = str(
            row.get(
                "track_genre",
                "",
            )
        ).lower()

        artist_counts[
            artist
        ] += 1

        genre_counts[
            genre
        ] += 1

        remaining.remove(
            best
        )

    # Fallback si contraintes trop fortes.
    if len(
        selected_local
    ) < top_n:
        for local_index in remaining:
            row = pool.iloc[
                local_index
            ]

            artist = str(
                row.get(
                    "artists",
                    "",
                )
            ).lower()

            if (
                artist_counts[
                    artist
                ]
                >= MAX_SAME_ARTIST
            ):
                continue

            selected_local.append(
                local_index
            )

            artist_counts[
                artist
            ] += 1

            if len(
                selected_local
            ) >= top_n:
                break

    return (
        pool.iloc[
            selected_local
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PUBLIC API
# ============================================================

def recommend(
    df,
    audio_matrix,
    scaler,
    context_embeddings,
    *,
    liked_track_ids,
    preferred_genres,
    main_genre,
    ambiances=None,
    discovery_mode="equilibre",
    top_n=10,
):
    scored, diagnostics = (
        score_candidates(
            df,
            audio_matrix,
            scaler,
            context_embeddings,
            liked_track_ids=(
                liked_track_ids
            ),
            preferred_genres=(
                preferred_genres
            ),
            main_genre=(
                main_genre
            ),
            ambiances=(
                ambiances or []
            ),
            discovery_mode=(
                discovery_mode
            ),
        )
    )

    liked_ids = {
        str(value)
        for value
        in liked_track_ids
    }

    scored[
        "_row_position"
    ] = np.arange(
        len(scored)
    )

    if liked_ids:
        scored = scored[
            ~scored[
                "track_id"
            ]
            .astype(str)
            .isin(
                liked_ids
            )
        ]

    ranked = (
        scored
        .sort_values(
            "_relevance",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "track_name",
                "artists",
            ]
        )
    )

    diversity_strength = {
        "proche": 0.07,
        "equilibre": 0.10,
        "decouverte": 0.15,
    }.get(
        discovery_mode,
        0.10,
    )

    result = rerank_mmr(
        ranked,
        context_embeddings,
        top_n=top_n,
        diversity_strength=(
            diversity_strength
        ),
    )

    if result.empty:
        return (
            result,
            diagnostics,
        )

    for source, target in [
        (
            "_relevance",
            "recommendation_score_pct",
        ),
        (
            "_audio_similarity",
            "audio_similarity_pct",
        ),
        (
            "_context_similarity",
            "ai_context_similarity_pct",
        ),
        (
            "_country_affinity",
            "country_affinity_pct",
        ),
        (
            "_local_popularity",
            "local_popularity_percentile",
        ),
    ]:
        if source in result.columns:
            result[
                target
            ] = (
                pd.to_numeric(
                    result[
                        source
                    ],
                    errors="coerce",
                )
                * 100
            ).round(1)

    return (
        result,
        diagnostics,
    )
