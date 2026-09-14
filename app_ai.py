from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import (
    FINAL_DATASET,
    SPOTIFY_CLEAN,
    USER_PREFERENCES,
)
from src.ai.embedding_store import (
    load_embeddings,
)
from src.recommender.engine_ai import (
    AMBIANCE_PRESETS,
    fit_feature_space,
    representative_tracks,
    recommend,
)


st.set_page_config(
    page_title=(
        "Lalachante AI"
    ),
    page_icon="🎵",
    layout="wide",
)


DISCOVERY_LEVELS = {
    "🎯 Rester proche de mes goûts": "proche",
    "⚖️ Un mélange": "equilibre",
    "🌱 Me faire découvrir": "decouverte",
}


AMBIANCES = list(
    AMBIANCE_PRESETS.keys()
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .hero {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.3rem;
    }
    .small-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: .9rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULTS = {
    "ai_step": 1,
    "ai_username": "",
    "ai_genres": [],
    "ai_main_genre": None,
    "ai_liked": [],
    "ai_ambiances": [],
    "ai_discovery_label": (
        "⚖️ Un mélange"
    ),
    "ai_profile": None,
    "ai_dialog_open": False,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[
            key
        ] = value


@st.cache_data
def load_dataset():
    source = (
        FINAL_DATASET
        if FINAL_DATASET.exists()
        else SPOTIFY_CLEAN
    )

    if not source.exists():
        return None

    return pd.read_csv(
        source,
        low_memory=False,
    )


@st.cache_resource
def load_models():
    df = load_dataset()

    if df is None:
        return (
            None,
            None,
            None,
            None,
        )

    scaler, audio_matrix = (
        fit_feature_space(df)
    )

    embeddings, metadata = (
        load_embeddings(df)
    )

    return (
        scaler,
        audio_matrix,
        embeddings,
        metadata,
    )


def save_profile(profile):
    USER_PREFERENCES.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "username": (
            profile[
                "username"
            ]
        ),
        "genres": "|".join(
            profile[
                "genres"
            ]
        ),
        "main_genre": (
            profile[
                "main_genre"
            ]
        ),
        "liked_track_ids": "|".join(
            str(value)
            for value
            in profile[
                "liked_track_ids"
            ]
        ),
        "ambiances": "|".join(
            profile[
                "ambiances"
            ]
        ),
        "discovery": (
            profile[
                "discovery"
            ]
        ),
        "engine": (
            "ai_v4"
        ),
        "updated_at": (
            datetime.now()
            .isoformat(
                timespec="seconds"
            )
        ),
    }

    if USER_PREFERENCES.exists():
        prefs = pd.read_csv(
            USER_PREFERENCES
        )
    else:
        prefs = pd.DataFrame()

    if (
        not prefs.empty
        and "username"
        in prefs.columns
        and profile[
            "username"
        ]
        in prefs[
            "username"
        ].astype(str).values
    ):
        mask = (
            prefs[
                "username"
            ].astype(str)
            == profile[
                "username"
            ]
        )

        for key, value in (
            row.items()
        ):
            prefs.loc[
                mask,
                key,
            ] = value
    else:
        prefs = pd.concat(
            [
                prefs,
                pd.DataFrame(
                    [row]
                ),
            ],
            ignore_index=True,
        )

    prefs.to_csv(
        USER_PREFERENCES,
        index=False,
    )


@st.dialog(
    "Créer ton profil musical IA",
    width="large",
)
def onboarding():
    df = load_dataset()

    if df is None:
        st.error(
            "Dataset absent."
        )
        return

    step = st.session_state[
        "ai_step"
    ]

    st.progress(
        step / 5
    )

    st.caption(
        f"Étape {step}/5"
    )

    genres = sorted(
        df[
            "track_genre"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if step == 1:
        st.subheader(
            "Quels genres écoutes-tu ?"
        )

        username = st.text_input(
            "Pseudo",
            value=st.session_state[
                "ai_username"
            ],
        )

        selected = st.pills(
            "Choisis au moins 3 genres",
            genres,
            selection_mode="multi",
            default=st.session_state[
                "ai_genres"
            ],
        )

        if st.button(
            "Continuer →",
            type="primary",
            use_container_width=True,
        ):
            if not username.strip():
                st.error(
                    "Entre un pseudo."
                )
            elif len(
                selected
            ) < 3:
                st.error(
                    "Choisis au moins 3 genres."
                )
            else:
                st.session_state[
                    "ai_username"
                ] = username.strip()

                st.session_state[
                    "ai_genres"
                ] = selected

                if (
                    st.session_state[
                        "ai_main_genre"
                    ]
                    not in selected
                ):
                    st.session_state[
                        "ai_main_genre"
                    ] = selected[0]

                st.session_state[
                    "ai_step"
                ] = 2

                st.rerun()

    elif step == 2:
        st.subheader(
            "Quel genre écoutes-tu le plus ?"
        )

        genres = st.session_state[
            "ai_genres"
        ]

        main_genre = st.radio(
            "Genre principal",
            genres,
            index=(
                genres.index(
                    st.session_state[
                        "ai_main_genre"
                    ]
                )
                if st.session_state[
                    "ai_main_genre"
                ]
                in genres
                else 0
            ),
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "← Retour",
                use_container_width=True,
            ):
                st.session_state[
                    "ai_step"
                ] = 1
                st.rerun()

        with c2:
            if st.button(
                "Voir les morceaux →",
                type="primary",
                use_container_width=True,
            ):
                st.session_state[
                    "ai_main_genre"
                ] = main_genre

                st.session_state[
                    "ai_step"
                ] = 3

                st.rerun()

    elif step == 3:
        genre = st.session_state[
            "ai_main_genre"
        ]

        st.subheader(
            f"Quels morceaux de {genre} "
            f"aimes-tu ?"
        )

        st.caption(
            "Les morceaux sont choisis selon leur "
            "diversité audio et, quand les données le "
            "permettent, selon le contexte géographique "
            "dominant réellement observé dans ce genre."
        )

        tracks = representative_tracks(
            df,
            genre,
            limit=10,
        )

        labels = {}

        for _, row in tracks.iterrows():
            country = (
                row.get(
                    "artist_country",
                    "",
                )
                if "artist_country"
                in tracks.columns
                else ""
            )

            suffix = (
                f" · {country}"
                if pd.notna(
                    country
                )
                and str(
                    country
                ).strip()
                not in {
                    "",
                    "None",
                    "nan",
                }
                else ""
            )

            label = (
                f"{row['track_name']} — "
                f"{row['artists']}"
                f"{suffix}"
            )

            labels[
                label
            ] = row[
                "track_id"
            ]

        chosen_labels = (
            st.pills(
                "Morceaux aimés",
                list(
                    labels.keys()
                ),
                selection_mode="multi",
            )
        )

        chosen_ids = [
            labels[label]
            for label
            in chosen_labels
        ]

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "← Retour",
                use_container_width=True,
            ):
                st.session_state[
                    "ai_step"
                ] = 2
                st.rerun()

        with c2:
            if st.button(
                "Continuer →",
                type="primary",
                use_container_width=True,
            ):
                if not chosen_ids:
                    st.error(
                        "Choisis au moins un morceau."
                    )
                else:
                    st.session_state[
                        "ai_liked"
                    ] = chosen_ids

                    st.session_state[
                        "ai_step"
                    ] = 4

                    st.rerun()

    elif step == 4:
        st.subheader(
            "Personnalisation"
        )

        ambiances = st.pills(
            "Ambiances facultatives",
            AMBIANCES,
            selection_mode="multi",
            default=st.session_state[
                "ai_ambiances"
            ],
        )

        discovery = st.radio(
            "Niveau de découverte",
            list(
                DISCOVERY_LEVELS.keys()
            ),
            index=list(
                DISCOVERY_LEVELS.keys()
            ).index(
                st.session_state[
                    "ai_discovery_label"
                ]
            ),
        )

        st.info(
            "Le pays n'est pas demandé. "
            "Il est inféré depuis les artistes que tu as choisis. "
            "Si tes goûts sont géographiquement mixtes, "
            "le moteur réduit automatiquement son importance."
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "← Retour",
                use_container_width=True,
            ):
                st.session_state[
                    "ai_step"
                ] = 3
                st.rerun()

        with c2:
            if st.button(
                "Continuer →",
                type="primary",
                use_container_width=True,
            ):
                st.session_state[
                    "ai_ambiances"
                ] = ambiances

                st.session_state[
                    "ai_discovery_label"
                ] = discovery

                st.session_state[
                    "ai_step"
                ] = 5

                st.rerun()

    else:
        profile = {
            "username": (
                st.session_state[
                    "ai_username"
                ]
            ),
            "genres": (
                st.session_state[
                    "ai_genres"
                ]
            ),
            "main_genre": (
                st.session_state[
                    "ai_main_genre"
                ]
            ),
            "liked_track_ids": (
                st.session_state[
                    "ai_liked"
                ]
            ),
            "ambiances": (
                st.session_state[
                    "ai_ambiances"
                ]
            ),
            "discovery": (
                DISCOVERY_LEVELS[
                    st.session_state[
                        "ai_discovery_label"
                    ]
                ]
            ),
        }

        st.subheader(
            "Profil prêt 🎵"
        )

        st.write(
            "**Genres :**",
            ", ".join(
                profile[
                    "genres"
                ]
            ),
        )

        liked = df[
            df[
                "track_id"
            ]
            .astype(str)
            .isin(
                [
                    str(value)
                    for value
                    in profile[
                        "liked_track_ids"
                    ]
                ]
            )
        ]

        columns = [
            "track_name",
            "artists",
            "track_genre",
        ]

        if (
            "artist_country"
            in liked.columns
        ):
            columns.append(
                "artist_country"
            )

        st.dataframe(
            liked[columns],
            hide_index=True,
            use_container_width=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "← Modifier",
                use_container_width=True,
            ):
                st.session_state[
                    "ai_step"
                ] = 4
                st.rerun()

        with c2:
            if st.button(
                "Créer les recommandations",
                type="primary",
                use_container_width=True,
            ):
                save_profile(
                    profile
                )

                st.session_state[
                    "ai_profile"
                ] = profile

                st.session_state[
                    "ai_dialog_open"
                ] = False

                st.session_state[
                    "ai_step"
                ] = 1

                st.rerun()


def main():
    st.sidebar.title(
        "🎵 Lalachante AI"
    )

    st.sidebar.caption(
        "Content-Based + IA contextuelle"
    )

    df = load_dataset()

    st.markdown(
        """
        <div class="hero">
            <h1>Recommandation musicale assistée par IA</h1>
            <p>
            Proximité audio + embeddings contextuels +
            MusicBrainz + popularité locale ListenBrainz.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.error(
            "Dataset absent."
        )
        return

    try:
        (
            scaler,
            audio_matrix,
            embeddings,
            metadata,
        ) = load_models()
    except Exception as exc:
        st.error(
            str(exc)
        )

        st.code(
            "python -m src.ai.build_context_embeddings"
        )

        return

    profile = st.session_state[
        "ai_profile"
    ]

    if profile is None:
        st.write(
            "Commence par quelques préférences. "
            "Le moteur n'a pas besoin d'historique utilisateur."
        )

        if st.button(
            "🎧 Créer mon profil",
            type="primary",
        ):
            st.session_state[
                "ai_dialog_open"
            ] = True

            st.rerun()

    else:
        st.subheader(
            f"Recommandations pour "
            f"{profile['username']}"
        )

        top_n = st.select_slider(
            "Nombre de recommandations",
            options=[
                5,
                10,
                15,
                20,
            ],
            value=10,
        )

        result, diagnostics = (
            recommend(
                df,
                audio_matrix,
                scaler,
                embeddings,
                liked_track_ids=(
                    profile[
                        "liked_track_ids"
                    ]
                ),
                preferred_genres=(
                    profile[
                        "genres"
                    ]
                ),
                main_genre=(
                    profile[
                        "main_genre"
                    ]
                ),
                ambiances=(
                    profile[
                        "ambiances"
                    ]
                ),
                discovery_mode=(
                    profile[
                        "discovery"
                    ]
                ),
                top_n=top_n,
            )
        )

        with st.expander(
            "🧠 Ce que l'IA a déduit du profil",
            expanded=True,
        ):
            c1, c2, c3 = (
                st.columns(3)
            )

            country_profile = (
                diagnostics[
                    "country_profile"
                ]
            )

            country_text = (
                ", ".join(
                    f"{country}: "
                    f"{share * 100:.0f}%"
                    for country, share
                    in sorted(
                        country_profile.items(),
                        key=lambda item: (
                            item[1]
                        ),
                        reverse=True,
                    )
                )
                if country_profile
                else "Non déterminé"
            )

            c1.metric(
                "Profil pays",
                country_text,
            )

            c2.metric(
                "Couverture pays des likes",
                (
                    f"{diagnostics['country_evidence']['coverage'] * 100:.0f}%"
                ),
            )

            c3.metric(
                "Année médiane",
                (
                    int(
                        diagnostics[
                            "target_year"
                        ]
                    )
                    if diagnostics[
                        "target_year"
                    ]
                    is not None
                    else "N/A"
                ),
            )

            weights = (
                diagnostics[
                    "weights"
                ]
            )

            weights_df = pd.DataFrame(
                [
                    {
                        "Signal": signal,
                        "Poids adaptatif (%)": (
                            round(
                                weight * 100,
                                1,
                            )
                        ),
                    }
                    for signal, weight
                    in weights.items()
                    if weight > 0
                ]
            )

            st.dataframe(
                weights_df,
                hide_index=True,
                use_container_width=True,
            )

        display_columns = [
            "track_name",
            "artists",
            "track_genre",
            "recommendation_score_pct",
            "audio_similarity_pct",
            "ai_context_similarity_pct",
        ]

        optional = [
            "artist_country",
            "artist_area_name",
            "release_year",
            "country_affinity_pct",
            "local_popularity_percentile",
        ]

        display_columns += [
            column
            for column
            in optional
            if column
            in result.columns
        ]

        display = (
            result[
                display_columns
            ]
            .rename(
                columns={
                    "track_name": "Titre",
                    "artists": "Artiste",
                    "track_genre": "Genre",
                    "recommendation_score_pct": "Score (%)",
                    "audio_similarity_pct": "Audio (%)",
                    "ai_context_similarity_pct": "Contexte IA (%)",
                    "artist_country": "Pays",
                    "artist_area_name": "Zone",
                    "release_year": "Année",
                    "country_affinity_pct": "Affinité pays (%)",
                    "local_popularity_percentile": "Popularité locale (%)",
                }
            )
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "La popularité affichée est un percentile "
            "au sein du pays. Le nombre brut d'auditeurs "
            "n'est pas utilisé pour classer les pays entre eux."
        )

        if st.button(
            "✏️ Modifier mon profil"
        ):
            st.session_state[
                "ai_dialog_open"
            ] = True

            st.session_state[
                "ai_step"
            ] = 1

            st.rerun()

    if st.session_state[
        "ai_dialog_open"
    ]:
        onboarding()


if __name__ == "__main__":
    main()
