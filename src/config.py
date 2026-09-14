from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# SPOTIFY
# ============================================================

SPOTIFY_RAW = (
    RAW_DIR / "spotify" / "dataset.csv"
)

SPOTIFY_CLEAN = (
    PROCESSED_DIR / "spotify_tracks_clean.csv"
)


# ============================================================
# MUSICBRAINZ
# ============================================================

MUSICBRAINZ_ARCHIVES = (
    RAW_DIR / "musicbrainz" / "archives"
)

MB_AREA_ARCHIVE = (
    MUSICBRAINZ_ARCHIVES / "area.tar.xz"
)

MB_ARTIST_ARCHIVE = (
    MUSICBRAINZ_ARCHIVES / "artist.tar.xz"
)

MB_RECORDING_ARCHIVE = (
    MUSICBRAINZ_ARCHIVES / "recording.tar.xz"
)

MB_RELEASE_GROUP_ARCHIVE = (
    MUSICBRAINZ_ARCHIVES / "release-group.tar.xz"
)


# MusicBrainz intermédiaire

MB_AREAS_CSV = (
    INTERMEDIATE_DIR / "musicbrainz_areas.csv"
)

MB_ARTISTS_CSV = (
    INTERMEDIATE_DIR / "musicbrainz_artists.csv"
)

MB_RECORDINGS_CSV = (
    INTERMEDIATE_DIR / "musicbrainz_recordings.csv"
)

MB_RELEASE_GROUPS_CSV = (
    INTERMEDIATE_DIR
    / "musicbrainz_release_groups.csv"
)

SPOTIFY_MB_MATCHED = (
    INTERMEDIATE_DIR
    / "spotify_musicbrainz_matched.csv"
)


# ============================================================
# LISTENBRAINZ
# ============================================================

LISTENBRAINZ_API = (
    "https://api.listenbrainz.org/1"
)


# ============================================================
# LISTENBRAINZ MAPPING
# ============================================================

LISTENBRAINZ_MAPPING_URL = (
    f"{LISTENBRAINZ_API}/metadata/lookup/"
)

MAPPING_CACHE_CSV = (
    INTERMEDIATE_DIR
    / "listenbrainz_mapping_cache.csv"
)

MAPPING_BATCH_SIZE = 25

MAPPING_CONFIDENCE_THRESHOLD = 0.85

MAPPING_BATCH_PAUSE_SECONDS = 0.25

MAPPING_MAX_RETRIES = 6

MAPPING_TIMEOUT_SECONDS = 60


# ============================================================
# LISTENBRAINZ POPULARITY
# ============================================================

LB_RECORDING_POPULARITY = (
    INTERMEDIATE_DIR
    / "listenbrainz_recording_popularity.csv"
)

LB_ARTIST_POPULARITY = (
    INTERMEDIATE_DIR
    / "listenbrainz_artist_popularity.csv"
)

LB_BATCH_SIZE = 100


# ============================================================
# PROCESSED DATA
# ============================================================

FINAL_DATASET = (
    PROCESSED_DIR
    / "recommendation_dataset.csv"
)

FEATURE_DATASET = (
    PROCESSED_DIR
    / "spotify_tracks_features.csv"
)

EVALUATION_RESULTS = (
    PROCESSED_DIR
    / "evaluation_results.csv"
)

USER_PREFERENCES = (
    DATA_DIR
    / "user_preferences.csv"
)


# ============================================================
# AUDIO FEATURES
# ============================================================

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
]


# ============================================================
# RECOMMENDER
# ============================================================

# Le contenu musical reste le signal principal.
RECOMMENDATION_WEIGHTS = {
    "content": 0.90,
    "preferred_genre": 0.07,
    "main_genre": 0.03,
}


# ============================================================
# POPULARITY / TREND
# ============================================================

# Désactivé dans la V1 pour éviter que
# la popularité domine les recommandations.
ENABLE_TREND_SCORE = False

TREND_WEIGHT = 0.05


# ============================================================
# DIVERSITY
# ============================================================

# Maximum 1 morceau par artiste
# dans les recommandations finales.
MAX_SAME_ARTIST = 1

# Maximum 50 % des recommandations
# appartenant au même genre.
MAX_SAME_GENRE_RATIO = 0.50