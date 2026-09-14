import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


AI_DIR = PROCESSED_DIR / "ai"

CONTEXT_EMBEDDINGS_FILE = (
    AI_DIR / "context_embeddings.npy"
)

CONTEXT_META_FILE = (
    AI_DIR / "context_embeddings_meta.json"
)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Empreinte de l'ordre exact des track_id.

    Les embeddings sont alignés ligne par ligne avec le dataset.
    Si le dataset change, l'application détecte le décalage.
    """
    if "track_id" not in df.columns:
        raise ValueError(
            "Le dataset doit contenir track_id."
        )

    digest = hashlib.sha256()

    for track_id in df["track_id"].astype(str):
        digest.update(
            track_id.encode(
                "utf-8",
                errors="replace",
            )
        )
        digest.update(b"\n")

    return digest.hexdigest()


def save_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    model_name: str,
):
    AI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if len(matrix) != len(df):
        raise ValueError(
            "Le nombre d'embeddings ne correspond "
            "pas au nombre de morceaux."
        )

    np.save(
        CONTEXT_EMBEDDINGS_FILE,
        matrix,
    )

    metadata = {
        "model_name": model_name,
        "row_count": len(df),
        "dimension": int(
            matrix.shape[1]
        ),
        "dataset_fingerprint": (
            dataset_fingerprint(df)
        ),
    }

    CONTEXT_META_FILE.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_embeddings(
    df: pd.DataFrame,
    *,
    mmap=True,
):
    if not CONTEXT_EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Embeddings IA absents. Lance : "
            "python -m src.ai.build_context_embeddings"
        )

    if not CONTEXT_META_FILE.exists():
        raise FileNotFoundError(
            "Métadonnées des embeddings absentes."
        )

    metadata = json.loads(
        CONTEXT_META_FILE.read_text(
            encoding="utf-8"
        )
    )

    expected_rows = int(
        metadata["row_count"]
    )

    if expected_rows != len(df):
        raise RuntimeError(
            "Le dataset a changé depuis la création "
            "des embeddings. Relance : "
            "python -m src.ai.build_context_embeddings"
        )

    current_fingerprint = (
        dataset_fingerprint(df)
    )

    if (
        current_fingerprint
        != metadata[
            "dataset_fingerprint"
        ]
    ):
        raise RuntimeError(
            "L'ordre/les track_id du dataset ont changé. "
            "Reconstruis les embeddings avec : "
            "python -m src.ai.build_context_embeddings"
        )

    matrix = np.load(
        CONTEXT_EMBEDDINGS_FILE,
        mmap_mode="r" if mmap else None,
    )

    return matrix, metadata
