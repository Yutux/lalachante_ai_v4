import json
import tarfile
from pathlib import Path
from typing import Iterator


def find_entity_member(tar: tarfile.TarFile, entity_name: str):
    candidates = [
        member
        for member in tar.getmembers()
        if member.isfile()
        and (
            member.name == f"mbdump/{entity_name}"
            or member.name.endswith(f"/mbdump/{entity_name}")
            or member.name.endswith(f"/{entity_name}")
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Impossible de trouver mbdump/{entity_name} dans l'archive."
        )
    return max(candidates, key=lambda member: member.size)


def iter_json_dump(archive_path: Path, entity_name: str) -> Iterator[dict]:
    """Lit directement un dump JSON MusicBrainz depuis .tar.xz, ligne par ligne."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive introuvable : {archive_path}")

    with tarfile.open(archive_path, mode="r:xz") as tar:
        member = find_entity_member(tar, entity_name)
        stream = tar.extractfile(member)
        if stream is None:
            raise RuntimeError(f"Impossible d'ouvrir {member.name}")

        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {entity_name}, ligne {line_number}: {exc}"
                ) from exc
