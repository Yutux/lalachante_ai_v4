import re
from unidecode import unidecode

FEAT_PATTERN = re.compile(
    r"\s*[\(\[]?\s*(feat\.?|ft\.?|featuring)\s+.*$",
    flags=re.IGNORECASE,
)


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = unidecode(str(value)).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"['’`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_featured_artist(value) -> str:
    if value is None:
        return ""
    return FEAT_PATTERN.sub("", str(value)).strip()


def primary_artist(value) -> str:
    if value is None:
        return ""
    text = strip_featured_artist(value)
    # Le Spotify Tracks Dataset utilise souvent ';' entre artistes.
    if ";" in text:
        text = text.split(";", 1)[0]
    return text.strip()


def make_track_key(track_name, artist_name) -> str:
    return f"{normalize_text(track_name)}|||{normalize_text(primary_artist(artist_name))}"
