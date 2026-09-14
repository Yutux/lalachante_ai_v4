import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()


def listenbrainz_headers():
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "LalachanteMasterPrototype/1.0 (academic recommendation prototype)",
    }
    token = os.getenv("LISTENBRAINZ_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def post_json_with_retry(url, payload, max_retries=6, timeout=60):
    delay = 2
    for _ in range(max_retries):
        response = requests.post(
            url,
            json=payload,
            headers=listenbrainz_headers(),
            timeout=timeout,
        )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            print(f"Rate limit ListenBrainz. Nouvelle tentative dans {wait:.1f}s...")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        if 500 <= response.status_code < 600:
            print(f"Erreur serveur {response.status_code}. Nouvelle tentative dans {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Echec après {max_retries} tentatives sur {url}")
