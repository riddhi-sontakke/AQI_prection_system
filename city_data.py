"""Local city dataset and model artifact resolution."""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


CITY_DATASETS = {
    "Bengaluru": "BengluruData.csv",
    "Delhi": "DelhiData.csv",
    "Mumbai": "MumbaiData.csv",
}

CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bengluru": "Bengaluru",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
}


def _city_token(city: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (city or "").strip().lower())


def canonical_city(city: str) -> str:
    """Return the supported display name for a city."""
    token = _city_token(city)
    if token in CITY_ALIASES:
        return CITY_ALIASES[token]
    supported = ", ".join(CITY_DATASETS)
    raise ValueError(f"Unsupported city '{city}'. Choose one of: {supported}.")


def available_cities() -> list[str]:
    return list(CITY_DATASETS.keys())


def city_key(city: str) -> str:
    return _city_token(canonical_city(city))


def resolve_city_dataset(city: str, data_path: Optional[str] = None) -> str:
    """
    Resolve the local CSV used for a city.

    If data_path is supplied, it is validated and returned. Otherwise the
    bundled data/<city> CSV is used.
    """
    if data_path:
        path = os.path.abspath(data_path)
    else:
        display_city = canonical_city(city)
        path = os.path.join(DATA_DIR, CITY_DATASETS[display_city])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Local dataset not found: {path}")
    return path


def artifact_dir_for_city(city: str, root: str = MODELS_DIR) -> str:
    return os.path.join(root, city_key(city))


def list_existing_city_artifacts(cities: Optional[Iterable[str]] = None) -> dict[str, str]:
    found = {}
    for city in cities or available_cities():
        path = artifact_dir_for_city(city)
        if os.path.isdir(path):
            found[city] = path
    return found
