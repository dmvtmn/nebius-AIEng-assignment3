"""User profile management — distilled facts persisted per session."""
from __future__ import annotations

import json
from pathlib import Path

PROFILES_DIR = Path("profiles")
PROFILES_DIR.mkdir(exist_ok=True)


def load_profile(session_id: str) -> dict:
    """Load the user profile for a given session ID from disk.

    Returns an empty dict if no profile exists yet.
    """
    path = PROFILES_DIR / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_profile(session_id: str, profile: dict) -> None:
    """Persist the user profile dict to disk for the given session ID."""
    path = PROFILES_DIR / f"{session_id}.json"
    path.write_text(json.dumps(profile, indent=2))


def format_profile(profile: dict) -> str:
    """Format the profile dict as a human-readable string for injection into system prompt."""
    if not profile:
        return "No user profile yet."
    lines = [f"- {k}: {v}" for k, v in profile.items()]
    return "User profile:\n" + "\n".join(lines)
