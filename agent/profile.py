"""User profile management — distilled facts persisted per session."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

PROFILES_DIR = Path("profiles")

class UserProfile(BaseModel):
    name: str | None = Field(default=None)
    frequent_topics: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    last_seen: str = Field(default="")

def load_profile(session_id: str) -> dict:
    """Load the user profile for a given session ID from disk.

    Returns an empty dict if no profile exists yet.
    """
    PROFILES_DIR.mkdir(exist_ok=True)
    path = PROFILES_DIR / f"{session_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}

def save_profile(session_id: str, profile: dict) -> None:
    """Persist the user profile dict to disk for the given session ID."""
    PROFILES_DIR.mkdir(exist_ok=True)
    path = PROFILES_DIR / f"{session_id}.json"
    path.write_text(json.dumps(profile, indent=2))

def update_profile(session_id: str, last_human: str, last_agent: str, model: BaseChatModel) -> None:
    """Extract new user facts from the latest exchange and persist the updated profile."""
    current_profile = load_profile(session_id)

    prompt = f"""Given this conversation exchange and the existing profile, extract any new facts about the user (name, interests, preferences).
Return ONLY a JSON object with the same structure as the profile.
Do not invent facts. If nothing new was learned, return the profile unchanged.

Existing profile:
{json.dumps(current_profile, indent=2)}

Last human message: {last_human}
Last agent message: {last_agent}
"""

    structured_llm = model.with_structured_output(UserProfile)
    try:
        updated_profile_obj = structured_llm.invoke(prompt)
        # Convert pydantic model to dict
        if isinstance(updated_profile_obj, UserProfile):
            updated_profile = updated_profile_obj.model_dump()
        else:
            updated_profile = updated_profile_obj # if it's already a dict depending on the fallback

        # update last_seen
        updated_profile["last_seen"] = datetime.now(timezone.utc).isoformat()
        save_profile(session_id, updated_profile)
    except Exception as e:
        print(f"Error updating profile: {e}")
        pass
