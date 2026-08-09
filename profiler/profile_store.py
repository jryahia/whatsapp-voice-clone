"""ChromaDB-backed profile storage for WhatsApp Voice Clone profiles."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

try:  # chromadb >=1.x renamed InvalidCollectionException to NotFoundError
    from chromadb.errors import NotFoundError as CollectionNotFound
except ImportError:  # chromadb <1.x
    from chromadb.errors import InvalidCollectionException as CollectionNotFound

import structlog

from .analyzer import VoiceProfile

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "voice_profiles"


class ProfileStore:
    """ChromaDB-backed persistent storage for VoiceProfile objects.

    Each profile is stored as a JSON-serialized document with metadata
    containing the profile name and creation timestamp.
    """

    def __init__(self, persist_dir: str = "~/.voice_clone_profiles") -> None:
        """Initialize the profile store.

        Parameters
        ----------
        persist_dir:
            Directory path for ChromaDB persistent storage.
            Defaults to ~/.voice_clone_profiles.
        """
        self._persist_dir = str(Path(persist_dir).expanduser().resolve())
        logger.info("initializing_profile_store", persist_dir=self._persist_dir)

        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Get or create the collection
        try:
            self._collection = self._client.get_collection(COLLECTION_NAME)
            logger.debug("loaded_existing_collection", collection=COLLECTION_NAME)
        except (ValueError, CollectionNotFound):
            self._collection = self._client.create_collection(COLLECTION_NAME)
            logger.info("created_new_collection", collection=COLLECTION_NAME)

    @property
    def collection(self):
        """Expose the underlying ChromaDB collection for advanced queries."""
        return self._collection

    def save(self, profile: VoiceProfile) -> str:
        """Serialize a VoiceProfile and store it in ChromaDB.

        Parameters
        ----------
        profile:
            The VoiceProfile to persist.

        Returns
        -------
        str:
            The UUID assigned to the stored profile.
        """
        profile_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        # Serialize profile to JSON
        document = profile.to_json()

        metadata: dict[str, Any] = {
            "name": profile.name,
            "created_at": created_at,
        }

        self._collection.add(
            ids=[profile_id],
            documents=[document],
            metadatas=[metadata],
        )

        logger.info(
            "profile_saved",
            profile_id=profile_id,
            name=profile.name,
            created_at=created_at,
        )

        return profile_id

    def load(self, name: str) -> VoiceProfile | None:
        """Look up a VoiceProfile by name.

        Parameters
        ----------
        name:
            The profile name (stored in ChromaDB metadata).

        Returns
        -------
        VoiceProfile | None:
            The deserialized profile, or None if not found.
        """
        results = self._collection.get(
            where={"name": name},
            limit=1,
        )

        if not results or not results["ids"]:
            logger.info("profile_not_found", name=name)
            return None

        profile_id = results["ids"][0]
        document = results["documents"][0]
        metadata = results["metadatas"][0] if results["metadatas"] else {}

        logger.debug(
            "profile_loaded",
            profile_id=profile_id,
            name=name,
            created_at=metadata.get("created_at"),
        )

        return VoiceProfile.from_json(document)

    def list_profiles(self) -> list[dict]:
        """List all stored profiles.

        Returns
        -------
        list[dict]:
            Each dict contains keys: name, created_at, id.
            Returns empty list if no profiles exist.
        """
        results = self._collection.get()

        if not results or not results["ids"]:
            return []

        profiles: list[dict] = []
        for pid, metadata in zip(results["ids"], results["metadatas"] or []):
            profiles.append({
                "name": metadata.get("name", "unknown"),
                "created_at": metadata.get("created_at", ""),
                "id": pid,
            })

        # Sort by created_at descending (newest first)
        profiles.sort(key=lambda p: p.get("created_at", ""), reverse=True)

        return profiles

    def delete(self, name: str) -> bool:
        """Delete a profile by name.

        Parameters
        ----------
        name:
            The profile name to delete.

        Returns
        -------
        bool:
            True if a profile was deleted, False if not found.
        """
        # First find the profile IDs matching this name
        results = self._collection.get(
            where={"name": name},
        )

        if not results or not results["ids"]:
            logger.info("profile_not_found_for_deletion", name=name)
            return False

        ids_to_delete = results["ids"]
        self._collection.delete(ids=ids_to_delete)

        logger.info(
            "profile_deleted",
            name=name,
            ids_deleted=ids_to_delete,
        )

        return True
