"""
core/conversation.py — Conversation history persistence.

Serialises LangChain BaseMessage objects to JSON so the conversation
survives application restarts.
"""

import json
import logging
from pathlib import Path
from typing import List

from langchain.schema import BaseMessage
from langchain.schema.messages import message_to_dict, messages_from_dict

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Persists LangChain conversation history to a JSON file.

    Methods
    -------
    save(messages)   — serialise and write to disk
    load()           — read from disk, return list of BaseMessage
    clear()          — delete the state file
    """

    def __init__(self, state_file: str = "data/conversation.json") -> None:
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, messages: List[BaseMessage]) -> None:
        """Write a list of BaseMessage objects to the state file."""
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump([message_to_dict(m) for m in messages], f, indent=2)
        logger.debug("Saved %d messages to %s", len(messages), self.state_file)

    def load(self) -> List[BaseMessage]:
        """
        Load conversation history from disk.

        Returns an empty list if the file does not exist or is empty.
        """
        if not self.state_file.exists():
            return []
        if self.state_file.stat().st_size == 0:
            logger.debug("State file is empty — returning fresh history.")
            return []
        try:
            with self.state_file.open(encoding="utf-8") as f:
                data = json.load(f)
            messages = messages_from_dict(data)
            logger.debug("Loaded %d messages from %s", len(messages), self.state_file)
            return messages
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Could not parse conversation state (%s) — starting fresh.", exc)
            return []

    def clear(self) -> None:
        """Delete the conversation state file."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.debug("Cleared conversation state file: %s", self.state_file)
