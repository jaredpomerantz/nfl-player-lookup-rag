"""Module for data models."""

from typing import TypedDict


class DocumentChunk(TypedDict):
    """A class definition for document chunks to be embedded and stored."""

    item_id: str
    content: str


PLAYER_STAT_COLS = [
    "season_type"
    "player_name"
    "draft_round"
    "draft_pick"
    "draft_year"
    "draft_team"
    "season"
    "player_name"
    "position"
    "team"
    "passing_yards"
    "pass_touchdown"
    "interception"
    "rush_attempts"
    "rushing_yards"
    "rush_touchdown"
    "receptions"
    "targets"
    "receiving_yards"
    "receiving_touchdown"
    "player_id"
    "season"
    "season_type"
]
