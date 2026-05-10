"""A module for converting CSVs to document chunks."""

import pandas as pd


SEASON_TYPES: dict[str, str] = {
    "REG": "regular season",
    "POST": "postseason"
}

def get_processed_string_column(df: pd.DataFrame) -> pd.Series:
    """Generates the text for each player-season entry in CSV.

    Args:
        filters: The filters, in SQL string syntax.
            Example: 'player_name = "Davante Adams" AND season = "2020"'
    """
    df["summary"] = df.apply(lambda x: f"{x['player_name']} was drafted in round {int(x['draft_round'])}, "
        f"pick {int(x['draft_pick'])} of the {int(x['draft_year'])} draft by "
        f"{x['draft_team']}. "
        f"In the {int(x['season'])} {SEASON_TYPES[x['season_type']]}, {x['player_name']} "
        f"played the {x['position']} position for {x['team']}. "
        f"He had {int(x['passing_yards'])} passing yards, "
        f"{int(x['pass_touchdown'])} passing touchdowns, and "
        f"{int(x['interception'])} interceptions. "
        f"He had {int(x['rush_attempts'])} rush attempts for "
        f"{int(x['rushing_yards'])} rushing yards and "
        f"{int(x['rush_touchdown'])} rushing touchdowns. "
        f"He had {int(x['receptions'])} receptions on "
        f"{int(x['targets'])} targets for "
        f"{int(x['receiving_yards'])} receiving yards and "
        f"{int(x['receiving_touchdown'])} receiving touchdowns.", axis=1)

    return df["summary"]


