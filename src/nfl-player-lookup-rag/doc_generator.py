"""A module for converting CSVs to documents."""

from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

from .config import REPO_PATH


class DocGenerator(ABC):
    """A base class for converting CSV data to text documents."""

    def __init__(self, csv_path: Path) -> None:
        """Initializes the DocGenerator.

        Args:
            csv_path: The path to the CSV file.
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

    def generate_text_documents(
        self, indices: tuple[int, int] | None = None
    ) -> list[str]:
        """Generates the text for each player-season entry in CSV."""

        sub_df = self.df
        if indices:
            if all([isinstance(index, int) for index in indices]):
                start_ind = min(indices[0], len(self.df))
                end_ind = min(indices[1], len(self.df))
                sub_df = self.df.iloc[start_ind:end_ind]

        output: list[str] = []
        for i in range(len(sub_df)):
            row = sub_df.iloc[i]
            output.append(self.convert_df_row_to_text(row))

        return output

    @abstractmethod
    def convert_df_row_to_text(self, row: pd.Series) -> str:
        """Converts a dataframe row to text by indexing its fields.

        Args:
            row: The row to convert to text.
        """


class NFLPlayerDocGenerator(DocGenerator):
    """Converts Yearly Player Stats CSV to usable text."""

    def __init__(self) -> None:
        """Initializes the NFLPlayerDocGenerator.

        Args:
            csv_path: The path to the CSV file for NFL Player Stats.
        """
        super().__init__(REPO_PATH / "resources" / "yearly_player_stats_offense.csv")
        self._season_type_to_str_map: dict[str, str] = {
            "REG": "regular season",
            "POST": "postseason",
        }

    def convert_df_row_to_text(self, row: pd.Series) -> str:
        """Converts a dataframe row to text by indexing its fields.

        For NFL Players, this will create a summary of the player stats for the
        year.

        Args:
            row: The row to convert to text.
        """

        season_type = self._season_type_to_str_map[row["season_type"]]
        return (
            f"{row['player_name']} was drafted in round {row['draft_round']}, "
            f"pick {row['draft_pick']} of the {row['draft_year']} draft by the "
            f"{row['draft_team']}"
            f"In the {row['season']} {season_type}, {row['player_name']} "
            f"played the {row['position']} position for {row['team']}."
            f"He had {row['passing_yards']} passing yards, "
            f"{row['passing_tds']} passing touchdowns, and "
            f"{row['interception']} interceptions. "
            f"He had {row['rush_attempts']} rush attempts for "
            f"{row['rushing_yards']} rushing yards and "
            f"{row['rush_touchdown']} rushing touchdowns. "
            f"He had {row['receptions']} receptions on "
            f"{row['targets']} targets for "
            f"{row['receiving yards']} receiving yards and"
            f"{row['receiving_touchdown']} receiving touchdowns."
        )
