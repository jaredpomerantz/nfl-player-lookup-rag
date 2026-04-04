"""A module for converting CSVs to documents."""

from abc import ABC, abstractmethod
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame, Column
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import sparknlp
from sparknlp.base import DocumentAssembler
from pyspark.ml import Pipeline

from nfl_player_lookup_rag.config import REPO_PATH

SPARK_SESSION = (
    SparkSession.builder.appName("Python Spark SQL basic example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
)

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

PLAYER_DATA_SUMMARY_COLUMN_NAME = "compiled_summary"

class DocGenerator(ABC):
    """A base class for converting CSV data to text documents."""

    def __init__(self, csv_path: Path) -> None:
        """Initializes the DocGenerator.

        Args:
            csv_path: The path to the CSV file.
        """
        self.csv_path = csv_path
        self.df = SPARK_SESSION.read.csv(
            self.csv_path.name, header=True, inferSchema=True, sep=","
        )

    def get_processed_string_column(self, filters: str = "") -> Column:
        """Generates the text for each player-season entry in CSV.

        Args:
            filters: The filters, in SQL string syntax.
                Example: 'player_name = "Davante Adams" AND season = "2020"'
        """

        sub_df = self.df.filter(filters)
        compile_strings = udf(self.convert_df_row_to_document_chunk, StringType())
        sub_df = sub_df.withColumn(
            PLAYER_DATA_SUMMARY_COLUMN_NAME,
            compile_strings(*[F.col(stat) for stat in PLAYER_STAT_COLS]),
        )

        return sub_df[PLAYER_DATA_SUMMARY_COLUMN_NAME]
    
    def convert_text_to_document(self, player_data_summary: str) -> None:
        """Converts text to document."""

    @abstractmethod
    def convert_df_row_to_document_chunk(self, *row_col_inputs) -> str:
        """Converts a dataframe row to text by indexing its fields.

        Args:
            row_col_inputs: The inputs to convert to text.
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

    def convert_df_row_to_document_chunk(self, *row_col_inputs) -> str:
        """Converts dataframe rows to chunks by indexing their fields.

        For NFL Players, this will create a summary of the player stats for the
        year.

        Args:
            row_col_inputs: The inputs to convert to text.
        """
        row_cols = dict(zip(PLAYER_STAT_COLS, row_col_inputs))

        season_type = self._season_type_to_str_map[row_cols["season_type"]]
        content = (
            f"{row_cols['player_name']} was drafted in round {int(row_cols['draft_round'])}, "
            f"pick {int(row_cols['draft_pick'])} of the {int(row_cols['draft_year'])} draft by "
            f"{row_cols['draft_team']}. "
            f"In the {int(row_cols['season'])} {season_type}, {row_cols['player_name']} "
            f"played the {row_cols['position']} position for {row_cols['team']}. "
            f"He had {int(row_cols['passing_yards'])} passing yards, "
            f"{int(row_cols['pass_touchdown'])} passing touchdowns, and "
            f"{int(row_cols['interception'])} interceptions. "
            f"He had {int(row_cols['rush_attempts'])} rush attempts for "
            f"{int(row_cols['rushing_yards'])} rushing yards and "
            f"{int(row_cols['rush_touchdown'])} rushing touchdowns. "
            f"He had {int(row_cols['receptions'])} receptions on "
            f"{int(row_cols['targets'])} targets for "
            f"{int(row_cols['receiving_yards'])} receiving yards and "
            f"{int(row_cols['receiving_touchdown'])} receiving touchdowns."
        )

        return content
