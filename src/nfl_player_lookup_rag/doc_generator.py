"""A module for converting CSVs to documents."""

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from sparknlp.annotator.embeddings.albert_embeddings import AlbertEmbeddings
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from sparknlp.annotator import Tokenizer
from pyspark.ml import Pipeline

from nfl_player_lookup_rag.config import REPO_PATH

SPARK_SESSION = (
    SparkSession.builder.appName("Python Spark SQL basic example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
)

PLAYER_STAT_COLS = [
    "season_type",
    "player_name",
    "draft_round",
    "draft_pick",
    "draft_year",
    "draft_team",
    "season",
    "player_name",
    "position",
    "team",
    "passing_yards",
    "pass_touchdown",
    "interception",
    "rush_attempts",
    "rushing_yards",
    "rush_touchdown",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_touchdown",
    "player_id",
    "season",
    "season_type",
]

PLAYER_DATA_SUMMARY_COLUMN_NAME = "compiled_summary"

    
def convert_to_doc(*row_col_inputs) -> str:
    cols = [
        "season_type",
        "player_name",
        "draft_round",
        "draft_pick",
        "draft_year",
        "draft_team",
        "season",
        "player_name",
        "position",
        "team",
        "passing_yards",
        "pass_touchdown",
        "interception",
        "rush_attempts",
        "rushing_yards",
        "rush_touchdown",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_touchdown",
        "player_id",
        "season",
        "season_type",
    ]
    row_cols = dict(zip(cols, row_col_inputs))
    _season_type_to_str_map: dict[str, str] = {
        "REG": "regular season",
        "POST": "postseason",
    }

    season_type = _season_type_to_str_map[row_cols["season_type"]]
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

class NFLPlayerDocGenerator:
    """Converts Yearly Player Stats CSV to usable text."""

    def __init__(self) -> None:
        """Initializes the NFLPlayerDocGenerator.

        Args:
            csv_path: The path to the CSV file.
        """
        self.csv_path = REPO_PATH / "resources" / "yearly_player_stats_offense.csv"
        self.df = SPARK_SESSION.read.csv(
            str(self.csv_path), header=True, inferSchema=True, sep=","
        )

    def get_processed_string_column(self, filters: str = "1=1") -> DataFrame:
        """Generates the text for each player-season entry in CSV.

        Args:
            filters: The filters, in SQL string syntax.
                Example: 'player_name = "Davante Adams" AND season = "2020"'
        """

        sub_df = self.df.filter(condition=filters)
        compile_strings = udf(convert_to_doc, StringType())

        sub_df = sub_df.withColumn(
            PLAYER_DATA_SUMMARY_COLUMN_NAME,
            compile_strings(*[F.col(stat) for stat in PLAYER_STAT_COLS]),
        )

        return sub_df.select(PLAYER_DATA_SUMMARY_COLUMN_NAME)

    def convert_text_to_embeddings(self, player_data_summaries: DataFrame) -> DataFrame:
        """Converts text to document.

        Args:
            player_data_summaries: The dataframe containing player data summaries.
        """

        document_assembler = (
            DocumentAssembler()
            .setInputCol(PLAYER_DATA_SUMMARY_COLUMN_NAME)
            .setOutputCol("document")
        )

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("tokens")

        albert: AlbertEmbeddings = AlbertEmbeddings.pretrained()  # type: ignore
        embedder = albert.setInputCols(["document", "token"]).setOutputCol("embeddings")

        embeddings_finisher = (
            EmbeddingsFinisher()
            .setInputCols(["embeddings"])
            .setOutputCols(["finished_embeddings"])
            .setOutputAsVector(True)
            .setCleanAnnotations(False)
        )

        pipeline = Pipeline().setStages(
            [document_assembler, tokenizer, embedder, embeddings_finisher]
        )

        return pipeline.fit(player_data_summaries).transform(player_data_summaries)
