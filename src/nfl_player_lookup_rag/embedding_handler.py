"""A module for an embedding handler."""

from pathlib import Path
import pandas as pd
import chromadb
from chromadb import EmbeddingFunction
from chromadb.api.types import QueryResult
from sentence_transformers import SentenceTransformer

from nfl_player_lookup_rag.doc_utils import get_processed_string_column
import uuid

EMBEDDING_CHUNK_SIZE: int = 5000
N_QUERY_RESULTS: int = 5


class LocalEmbeddingFunction(EmbeddingFunction):
    """An EmbeddingFunction that converts embeddings to ChromaDB input."""

    def __init__(self, model: SentenceTransformer) -> None:
        """Instantiates the EmbeddingFunction.

        Args:
            model: The model to be used to handle embeddings.
        """
        self.model = model

    def __call__(self, input_docs: list[str]) -> list[list[float]]:
        """Converts input document chunks to embeddings.

        Args:
            input_docs: The input docs to convert with the embedding model.
        """
        return self.model.encode(input_docs).tolist()


class EmbeddingHandler:
    """The EmbeddingHandler class definition.

    This class is responsible for storing an embedding model and converting
    document chunks to embeddings.
    """

    def __init__(self, local_embedding_model_path: Path | None = None) -> None:
        """Initializes the EmbeddingHandler."""

        self.embedding_model = None
        if local_embedding_model_path is not None:
            _embedding_model = SentenceTransformer(str(local_embedding_model_path))
            self.embedding_function = LocalEmbeddingFunction(_embedding_model)

        self.chromadb_client = chromadb.PersistentClient(path="./chroma_db")

        self.chromadb_collection = self.chromadb_client.get_or_create_collection(
            name="player-summaries", embedding_function=self.embedding_function
        )

    def convert_csv_to_embeddings(self, csv_path: Path) -> None:
        """Converts CSV to stored ChromaDB embeddings.

        Args:
            csv_path: The path to the CSV containing player data.
        """
        with csv_path.open() as file:
            df = pd.read_csv(file)

        player_summaries: pd.Series = get_processed_string_column(df)

        for i in range(0, len(player_summaries), EMBEDDING_CHUNK_SIZE):
            print(i)
            summary_chunk = player_summaries.iloc[i : i + EMBEDDING_CHUNK_SIZE]

            self.chromadb_collection.add(
                documents=summary_chunk.to_list(),
                ids=[str(uuid.uuid4()) for _ in range(len(summary_chunk))],
            )

    def query(self, prompt: str) -> QueryResult:
        """Queries ChromaDB for documents related to the prompt.

        Args:
            prompt: The user prompt being queried for.
        """
        return self.chromadb_collection.query(
            query_texts=[prompt], n_results=N_QUERY_RESULTS
        )
