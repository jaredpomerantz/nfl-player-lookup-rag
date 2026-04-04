"""A module for interfacing with the vector store."""

from nfl_player_lookup_rag.data_model import DocumentChunk
import boto3


class VectorInterface:
    """The VectorInterface class.

    This object is responsible for interfacing with the Vector Store. It will
    take in CSV data, generate embeddings with the embedding model, store it
    in the Vector Store, and retrieve the proper embeddings using the embedding
    model.
    """

    def __init__(self, embedding_model_name: str, vector_store_location: str) -> None:
        """Initializes the VectorInterface.

        Args:
            embedding_model_name: The name of the embedding model to use.
            vector_store_location: The location of the vector store.
        """
        self.embedding_model_name = embedding_model_name

        self.vector_store_location = vector_store_location
        self.vector_store_client = boto3.client(
            self.vector_store_location, region_name="us-west-2"
        )

    def upload_to_vector_store(self, documents: list[DocumentChunk]) -> None:
        """Uploads the documents to the vector store.

        Args:
            documents: The documents (text chunks) being uploaded.
        """
