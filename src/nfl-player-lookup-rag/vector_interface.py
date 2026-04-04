"""A module for interfacing with the vector store."""


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
