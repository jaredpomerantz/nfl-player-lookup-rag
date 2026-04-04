"""A module for an embedding handler."""


class EmbeddingHandler:
    """The EmbeddingHandler class definition.

    This class is responsible for storing an embedding model and converting
    document chunks to embeddings.
    """

    def __init__(self, model_name: str) -> None:
        """Initializes the EmbeddingHandler.

        Args:
            model_name: The name of the model, regardless of namespace.
        """
