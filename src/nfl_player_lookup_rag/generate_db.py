"""DB Generation script for NFL Player Lookup RAG tool."""

from nfl_player_lookup_rag.embedding_handler import EmbeddingHandler
from nfl_player_lookup_rag.config import REPO_PATH, MODEL_LOCATION

if __name__ == "__main__":
    embedding_handler = EmbeddingHandler(MODEL_LOCATION)
    embedding_handler.convert_csv_to_embeddings(REPO_PATH / "resources" / "yearly_player_stats_offense.csv")
    print(embedding_handler.chromadb_collection.peek())
