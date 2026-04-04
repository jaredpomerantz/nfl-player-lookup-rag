"""Main function for NFL Player Lookup RAG tool."""

from nfl_player_lookup_rag.doc_generator import NFLPlayerDocGenerator

if __name__ == "__main__":
    nfl_player_doc_generator = NFLPlayerDocGenerator()
    strings = nfl_player_doc_generator.generate_text_documents()
    print(strings)
