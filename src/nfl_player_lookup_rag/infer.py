"""Main function for NFL Player Lookup RAG tool."""

import argparse
import requests

from nfl_player_lookup_rag.embedding_handler import EmbeddingHandler
from nfl_player_lookup_rag.config import MODEL_LOCATION
from nfl_player_lookup_rag.doc_utils import generate_language_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Parser")

    parser.add_argument(
        "prompt", help="The prompt to address with the RAG pipeline.", type=str
    )
    parser.add_argument("-v", help="Determines verbosity of logs.", action="store_true")

    args = parser.parse_args()

    embedding_handler = EmbeddingHandler(MODEL_LOCATION)

    query_result = embedding_handler.query(args.prompt)
    language_prompt = generate_language_prompt(args.prompt, query_result)
    if args.v:
        print(language_prompt)

    response = requests.post(
        url="http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": language_prompt,
            "stream": False
        }
    )
    try:
        text_response = response.json()["response"]
        print("Model response:", text_response)
    except requests.exceptions.JSONDecodeError:
        print("Decode error. See full response text", response.text)

