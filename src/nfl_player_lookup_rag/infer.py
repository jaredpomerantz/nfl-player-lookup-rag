"""Main function for NFL Player Lookup RAG tool."""

import argparse
import requests

from nfl_player_lookup_rag.embedding_handler import EmbeddingHandler
from nfl_player_lookup_rag.config import MODEL_LOCATION
from nfl_player_lookup_rag.doc_utils import generate_language_prompt

EMBEDDING_HANDLER = EmbeddingHandler(MODEL_LOCATION)

def answer_prompt(prompt: str, verbose: bool) -> None:
    query_result = EMBEDDING_HANDLER.query(prompt)
    language_prompt = generate_language_prompt(prompt, query_result)
    if verbose:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Parser")
    parser.add_argument("-v", help="Determines verbosity of logs.", action="store_true")

    args = parser.parse_args()

    while True:
        prompt = input("Prompt here: ").strip()
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        answer_prompt(prompt, verbose=args.v)

