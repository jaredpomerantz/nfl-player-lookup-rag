"""Test script for NFL Player Lookup RAG tool."""

import argparse
from pathlib import Path
import pandas as pd

from nfl_player_lookup_rag.infer import answer_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Parser")

    parser.add_argument(
        "--dataset_location", help="The location of the dataset to test on.", type=str
    )
    parser.add_argument(
        "--output_response_file", help="The location to output responses to.", type=str
    )

    args = parser.parse_args()

    dataset_location = Path(args.dataset_location).resolve()
    with dataset_location.open("r") as file:
        test_dataset = pd.read_csv(file)

    output_df = pd.DataFrame(data={"question": [], "answer": [], "expected_answer": []})

    for i in range(len(test_dataset)):
        question = test_dataset.iloc[i]["question"]
        response = answer_prompt(prompt=question)
        new_row = pd.DataFrame(
            data={
                "question": [question],
                "answer": [response],
                "expected_answer": [test_dataset.iloc[i]["expected_answer"]],
            }
        )
        print(f'{i},{question},{response},{test_dataset.iloc[i]["expected_answer"]}')
        output_df = pd.concat([output_df, new_row], ignore_index=True)

    output_location = Path(args.output_response_file).resolve()
    output_df.to_csv(output_location, index=False)
