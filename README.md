# nfl-player-lookup-rag
A RAG tool for NFL player stat lookup from the [NFL Player Stats Kaggle Dataset](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022). Currently, the following data can be queried for from each entry in the dataset:

- `player_name`
- `draft_round`
- `draft_pick`
- `draft_year`
- `draft_team`
- `season`
- `season_type`
- `position`
- `team`
- `passing_yards`
- `pass_touchdown`
- `interception`
- `rush_attempts`
- `rushing_yards`
- `rush_touchdown`
- `receptions`
- `targets`
- `receiving_yards`
- `receiving_touchdown`

## Setup
Download the UV package manager, if not available:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Create environment:
```
python -m venv .venv
source .venv/bin/activate
uv sync
```

This tool requires a locally hosted Ollama model; Gemma 3 4B is recommended for GPU or CPU-only environments. In Ubuntu, run the following commands:
```
curl -fsSL https://ollama.com/install.sh | sh
``

## Use
To infer, run the following to initiate the language model and prompt loop:
```
ollama run gemma3:4b
python src/nfl_player_lookup_rag/infer.py
```
