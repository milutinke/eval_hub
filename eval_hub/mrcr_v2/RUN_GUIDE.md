# MRCR v2 Run Guide

This guide explains how to set up and run the MRCR (Multiple-choice Response Consistency and Recall) v2 evaluation using the refactored OpenAI-compatible scripts.

## 1. Prerequisites

Ensure you have `uv` installed. If not, follow the [uv installation guide](https://github.com/astral-sh/uv).

Setup the project environment:
```bash
uv sync
```

## 2. Configuration

Create a `.env` file in the project root to configure your model and endpoint. 

Example `.env`:
```env
OPENAI_BASE_URL=http://your-api-endpoint:8000/v1
OPENAI_API_KEY=your-api-key (or empty if not required)
OPENAI_MODEL_NAME=your-model-name
```

The scripts will automatically load these variables.

## 3. Downloading Data

Use the `download.sh` script to fetch the evaluation datasets. You can choose different size groups:
- `-s`: Small (<= 128K tokens)
- `-m`: Medium (128K - 1M tokens)
- `-l`: Large (> 1M tokens)

Example: Download small datasets to a local folder named `data`:
```bash
./eval_hub/mrcr_v2/download.sh ./data -s
```

This will create a `data/mrcr_v2` directory containing the CSV files.

## 4. Running Evaluation

Run the `run_evaluation.py` script using `uv run`. You can specify the input dataset and output result file using flags.

### Example: Running a specific small dataset
```bash
uv run eval_hub/mrcr_v2/run_evaluation.py \
  --input_path=data/mrcr_v2/mrcr_v2p1_8needle_upto_128K_dynamic_fewshot_text_style_fast.csv \
  --output_path=mrcr_results_8needle.csv
```

### Available Flags:
- `--input_path`: Path to the input CSV file.
- `--output_path`: Path to save the results CSV (default: `results.csv`).
- `--model_name`: Override the model name (default: `gpt-5.5` or `OPENAI_MODEL_NAME` env var).
- `--openai_api_key`: Override the API key.
- `--openai_base_url`: Override the Base URL.

## 5. Understanding Results

The output `results.csv` will contain:
- `prediction`: The raw response from the model.
- `score`: The MRCR metric score (0.0 to 1.0), where 1.0 is a perfect match.

The script will also print the **Average Score** to the console upon completion.
