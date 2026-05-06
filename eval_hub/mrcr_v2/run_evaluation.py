# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""A basic example script to run evaluation on MRCR V2.

This file also includes the mrcr_v2_metric function to compute the
score for the MRCR V2 task per example.
"""

import difflib
import os
import sys
from typing import Sequence

from absl import app
from absl import flags
import openai
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_path",
    "mrcr_v2_data.csv",
    "Path to the input CSV file (must have 'queries' and 'answer' columns).",
)
flags.DEFINE_string(
    "output_path",
    "results.csv",
    "Path to save the results CSV file.",
)
flags.DEFINE_string(
    "model_name",
    "gpt-5.5",
    "The name of the OpenAI model to use.",
)
flags.DEFINE_string(
    "openai_api_key",
    None,
    "OpenAI API key. If None, uses OPENAI_API_KEY env var.",
)
flags.DEFINE_string(
    "openai_base_url",
    None,
    "OpenAI API base URL.",
)


def mrcr_v2_metric(prediction: str, target: str) -> float:
  """Computes the MRCR V2 metric.

  This metric uses difflib SequenceMatcher to compute a notion of approximate
  edit distance between the target reference and the model's output, scaled
  to lie within [0, 1]. 1 is a perfect match, 0 is a non-match. Additionally,
  the metric score is 0 if the random string is not the first 12 characters of
  the output (after stripping whitespace). For outputs where there are multiple
  matches of the random string, we keep only the last one and only consider the
  substring of the output following this last match.

  Args:
    prediction: The model's output.
    target: The target output. Note that it contains the random hash string as a
      prefix.

  Returns:
    The MRCR V2 metric score contained in the interval [0, 1].
  """
  if not isinstance(prediction, str) or not prediction:
    return 0.0

  target = target.strip()
  # The target format is strictly: [12-char-hash][actual-content]
  if len(target) < 12:
    return 0.0

  random_hash = target[:12]
  target_ref = target[12:].strip()
  prediction = prediction.strip()

  # Find the *last* instance of the random hash in the prediction
  start_index = prediction.rfind(random_hash)

  if start_index == -1:
    return 0.0

  # Extract content immediately following the last hash
  # start_index + 12 skips past the hash itself.
  prediction_content = prediction[start_index + 12 :].strip()

  d = difflib.SequenceMatcher(a=target_ref, b=prediction_content)
  return d.ratio()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # --- Initialization ---
  api_key = FLAGS.openai_api_key or os.environ.get("OPENAI_API_KEY")
  if not api_key:
    print("❌ ERROR: OpenAI API key not found. Use --openai_api_key or set OPENAI_API_KEY.")
    sys.exit(1)

  base_url = FLAGS.openai_base_url or os.environ.get("OPENAI_BASE_URL")
  model_name = os.environ.get("OPENAI_MODEL_NAME") or FLAGS.model_name

  client = openai.OpenAI(api_key=api_key, base_url=base_url)

  if not os.path.exists(FLAGS.input_path):
    print(f"❌ ERROR: Input file not found: {FLAGS.input_path}")
    sys.exit(1)

  df = pd.read_csv(FLAGS.input_path)
  print(f"Loaded {len(df)} samples from {FLAGS.input_path}")
  print(f"Starting evaluation with model: {model_name}")

  # --- Main Eval Loop ---
  for index, row in df.iterrows():
    prompt = row["queries"]
    target = row["answer"]

    try:
      response = client.chat.completions.create(
          model=model_name,
          messages=[{"role": "user", "content": prompt}],
          temperature=1.0,
      )
      prediction = response.choices[0].message.content if response.choices[0].message.content else ""
    except openai.OpenAIError as e:
      print(f"ALERT! API error for sample {index + 1}/{len(df)}: {e}")
      prediction = ""

    if not prediction:
      print(f"ALERT! No response for sample {index + 1}/{len(df)}")

    score = mrcr_v2_metric(prediction, target)

    df.at[index, "prediction"] = prediction
    df.at[index, "score"] = score

    print(f"Processed sample {index + 1}/{len(df)}")

  # --- Final Output ---
  print(f"Completed. Average Score: {df['score'].mean():.4f}")
  df.to_csv(FLAGS.output_path, index=False)
  print(f"Results saved to {FLAGS.output_path}")


if __name__ == "__main__":
  app.run(main)
