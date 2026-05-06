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

r"""Basic OpenAI evaluation script for Live Math problems.

Example usage (public):
export OPENAI_API_KEY=<YOUR_API_KEY>
python generate_samples.py --input_dir=./putnam_like --num_samples=1
# BEGIN GOOGLE-INTERNAL
export OPENAI_API_KEY=<YOUR_API_KEY>
blaze run //third_party/eval_hub/putnam_like:generate_samples -- \
  --input_dir=./putnam_like \
  --num_samples=1 \
  --debug_k=2 \
  --alsologtostderr
# END GOOGLE-INTERNAL

Expected Input Directory Structure (--input_dir):
└── putnam_like_validated/
    └── {set_name}/
        └── {problem_name}/
            ├── metadata.json
            ├── question/
            │   ├── question.md
            │   └── (optional images)
            └── samples/
                └── ... (other model samples)
"""

import base64
import concurrent.futures
import datetime
import io
import os
import pathlib
import random
import time
from typing import Any, Dict, Sequence

from absl import app
from absl import flags
from absl import logging
import openai
from PIL import Image  # pylint: disable=g-importing-member
from PIL import UnidentifiedImageError  # pylint: disable=g-importing-member


FLAGS = flags.FLAGS
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    './putnam_like_validated',
    'Input directory for the final problem folders.',
)
_NUM_SAMPLES = flags.DEFINE_integer(
    'num_samples',
    1,
    'Number of samples to generate per problem.',
)
_MAX_WORKERS = flags.DEFINE_integer(
    'max_workers', 50, 'Maximum number of parallel API requests.'
)
_DEBUG_K = flags.DEFINE_integer(
    'debug_k',
    0,
    'Run in debug mode with K problems from each set. If 0, runs on all'
    ' problems.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', 'gpt-5.5', 'The name of the OpenAI model to use.'
)
_OPENAI_API_KEY = flags.DEFINE_string(
    'openai_api_key', None, 'OpenAI API key. If None, uses OPENAI_API_KEY env var.'
)
_OPENAI_BASE_URL = flags.DEFINE_string(
    'openai_base_url', None, 'OpenAI API base URL.'
)
_MAX_API_RETRIES = flags.DEFINE_integer(
    'max_api_retries', 3, 'Maximum number of retries for API calls.'
)
_TEMPERATURE = flags.DEFINE_float(
    'temperature', 1.0, 'Temperature for model sampling.'
)


def get_api_key() -> str | None:
  """Retrieves the OpenAI API key."""
  if _OPENAI_API_KEY.value:
    return _OPENAI_API_KEY.value
  try:
    return os.environ['OPENAI_API_KEY']
  except KeyError:
    logging.error('❌ FATAL: OPENAI_API_KEY environment variable not found.')
    return None


def load_system_prompt() -> str:
  """Loads the system prompt from the markdown file."""
  prompt_path = pathlib.Path(__file__).parent / 'system_prompt.md'
  try:
    return prompt_path.read_text(encoding='utf-8')
  except IOError as e:
    logging.error(
        '❌ FATAL: Could not load system prompt from %s: %s', prompt_path, e
    )
    return ''


def get_openai_sample(
    problem_path: pathlib.Path,
    sample_num: int,
    system_prompt: str,
) -> Dict[str, Any]:
  """Sends one problem to OpenAI and returns the raw response.

  Args:
      problem_path: Path to the problem directory.
      sample_num: The sample number.
      system_prompt: The system prompt to use for the API call.

  Returns:
      A dictionary containing the status and raw answer.
  """
  logging.info(
      '  -> Processing: %s from %s (Sample %d)',
      problem_path.name,
      problem_path.parent.name,
      sample_num,
  )
  try:
    # Load content
    question_path = problem_path / 'question' / 'question.md'
    question = question_path.read_text(encoding='utf-8')

    # Find images if they exist
    image_dir = problem_path / 'question'
    image_paths = sorted(
        list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    )
    image_parts = []
    for p in image_paths:
      try:
        with p.open('rb') as f:
          img = Image.open(f)
          if img:
            img.load()
            # Encode image to base64
            buffered = io.BytesIO()
            img.save(buffered, format='PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_parts.append(img_base64)
          else:
            logging.warning(
                'Failed to open image %s: Image.open() returned None', p
            )
      except (IOError, UnidentifiedImageError) as e:
        logging.warning('Failed to open image %s: %s', p, e)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('Unexpected error opening image %s: %s', p, e)

    # Construct the messages
    messages = [
        {'role': 'system', 'content': system_prompt},
    ]
    user_content = [{'type': 'text', 'text': question}]
    if image_parts:
      for img_b64 in image_parts:
        user_content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{img_b64}'},
        })
    messages.append({'role': 'user', 'content': user_content})

    # Call API with retry logic
    raw_answer_text = 'API Failure'  # Default value
    api_key = get_api_key()
    if not api_key:
      return {
          'status': 'FAILURE',
          'raw_answer': 'API Key Missing',
      }

    client = openai.OpenAI(
        api_key=api_key, base_url=_OPENAI_BASE_URL.value
    )

    for attempt in range(_MAX_API_RETRIES.value):
      try:
        response = client.chat.completions.create(
            model=_MODEL_NAME.value,
            messages=messages,
            temperature=_TEMPERATURE.value,
        )
        raw_answer_text = response.choices[0].message.content
        return {
            'status': 'SUCCESS',
            'raw_answer': raw_answer_text,
        }

      except openai.OpenAIError as e:
        logging.warning(
            '       - WARNING: API call failed on attempt %d: %s',
            attempt + 1,
            e,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            '       - WARNING: Unexpected error on attempt %d: %s',
            attempt + 1,
            e,
        )

      if attempt < _MAX_API_RETRIES.value - 1:
        time.sleep(2 ** attempt)  # Exponential backoff

    return {
        'status': 'FAILURE',
        'raw_answer': raw_answer_text,
    }

  except IOError as e:
    logging.error(
        '  -> ❌ ERROR reading files for %s: %s', problem_path.name, e
    )
    return {
        'status': 'FAILURE',
        'raw_answer': 'File Read Error',
    }


def save_sample(
    problem_path: pathlib.Path, sample_num: int, result: Dict[str, Any]
):
  """Saves the generated sample to the samples directory."""
  if result['status'] == 'SUCCESS':
    sample_dir = problem_path / 'samples' / _MODEL_NAME.value
    try:
      sample_dir.mkdir(parents=True, exist_ok=True)
      timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
      sample_filename = f'sample_{sample_num:03d}_{timestamp}.md'
      sample_filepath = sample_dir / sample_filename
      sample_filepath.write_text(result['raw_answer'], encoding='utf-8')
      logging.info(
          '    - Sample %d saved to %s', sample_num, sample_filepath
      )
    except IOError as e:
      logging.error(
          '  -> ❌ ERROR saving sample %d for %s: %s',
          sample_num,
          problem_path.name,
          e,
      )
  else:
    logging.warning(
        '  -> ⚠️ WARNING: Sample %d for %s failed: %s',
        sample_num,
        problem_path.name,
        result['raw_answer'],
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  system_prompt = load_system_prompt()
  api_key = get_api_key()

  if not api_key or not system_prompt:
    return

  input_dir_str = _INPUT_DIR.value

  # Adjust paths when running with blaze run
  workspace_dir = os.environ.get('BUILD_WORKSPACE_DIRECTORY')
  if workspace_dir:
    if not os.path.isabs(input_dir_str):
      input_dir_str = os.path.join(workspace_dir, input_dir_str)

  input_path = pathlib.Path(input_dir_str)

  if not input_path.is_dir():
    logging.error("❌ Error: Input directory '%s' not found.", input_path)
    return

  all_problems = []
  logging.info('Searching for problems in: %s', input_path)
  for root, _, files in os.walk(input_path):
    if 'metadata.json' in files:
      problem_path = pathlib.Path(root)
      # Basic check for question.md
      if (problem_path / 'question' / 'question.md').exists():
        all_problems.append(problem_path)
      else:
        logging.warning(
            "Skipping %s: Missing question/question.md", problem_path
        )
  all_problems.sort()

  if not all_problems:
    logging.info("No valid problems found in '%s'.", input_path)
    return
  else:
    logging.info("Found %d problems.", len(all_problems))

  if _DEBUG_K.value > 0:
    logging.info(
        '\n🔬 DEBUG MODE: Selecting K=%d random problems from each set...',
        _DEBUG_K.value,
    )
    problems_by_set = {}
    for p in all_problems:
      set_name = p.parent.name
      problems_by_set.setdefault(set_name, []).append(p)

    debug_problems = []
    for set_name, problem_list in sorted(problems_by_set.items()):
      total_in_set = len(problem_list)
      random.shuffle(problem_list)
      num_to_select = min(_DEBUG_K.value, total_in_set)
      selected = problem_list[:num_to_select]
      debug_problems.extend(selected)
      logging.info(
          "  - Set '%s': Selected %d/%d problems.",
          set_name,
          len(selected),
          total_in_set,
      )

    all_problems = debug_problems
    random.shuffle(all_problems)
    logging.info('Running on %d debug problems.', len(all_problems))

  jobs = []
  for p in all_problems:
    for i in range(_NUM_SAMPLES.value):
      jobs.append(
          {
              'problem_path': p,
              'sample_num': i + 1,
              'system_prompt': system_prompt,
          }
      )

  logging.info(
      '\nStarting sample generation for %d problems with %d samples each (Total'
      ' jobs: %d)...',
      len(all_problems),
      _NUM_SAMPLES.value,
      len(jobs),
  )

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_MAX_WORKERS.value
  ) as executor:
    future_to_job = {
        executor.submit(get_openai_sample, **job): job for job in jobs
    }
    for future in concurrent.futures.as_completed(future_to_job):
      job = future_to_job[future]
      problem_path = job['problem_path']
      sample_num = job['sample_num']
      try:
        result = future.result()
        save_sample(problem_path, sample_num, result)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            "❌ Job for '%s' sample %d generated an exception: %s",
            problem_path.name,
            sample_num,
            e,
        )

  logging.info("\n✅ Sample generation complete.")


if __name__ == '__main__':
  app.run(main)
