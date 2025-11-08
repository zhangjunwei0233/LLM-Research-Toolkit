# LLM Research Toolkit

TokenProber is an interactive playground for inspecting token-by-token generation
from any Hugging Face causal language model. It shows the top candidate tokens
with their probabilities, lets you override the model's choice at each step, and
provides rollback/continue controls so you can experiment with different decoding
paths.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment,
but any modern Python 3.12+ toolchain works.

```bash
uv sync
```

or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running TokenProber

TokenProber can be launched either as a module or through the package API:

```bash
uv run python -m TokenProber --config configs/TokenProber.yaml
```

or in a Python session:

```python
from TokenProber import run_probe_session
run_probe_session(["--config", "configs/TokenProber.yaml"])
```

While a session is active you can:

- Press `Enter` or the right arrow key to accept the highest-probability token.
- Enter a rank number (1-based) to pick a candidate from the displayed table.
- Type a token string that resolves to exactly one tokenizer piece to force it.
- Use the left arrow key to roll back the previous token.
- Type `/stop`, `quit`, or `exit` to finish the session.

## Configuration

All runtime options live in `configs/*.yaml`. A minimal example is included at
`configs/TokenProber.yaml`. The available keys are:

| key              | description                                                                 | default (see `default_config.yaml`) |
|------------------|-----------------------------------------------------------------------------|------------------------------------|
| `model_path`     | Hugging Face repo or local path of an AutoModelForCausalLM compatible model | `openai-community/gpt2`            |
| `prompt`         | Initial text prompt used to prime the model                                 | instructional sample text          |
| `device`         | Optional explicit device (`cpu`, `cuda`, `cuda:0`, `mps`, …)                | auto-detected                      |
| `torch_dtype`    | Torch dtype hint (`auto`, `float16`, `bfloat16`, etc.)                      | `auto`                             |
| `max_new_tokens` | Maximum number of tokens you can append during the session                  | `256`                              |
| `temperature`    | Softmax temperature applied before sampling                                 | `1.0`                              |
| `top_k_display`  | Number of high-probability candidates shown each step                       | `12`                               |

Unrecognized keys are ignored so you can keep notes or comments in the same file.

## Development notes

- The renderer emits colorized output when stdout is a TTY that supports ANSI escape
  codes. Set `NO_COLOR=1` to disable colors.
- GPU execution is supported; tensors are copied back to CPU before formatting, so you
  can run large models without modifying the UI layer.
- Public APIs are exposed from `TokenProber.__init__`, making it easy to embed the
  tool inside notebooks or other scripts.

Contributions and feature requests are welcome—feel free to open issues or PRs as you
extend the toolkit.
