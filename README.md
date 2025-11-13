# LLM Research Toolkit

## TokenProber

TokenProber is an interactive playground for inspecting token-by-token generation
from any Hugging Face causal language model. It shows the top candidate tokens
with their probabilities, lets you override the model's choice at each step, and
provides rollback/continue controls so you can experiment with different decoding
paths.

### Installation

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

### Running TokenProber

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

### Configuration

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

### Development notes

- The renderer emits colorized output when stdout is a TTY that supports ANSI escape
  codes. Set `NO_COLOR=1` to disable colors.
- GPU execution is supported; tensors are copied back to CPU before formatting, so you
  can run large models without modifying the UI layer.
- Public APIs are exposed from `TokenProber.__init__`, making it easy to embed the
  tool inside notebooks or other scripts.

Contributions and feature requests are welcome—feel free to open issues or PRs as you
extend the toolkit.

## TruthfulnessEvaluator

`TruthfulnessEvaluator` is an extensible pipeline that benchmarks the factual accuracy
of a local “under-test” model with help from a separate local judge model. The initial
dataset integration targets TruthfulQA, but the package is structured so new datasets,
prompt templates, or judge strategies can be added with minimal changes.

### Workflow

1. Load and normalize dataset examples (see `TruthfulnessEvaluator/datasets/`).
2. Run the under-test model locally and save its responses to JSONL for auditing.
3. Prompt the judge model with each question/answer pair to obtain verdicts
   (A = correct, B = incorrect, C = abstain, D = test model hit `max_new_tokens`,
   E = judge failed to parse).
4. Summarize verdict counts and render a quick ASCII visualization plus JSONL logs
   for both inference and judgement stages.

### Running the pipeline

```python
from pathlib import Path

from TruthfulnessEvaluator import PipelineConfig, run_truthfulness_evaluation
from TruthfulnessEvaluator.config import DatasetSelection, ModelSelection

config = PipelineConfig(
    dataset=DatasetSelection(name="truthfulqa", limit=50),
    test_model=ModelSelection(name="gpt2", engine="vllm"),
    judge_model=ModelSelection(name="gpt2", engine="transformers"),
    output_dir=Path("truthfulness_evaluation_results"),
    batch_size=8,
)

run_truthfulness_evaluation(config)
```

When invoking the CLI, reference the preset names backed by Python modules under
`TruthfulnessEvaluator/datasets/` and `TruthfulnessEvaluator/models/`:

```bash
python -m TruthfulnessEvaluator \
  --dataset truthfulqa \
  --test-model gpt2 \
  --judge-model gpt2 \
  --engine vllm \
  --limit 10 \
  --batch-size 8
```

Every run directory stores its resolved configuration at `config.json`. To resume a
previous attempt that halted mid-way, point the CLI to the existing run directory:

```bash
python -m TruthfulnessEvaluator --resume truthfulness_evaluation_results/run_20240101-120000
```

The pipeline inspects which artifacts (inference, judgements, summary) already exist
and continues from the first incomplete stage.

Artifacts such as inference traces, judgement files, and summary reports are written to
`truthfulness_artifacts/run_<timestamp>/`. Extendability hooks live in
`TruthfulnessEvaluator/datasets/`, `TruthfulnessEvaluator/inference.py`, and
`TruthfulnessEvaluator/judging.py`. Customize defaults by editing/adding preset Python
modules (e.g., tweak `datasets/truthfulqa.py` or create `models/my_model.py` that
returns tailored runners). If a preset sets `is_reasoning_model=True`, the pipeline
automatically strips `<think>...</think>` reasoning traces and only forwards the final
answer to the judge, handling missing opening/closing tags gracefully.

When running on multi-GPU machines (e.g., 4×A100s) and you do not pin a specific
`device`, transformers-based models default to `device_map="auto"` so Hugging Face can
shard weights across all GPUs. For even higher throughput, choose `--engine vllm`
(`ModelSelection(engine="vllm")` in code) to route inference through vLLM’s optimized
runtime; it automatically uses all visible GPUs via tensor parallelism. vLLM requires a
CUDA-enabled environment—if you are developing on a CPU-only machine, keep the engine
set to `transformers` locally and switch to `vllm` only on GPU hosts.

Use the `--batch-size` CLI parameter (or the matching `PipelineConfig.batch_size` field)
to send multiple prompts per generation call for both the tested and judge models,
significantly reducing wall-clock time on larger evaluations.
