# MIRA Agent

This repository is a monorepo for reproducible training of tool-using math, search, and code agents with slime. The first implemented vertical is MathAgent, whose initial gates are deliberately separated:

1. SFT smoke test: `Qwen/Qwen3-8B-Base` on converted ReTool trajectories.
2. RL smoke test: `Qwen/Qwen3-8B` on deduplicated DAPO-Math-17k with plain GRPO.
3. Only after both contracts pass: evaluate SFT-to-RL initialization and methods intended to improve over GRPO.

The two smoke tests do **not** yet form one training chain. This keeps failures in SFT formatting, online tool interaction, and GRPO optimization distinguishable.

## Pinned inputs

- slime: `THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`
- ReTool SFT: Hugging Face revision `74943ce52f389f16926702302fcff6255875cbb2`, file SHA-256 `152dd6fa574d1a095064e89230b550521c9fa6b00b22e816f0f6490b0c5ab72a`
- DAPO-Math-17k: Hugging Face revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`, file SHA-256 `534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`
- Qwen3-8B-Base: revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Qwen3-8B: revision `b968826d9c46dd6066d109eabc6255188de91218`
- slime CUDA 12 runtime: `slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c`

The raw datasets are already present under `data/raw/`; generated JSONL files, model weights, checkpoints, and logs are ignored by Git.

## Data contracts

ReTool conversion removes the dataset-specific instruction wrapper and turns every strict `<code>...</code>` plus `<interpreter>...</interpreter>` pair into Qwen/Hermes messages for one function named `code_interpreter`. Assistant reasoning and calls are trainable; tool observations are not. Rows with malformed or out-of-order tags are rejected rather than repaired heuristically. The audited result is 1,992 usable rows and 8 rejected rows. The old `<answer>` wrapper is normalized to an exact final line:

```text
Answer: \boxed{answer}
```

DAPO conversion scans all 1,791,700 physical rows, deduplicates by extracted question text after trimming only its outer whitespace, and drops every question for which more than one ground-truth answer exists. Internal TeX/whitespace is intentionally not normalized because it can be mathematically significant. The audited result is 17,255 unique questions, 12 conflicting questions dropped, and 17,243 usable rows. The original 100-fold row replication therefore does not alter sampling probability.

Generate data directly when needed:

```bash
cd /path/to/mira-agent
python -m math_agent.data retool \
  --input data/raw/retool_sft/train_2000.parquet \
  --output data/processed/retool_sft.jsonl

python -m math_agent.data dapo \
  --input data/raw/dapo_math_17k/data/dapo-math-17k.parquet \
  --output data/processed/dapo_math_rl.jsonl
```

Both commands print a machine-readable audit report, including rejected ReTool rows or conflicting DAPO labels.

## Python tool boundary

Online Python is stateless. Each call starts a fresh container with:

- no network;
- a read-only root filesystem and bounded temporary filesystem;
- a non-root user, all Linux capabilities dropped, and `no-new-privileges`;
- CPU, memory, PID, wall-time, input-size, and combined-output limits;
- escaped chat/tool delimiters in stdout before it returns to the model.

The default image is pinned as `python@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76`. The RL process needs Docker CLI access and permission to create only these ephemeral sandbox containers; run the training job itself in a dedicated allocation/container.

## Runtime environment

The configured runtime is the pinned slime CUDA 12 image above. It contains Python 3.12, PyTorch 2.11.0+cu129, SGLang 0.5.15.post1, Ray 2.58.0, and Megatron commit `1dcf0dafa884ad52ffb243625717a3471643e087`. The image's bundled slime checkout is older than this repository's pinned source, so the commands mount and execute `third_party/slime` from this repository while reusing the image's native dependencies.

Start an interactive environment with all GPUs visible:

```bash
cd /path/to/mira-agent
MIRA_AGENT_ROOT="$(pwd -P)"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MIRA_AGENT_ROOT}:/workspace/mira-agent" \
  -v /usr/bin/docker:/usr/local/bin/docker:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace/mira-agent \
  -e SLIME_ROOT=/workspace/mira-agent/third_party/slime \
  -e MEGATRON_ROOT=/root/Megatron-LM \
  slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c \
  bash
```

The Docker socket and CLI mounts are needed only for RL's sibling Python sandboxes. They can be omitted for SFT and checkpoint conversion. The smoke scripts refuse to run when visible GPUs are occupied, so this container does not terminate or reuse an unrelated training job.

## Model preparation

The smoke scripts fail closed if Hugging Face or Megatron checkpoints are missing. Download the two models at the pinned revisions, then convert each with slime's Qwen3-8B model definition:

```bash
cd /path/to/mira-agent
MIRA_AGENT_ROOT="$(pwd -P)"

hf download Qwen/Qwen3-8B-Base \
  --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --local-dir "${MIRA_AGENT_ROOT}/models/Qwen3-8B-Base"

hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir "${MIRA_AGENT_ROOT}/models/Qwen3-8B"

docker run --rm \
  --gpus device=0 \
  --ipc=host --shm-size=16g \
  -v "${MIRA_AGENT_ROOT}:/workspace/mira-agent" \
  -w /workspace/mira-agent \
  --entrypoint bash \
  slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c \
  -lc 'source third_party/slime/scripts/models/qwen3-8B.sh &&
       PYTHONPATH=/root/Megatron-LM:third_party/slime python \
         third_party/slime/tools/convert_hf_to_torch_dist.py \
         "${MODEL_ARGS[@]}" \
         --hf-checkpoint models/Qwen3-8B-Base \
         --save models/Qwen3-8B-Base_torch_dist &&
       chown -R "$(stat -c %u .):$(stat -c %g .)" models/Qwen3-8B-Base_torch_dist'

docker run --rm \
  --gpus device=0 \
  --ipc=host --shm-size=16g \
  -v "${MIRA_AGENT_ROOT}:/workspace/mira-agent" \
  -w /workspace/mira-agent \
  --entrypoint bash \
  slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c \
  -lc 'source third_party/slime/scripts/models/qwen3-8B.sh &&
       PYTHONPATH=/root/Megatron-LM:third_party/slime python \
         third_party/slime/tools/convert_hf_to_torch_dist.py \
         "${MODEL_ARGS[@]}" \
         --hf-checkpoint models/Qwen3-8B \
         --save models/Qwen3-8B_torch_dist &&
       chown -R "$(stat -c %u .):$(stat -c %g .)" models/Qwen3-8B_torch_dist'
```

Override `MEGATRON_ROOT`, checkpoint paths, and GPU settings through environment variables if the cluster layout differs.

## Smoke runs

Run these inside a slime-compatible environment with eight exclusively allocated GPUs. Each script refuses to stop/reuse an existing Ray cluster or run on GPUs that already expose compute processes.

```bash
cd /path/to/mira-agent
PYTHON_BIN=python bash scripts/run_sft_smoke.sh
PYTHON_BIN=python bash scripts/run_rl_smoke.sh
```

The SFT default is one update over 64 ReTool examples with Qwen3 multi-turn loss masking and a 16,384-token per-GPU ceiling. The full converted set has a maximum rendered length of 15,789 tokens; 21 examples exceed 8,192 tokens. The RL default is one on-policy update: 4 prompts × 8 samples, symmetric PPO clipping (`0.2/0.2`), an explicit low-variance KL loss (`0.001`), and one learner step per rollout. It intentionally enables none of dynamic sampling, partial rollout, TIS, speculative decoding, process reward, length reward, or tool-use bonus.

The RL data and rollout share one protocol: Qwen/Hermes `<tool_call>` JSON, exactly one `code_interpreter` function, and `Answer: \boxed{...}` as the terminal action. Model-generated tokens have loss mask 1; environment observations have loss mask 0.

## CPU verification

```bash
cd /path/to/mira-agent
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -X pycache_prefix=/tmp/mira-agent-pycache -m compileall -q math_agent tests
bash -n scripts/run_sft_smoke.sh scripts/run_rl_smoke.sh

MATH_AGENT_RUN_DOCKER_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_python_tool.py
```

The unit suite does not execute untrusted code on the host and does not require GPUs. A live sandbox integration check additionally requires the configured Docker image to be present.
