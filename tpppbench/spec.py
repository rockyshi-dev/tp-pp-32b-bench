"""Fixed parameters of the benchmark. Importing this module never imports vLLM or torch."""
import hashlib
import json
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-32B-Instruct'
MODEL_REVISION = '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
VLLM_VERSION = '0.30.0+cu129'
VLLM_COMMIT = 'ced6857afa0ea7b2e3f0846a62e1394e90f15607'
# sha256 of the prompt file that `python -m tpppbench prompts` builds from the vLLM checkout
# at VLLM_COMMIT and the tokenizer of MODEL at MODEL_REVISION.
REFERENCE_PROMPTS_SHA256 = '19affec706d6b5b6afaa318016fc59fbbf1d101e1f0e82866f5e309a9522e281'

# sha256 of the model's JSON files at MODEL_REVISION; a model directory must match them.
REFERENCE_MODEL_FILES = {
    'config.json': '9c6772f138ef9e5b3d1c18f2c87e451bbc01f5f1a4eabb36f9bf4f53829b903e',
    'generation_config.json': 'ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9',
    'model.safetensors.index.json': '0183543f2e6e40d3d1e863ed0b2a8c9cdaf2ee4045ce724a45cea30e9995a0af',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}

ARMS = ('TP4', 'PP4')
ARM_FLAGS = {'TP4': '--tensor-parallel-size', 'PP4': '--pipeline-parallel-size'}
PP_SIZE = {'TP4': 1, 'PP4': 4}
GPU_COUNT = 4
DEFAULT_GPUS = (0, 1, 2, 3)

LEVELS = {1: 30, 8: 80, 32: 160}          # concurrency -> measured requests per level
PROMPT_TOKENS = 1536
OUTPUT_TOKENS = 128
CAPACITY = 32 * (PROMPT_TOKENS + OUTPUT_TOKENS)  # KV tokens needed by 32 requests at once
ROUNDS = 5
CLIENT_CORES = 8
STREAMS_PER_CLIENT = 8

# Decision rule: per metric and level, five per-round ratios TP4/PP4.
TP_BETTER_BELOW = 0.90
PP_BETTER_ABOVE = 1.10

# Gates.
DROP_LIMIT = 0.02            # T: share of measured requests dropped
CACHE_HIT_LIMIT = 0.01       # C: prefix-cache hits / queries, strictly below
CHUNK_SHARE = 0.95           # G: share of requests with 128 content chunks
NODE_SHARE = 0.95            # N: share of anonymous memory on the servers' node
FOREIGN_CPU_LIMIT = 0.02     # Q: foreign CPU time on the servers' socket
CLOCK_FLOOR = 0.90           # Q: clock below 90% of the level median ...
CLOCK_SHARE = 0.10           # ... for at most 10% of samples
TELEMETRY_GAP_RECORD_S = 2.5
QUIET_RERUNS = 2             # reruns of a round after gate Q fails
PLACEMENT_RERUNS = 2         # reruns of a round after gate N fails, then one replacement per study

# Touch-and-free before every server start.
TOUCH_BYTES = 40 * 10**9
USABLE_FREE_AFTER_HELPER = 20 * 10**9

HTTP_TIMEOUT_S = 15
READY_TIMEOUT_S = 1800
NCCL_HANG_S = 1800
SHUTDOWN_TIMEOUT_S = 60

DECLARED_DEVIATIONS = {'NCCL_CUMEM_HOST_ENABLE': '0', 'NCCL_P2P_DISABLE': '1'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())
