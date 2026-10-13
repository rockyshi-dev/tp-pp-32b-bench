# tp-pp-32b-bench

Measures whether tensor parallelism (TP4) or pipeline parallelism (PP4) serves
Qwen2.5-32B-Instruct with the lower time to first token (TTFT) and the lower gap between
output tokens, in vLLM 0.30.0 on four 24 GB GPUs **without GPU peer-to-peer**, at 1, 8 and
32 concurrent requests of 1536 prompt tokens and 128 output tokens.

## What it measures

The model has 61 GiB of bf16 weights and needs four 24 GB cards. vLLM can spread it two ways:

| Arm | Server flag (everything else is a vLLM default) | Cross-GPU traffic per forward pass |
|---|---|---|
| TP4 | `--tensor-parallel-size 4` | two all-reduces per layer, 128 for the 64 layers |
| PP4 | `--pipeline-parallel-size 4` | three activation hand-offs between stages |

When the cards cannot reach each other directly (`nvidia-smi topo -p2p r` reports "CNS" for
every pair), NCCL stages every transfer through host memory (its SHM transport). TP pays that
cost 128 times per pass, PP three times. The only variable between the arms is the
parallelism mode; PP4 also schedules differently by construction (a request takes its next
decode step only every fourth scheduler step), which is part of the mode.

**Requests.** Chat requests whose rendered prompt is exactly 1536 tokens (below the default
2048-token per-step budget, so a prompt is never split across scheduler steps), `max_tokens`
and `min_tokens` 128, temperature 0, streaming with usage. Each prompt is an exact excerpt of
the vLLM v0.30.0 source tree and begins with a different first token and a different first
16-token block, so the prefix cache cannot help either arm. Both arms get the same prompts in
the same order.

**Load.** A closed loop at concurrency c = 1, 8 and 32: c streams, each sending its next
request when the previous one finishes; one client process per eight streams, pinned to eight
physical cores of the other CPU socket. Per level: a warm-up of 2c requests, then 30, 80 and
160 measured requests.

**Per request.** TTFT: send to the first non-empty content chunk. Token gap: the median of the
gaps between consecutive content chunks, counted only when the request arrived as exactly 128
chunks (vLLM merges tokens into one chunk when a client falls behind). Mean TPOT is recorded
but is not a decode measurement at c > 1. **Per level and round:** the median over requests
of TTFT and of the token gap, and output tokens per second.

**Procedure.** Five rounds. A round starts the server in one arm, runs c = 1, 8, 32, stops
it, then does the same in the other arm; TP4 goes first in rounds 1, 3 and 5.

## Decision rule

For each metric (TTFT, token gap) and level, each round k gives r_k = median(TP4) / median(PP4):

- **TP better** if all five r_k < 0.90; **PP better** if all five > 1.10;
- **no material difference** if all five are within 0.90-1.10; **unresolved** otherwise.

A level gives verdicts only if it is valid in all five rounds (gates K, T, C below, both arms);
its token-gap verdict also needs gate G. The rows are checked top to bottom:

| Row | Condition |
|---|---|
| NOT-MEASURABLE | gate S, P, N or Q ended the study, or fewer than two levels are valid |
| TRADE-OFF | at least one "TP better" and at least one "PP better" |
| UNSTABLE | no trade-off, and at least one verdict is unresolved |
| NO-STORY | every verdict is "no material difference" |
| ONE-SIDED | otherwise: one arm better on at least one verdict, the other on none |

The expectation written before the first measurement is TRADE-OFF: PP4 the lower TTFT (a
1536-token TP4 prefill spends most of its time in 128 host-staged all-reduces of about 16 MB),
TP4 the lower token gap. `report` says whether a result meets it.

## Gates

| Gate | Pass condition | If it fails |
|---|---|---|
| S start | both arms start and answer | NOT-MEASURABLE |
| P premise | in the transport check, every NCCL channel of both arms is SHM, no P2P or NET line appears, and every multi-rank communicator reports `isAllDirectP2p 0 ... isAllCudaP2p 0` | the premise is false on this machine; no timing claim |
| K capacity | KV capacity (`num_gpu_blocks x block_size` from `vllm:cache_config_info`) >= 32 x 1664 tokens, and no preemption during the level | the level is invalid in both arms |
| T tokens | every measured request: 1536 prompt and 128 completion tokens; failures are dropped | more than 2% dropped: the level is rerun once (both arms, fresh starts, unused prompts), then invalid |
| C cache | prefix-cache hits below 1% of queried tokens over the level | as T |
| G chunks | at least 95% of measured requests arrived as 128 content chunks | the level's token-gap verdict is invalid |
| N placement | after warm-up, every server process has at least 95% of its anonymous memory on the GPUs' NUMA node and its CPU affinity on that socket, and NCCL's `/dev/shm/nccl-*` staging pages exist and all lie on that node | rerun the round (at most twice); then replace it (once per study); then NOT-MEASURABLE |
| Q quiet | no other compute process on any card of the machine; CPU time of other processes on the servers' socket at most 2% over the level; the four cards' SM clocks below 90% of their level median in at most 10% of samples; no errored telemetry sample | rerun the round (at most twice), then NOT-MEASURABLE |

Telemetry sampling intervals over 2.5 s are recorded, not gated; when such an interval sits at
the edge of a level, the CPU share is also computed over the level's own duration and the
larger value is gated.

## What is held fixed, and how

- **Placement.** The server runs through a wrapper that binds its CPUs to the socket the four
  GPUs hang off and sets an inherited preferred-node memory policy (`set_mempolicy`,
  `MPOL_PREFERRED`) for that node; `numactl` is not needed. vLLM's own custom all-reduce
  disables itself for more than two PCIe-only GPUs, so TP4 uses NCCL; every TP4 start must log
  each faster all-reduce path as disabled and NCCL (pynccl) in use.
- **Touch-and-free before every server start.** On a machine whose page cache fills the
  servers' node, a preferred allocation falls back to the other node instead of reclaiming
  cache, and gate N fails. So before every start one helper process bound to that node writes
  40 GB and frees it, which makes the kernel reclaim page cache then. **This evicts page cache
  on that node, including other users' cached files.** It touches no process memory: it
  refuses, allocating nothing, unless the node's free memory plus file pages exceed 44 GB, and
  it sets its own `oom_score_adj` to 1000. Afterwards the node's usable free memory must be at
  least 20 GB (one repeat), or the run stops before the server starts. `--touch-gb 0` turns it
  off (recorded and shown by `report`).
- **Compile cache.** All starts of a study share one compile cache (`--cache-dir`). The first
  start of each arm (start test) compiles; every timed start must load that cache, because a
  cold compile leaves a smaller KV pool, and must have the capacity gate K needs.
- **Environment.** Inherited `VLLM_*`, `NCCL_*`, `PYTORCH_*`, `OMP_*`, `TORCH_NCCL_*`,
  `TORCHINDUCTOR_*`, `TRITON_*`, unmanaged `CUDA_*` and `LD_PRELOAD` stop a start, because each
  would change what is measured.
- **Declared deviations.** Two NCCL settings are allowed, only if a default start fails with
  the matching signature, and then in both arms: `NCCL_CUMEM_HOST_ENABLE=0` after a segfault in
  `cuMemCreate`, `NCCL_P2P_DISABLE=1` after a hang of 30 minutes following "vLLM is using
  nccl==" with no later loading or compilation progress. Pass the failed start's directory with
  `--deviation-evidence`; `report` prints the deviation as a condition of the result.
- **Failures are never killed automatically.** A start that fails or hangs is left running
  and `MANUAL_ATTENTION.json` names its pid; the operator decides. A completed start is shut
  down with SIGTERM, and the next start waits until every process of the old server has exited
  and the cards' memory is back to its value before the start.

## Requirements

- Linux x86_64 with two CPU sockets (two NUMA nodes); four NVIDIA GPUs on one socket, 24 GB
  each; at least 10 physical cores on the other socket for the client and the monitor;
  `nvidia-smi` and `lscpu`.
- vLLM 0.30.0, the CUDA 12.9 build (`vllm==0.30.0+cu129`, commit `ced6857a`), with the torch,
  transformers and aiohttp it installs. Other versions refuse to run unless
  `--allow-other-versions` (recorded).
- `Qwen/Qwen2.5-32B-Instruct` at revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` in a
  local directory, for example
  `huggingface-cli download Qwen/Qwen2.5-32B-Instruct --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
  (61 GiB). It is served by its path.
- A clean git checkout of vLLM v0.30.0 to build the prompts.
- An otherwise idle machine: a compute process on any card, or a busy port, stops a start.

## How to run

```bash
# 1. prompts (CPU only, about a minute); the printed sha256 should be
#    19affec706d6b5b6afaa318016fc59fbbf1d101e1f0e82866f5e309a9522e281
python -m tpppbench prompts --vllm-source ~/vllm --model /models/Qwen2.5-32B-Instruct --output prompts.json
python -m tpppbench verify-prompts --prompts prompts.json --vllm-source ~/vllm --model /models/Qwen2.5-32B-Instruct

# 2. start test, settings start, transport check and the five timed rounds, in one go,
#    started as a session leader so that a closed terminal cannot stop it midway
export TPPP_MODEL=/models/Qwen2.5-32B-Instruct TPPP_PROMPTS=$PWD/prompts.json
setsid python -m tpppbench all --study runs/study-1 > study-1.log 2>&1 < /dev/null &

# or one phase at a time (settings and transport need a passing start test in the same
#    study; timed needs all three)
setsid python -m tpppbench start-test --study runs/study-1 ...
setsid python -m tpppbench settings   --study runs/study-1 ...
setsid python -m tpppbench transport  --study runs/study-1 ...
setsid python -m tpppbench timed      --study runs/study-1 ...

# 3. every number, recomputed from the raw files
python -m tpppbench report runs/study-1 --json runs/study-1/report.json
```

Options common to the phases: `--gpus` (default `0,1,2,3`), `--port` (8140), `--vllm` (the
`vllm` beside the Python interpreter; it must run on the same interpreter, and the version it
reports with `--version` must be the installed one), `--cache-dir` (`<study>/compile-cache`;
every phase of a study must use the same one), `--lock-dir` (env `TPPP_LOCK_DIR`, default
`/tmp/tpppbench-locks`; one lock file per GPU), `--lock PATH` (repeatable: extra lock files
held for the whole phase; `{gpu}` is expanded once per GPU, duplicates are taken once),
`--expect-cards` (8: the number of cards the machine must show at start; gate Q watches all of
them, and a card that disappears later fails gate Q), `--touch-gb` (40),
`--min-usable-free-gb` (20), `--deviation-evidence`, `--allow-other-versions`,
`--allow-other-prompts` (a prompt file that is not the reference one stops the run without
it), `--allow-other-model` (the model's config, generation config, weight index and tokenizer
files must match the reference revision's hashes without it), `--no-setsid-check`. A wrong
path or version stops the command with an error before any server starts. See `--help`.

To share GPUs safely with other tools that lock them, point `--lock` at their lock files:

```bash
python -m tpppbench timed --study runs/study-1 --lock /shared/locks/gpu{gpu}.lock
```

Expected wall time on 8x RTX 4090: about four hours for `all` (the timed phase is most of it;
reruns add to it).

## Files written

`<study>/<phase>/` for each phase, never overwritten (a phase refuses an existing directory):

- `run-manifest.json`: versions, topology (server, client and monitor CPUs, NUMA node,
  `nvidia-smi topo` output), prompt hash, deviations, options, command line.
- per server start (`<arm>/`, in `timed/` under `round-N-attempt-K/`): `manifest.json` (argv,
  environment without credential-named variables, generation config), `server.log`,
  `placement-exec.json`, `before-gpus.json`, `node-memory-before.txt`, `ready-identity.json`,
  `ready-cards.json`, `metrics-startup.txt`, `start-record.json` (KV capacity, compile mode),
  `startup-fields.json` (resolved settings), `shutdown.json`; beside it
  `<arm>-host-memory/` (the helper's readings before and after, and at server ready).
- per level (`c1/`, `c8/`, `c32/`): `warmup/` and `measured/client-*.jsonl` (every request:
  send time, every content chunk's timestamp and text, usage, finish reason, errors),
  `numa/<pid>.numa_maps` and `placement.json`, `metrics-before.txt`, `metrics-after.txt`,
  `telemetry.jsonl` and `telemetry-with-boundaries.json` (1 Hz: per-CPU counters, the server
  tree's CPU ticks, clock, power, temperature, utilisation and compute processes of every
  card), `gate-calculation.json`.
- `transport/<arm>/nccl.<host>.<pid>.log` (one NCCL debug file per process) and
  `transport.json` (gate P); `start-test/<arm>/rehearsal.json` (an untimed c = 32 level through
  the timed code path); `settings/<arm>/startup-fields.json`.
- `timed/`: `round-N-attempt-K/paired.json`, `retry-action.json` for a failed attempt,
  `<arm>-level-retry/` for a level rerun after gate T or C, `accepted-rounds.json`,
  `decision.json`.
- `result.json` at the end of each phase; `INCOMPLETE.json`, `STOPPED-host-memory.json` and
  `MANUAL_ATTENTION.json` when something stopped it.

## How to read the result

`python -m tpppbench report <study>` recomputes every request summary, gate, median, ratio,
rerun decision and the decision row from the raw files above, compares them with what the run
stored, and prints:

- inputs: versions, GPUs and CPU placement, whether the prompts are the reference prompts,
  deviations;
- the start test (KV capacity per arm), the resolved settings of both arms side by side
  (model runner, async scheduling, `max_num_seqs`, batch queue size, KV blocks, attention
  backend) and whether TP4 fell back to NCCL;
- gate P per arm, recomputed from the NCCL debug files;
- every failed round attempt and why; for each accepted round and level, the median TTFT and
  token gap of both arms, their ratio, output tokens per second and gates K, T, C, G; the
  largest telemetry interval, the largest foreign CPU share and the smallest node share over
  the accepted levels;
- the six verdicts with their five ratios, the number of valid levels, the row, and whether the
  written expectation is met.

Exit code 0: the recomputation agrees with what was stored, for every level
(`gate-calculation.json`: gates, drops, counters, medians, tokens per second, foreign CPU
share), every round attempt (`paired.json`: N and Q) and the decision (row, verdicts,
ratios); 1: a disagreement (printed as WARNING); 2: the timed phase is missing or incomplete, so there is no decision. `--json`
writes every value, including each accepted round's per-arm cells, for comparing two studies
cell by cell.

**Scope.** Every statement holds for 1536 prompt tokens to 128 output tokens (12:1); a
different ratio weighs prefill and decode differently. Nothing here is about goodput under a
latency target, mixed prompt lengths, tuned pipeline schedules, quantized models, or cards with
peer-to-peer or NVLink.

## Tests

```bash
python -m unittest
# also check a built prompt file against the reference hash
TPPP_PROMPTS=prompts.json python -m unittest tests.test_prompts
```

The tests use synthetic evidence in the formats the run writes (request records, Prometheus
text, `numa_maps`, `/proc` trees, NCCL debug lines, vLLM log lines) and real `/proc/zoneinfo`
and node `meminfo` text from a two-socket machine. They
start no server and no GPU process and need no network; one test starts a real helper process
that writes 1 MiB, and one runs a whole c = 1 level (client processes, `/metrics` scrapes,
`numa_maps` reads, the monitor) against a fake streaming server on localhost.

## License

MIT
