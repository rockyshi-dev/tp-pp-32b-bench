"""Synthetic evidence in the formats the run writes. No GPU, no network, no server."""
import json
from pathlib import Path

from tpppbench.spec import CAPACITY, LEVELS, OUTPUT_TOKENS, PROMPT_TOKENS, write_json

TOPOLOGY = {'gpus': [0, 1, 2, 3], 'server_socket': 0, 'server_node': 0, 'server_cpus': [0, 1, 2, 3],
            'client_socket': 1, 'client_cpus': list(range(8, 16)), 'runner_cpus': [16, 17], 'card_count': 8}
SERVER_PID = 4242
BASE_NS = 10**12


def cache_metric(blocks=4000, block_size=16):
    return ('# TYPE vllm:cache_config_info gauge\n'
            f'vllm:cache_config_info{{block_size="{block_size}",enable_prefix_caching="True",engine="0",'
            f'gpu_memory_utilization="0.92",num_gpu_blocks="{blocks}"}} 1.0\n')


def metrics(queries=0.0, hits=0.0, preemptions=0.0, waiting=0.0):
    return (cache_metric() +
            f'vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} {queries}\n'
            f'vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} {hits}\n'
            f'vllm:num_preemptions_total{{engine="0",model_name="m"}} {preemptions}\n'
            f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {waiting}\n')


def request(i, ttft=0.5, gap=0.02, chunks=OUTPUT_TOKENS, prompt=PROMPT_TOKENS, completion=OUTPUT_TOKENS,
            send_ns=BASE_NS, error=None):
    first = send_ns + int(ttft * 1e9)
    stamps = [first + int(k * gap * 1e9) for k in range(chunks)]
    return {'id': i, 'send_ns': send_ns, 'content_ns': stamps, 'contents': ['x'] * chunks,
            'usage': {'prompt_tokens': prompt, 'completion_tokens': completion,
                      'total_tokens': prompt + completion},
            'finish_reason': 'length', 'done': error is None, 'error': error, 'status': 200,
            'end_ns': (stamps[-1] if stamps else first) + 1000}


def numa_maps(node0=1000, node1=0, shm0=200, shm1=0):
    lines = [f'7f0000000000 default anon={node0 + node1} dirty={node0 + node1} N0={node0}'
             + (f' N1={node1}' if node1 else '') + ' kernelpagesize_kB=4']
    if shm0 or shm1:
        lines.append('7f1000000000 default file=/dev/shm/nccl-abc123 mapped=10'
                     + (f' N0={shm0}' if shm0 else '') + (f' N1={shm1}' if shm1 else '') + ' kernelpagesize_kB=4')
    lines.append('7f2000000000 default file=/usr/lib/libc.so.6 mapped=100 N0=100 kernelpagesize_kB=4')
    return '\n'.join(lines) + '\n'


def sample(t_ns, busy, total, owned_ticks, clock=2500.0, apps=(), error=None, cards=8):
    if error:
        return {'monotonic_ns': t_ns, 'error': error}
    return {'monotonic_ns': t_ns, 'wall_ns': t_ns,
            'cpu': {str(c): {'busy': busy, 'total': total} for c in TOPOLOGY['server_cpus']},
            'owned': {str(SERVER_PID): {'ticks': owned_ticks, 'start': 99}},
            'gpus': [{'index': g, 'uuid': f'GPU-{g}', 'clock_mhz': clock, 'power_w': 100.0,
                      'temperature_c': 50.0, 'utilization_pct': 50.0, 'memory_mib': 1000.0} for g in range(cards)],
            'compute_apps': [dict(a) for a in apps]}


def samples_for(begin, end, foreign_ticks_per_s=0, step_s=1.0, clocks=None):
    """1 Hz samples enclosing [begin, end]. Each CPU accrues 100 ticks/s; the server tree
    accrues 50 ticks/s per CPU; foreign_ticks_per_s extra busy ticks are not owned."""
    out, t, k = [], begin - int(step_s * 1e9), 0
    n_cpu = len(TOPOLOGY['server_cpus'])
    while True:
        seconds = k * step_s
        busy = int((50 + foreign_ticks_per_s / n_cpu) * seconds)
        clock = clocks[k] if clocks and k < len(clocks) else 2500.0
        out.append(sample(t, busy, int(100 * seconds), int(50 * n_cpu * seconds), clock=clock))
        if t >= end:
            return out
        t += int(step_s * 1e9)
        k += 1


def write_level(directory, c, ttft=0.5, gap=0.02, rows=None, hits=0.0, preemptions=0.0, node0=1000, node1=0,
                shm1=0, foreign_ticks_per_s=0, telemetry=None, affinity=None):
    """A level directory as run.measured_level writes it."""
    directory = Path(directory)
    (directory / 'measured').mkdir(parents=True)
    (directory / 'numa').mkdir()
    if rows is None:
        n = LEVELS[c]
        rows = [request(i, ttft=ttft, gap=gap, send_ns=BASE_NS + i * 10**6) for i in range(n)]
    with (directory / 'measured' / 'client-0.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    (directory / 'metrics-before.txt').write_text(metrics(queries=1000.0, hits=0.0, preemptions=2.0))
    queries = 1000.0 + len(rows) * PROMPT_TOKENS
    (directory / 'metrics-after.txt').write_text(metrics(queries=queries, hits=hits, preemptions=2.0 + preemptions))
    pids = {str(SERVER_PID): (node0, node1, 200, shm1), str(SERVER_PID + 1): (node0, node1, 200, shm1)}
    placement = {}
    for pid, (a, b, s0, s1) in pids.items():
        (directory / 'numa' / f'{pid}.numa_maps').write_text(numa_maps(a, b, s0, s1))
        placement[pid] = {'start': 99, 'affinity': affinity or TOPOLOGY['server_cpus']}
    write_json(directory / 'placement.json', placement)
    begin = min(r['send_ns'] for r in rows)
    end = max(r['end_ns'] for r in rows)
    if telemetry is None:
        telemetry = samples_for(begin, end, foreign_ticks_per_s)
    write_json(directory / 'telemetry-with-boundaries.json', telemetry)
    return directory


def write_arm(directory, levels=tuple(LEVELS), blocks=4000, **per_level):
    directory = Path(directory)
    directory.mkdir(parents=True)
    (directory / 'metrics-startup.txt').write_text(cache_metric(blocks=blocks))
    for c in levels:
        kwargs = per_level.get(c, {}) if isinstance(per_level.get(c), dict) else {}
        write_level(directory / f'c{c}', c, **kwargs)
    return directory


def write_manifest(directory, phase):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'run-manifest.json', {
        'phase': phase, 'gpus': [0, 1, 2, 3], 'versions': {'vllm': '0.30.0+cu129', 'reference_versions': True},
        'prompts_sha256': 'abc', 'reference_prompts': False, 'deviations': {}, 'touch_bytes': 40 * 10**9,
        'min_usable_free_bytes': 20 * 10**9, 'topology': TOPOLOGY})


# Default TP4/PP4 medians per level used by the synthetic study (seconds):
# TTFT: PP4 faster; token gap: TP4 faster -> TRADE-OFF.
DEFAULT_SHAPE = {'TP4': {1: (0.60, 0.030), 8: (1.20, 0.035), 32: (2.60, 0.050)},
                 'PP4': {1: (0.45, 0.090), 8: (0.60, 0.095), 32: (0.95, 0.090)}}


def write_round(timed, number, attempt=0, shape=None, overrides=None):
    """One round attempt. overrides: {(arm, c): write_level kwargs}."""
    shape = shape or DEFAULT_SHAPE
    overrides = overrides or {}
    directory = Path(timed) / f'round-{number}-attempt-{attempt}'
    for arm in ('TP4', 'PP4'):
        base = directory / arm
        base.mkdir(parents=True)
        (base / 'metrics-startup.txt').write_text(cache_metric())
        for c in LEVELS:
            ttft, gap = shape[arm][c]
            kwargs = {'ttft': ttft, 'gap': gap, **overrides.get((arm, c), {})}
            write_level(base / f'c{c}', c, **kwargs)
    return directory


def store_cells(timed):
    """Write gate-calculation.json for every level and paired.json for every attempt, as
    the runner does, from the same evaluation code."""
    from tpppbench import analysis, gates
    timed = Path(timed)
    for attempt in sorted(timed.glob('round-*-attempt-*')):
        cells = {}
        for arm_dir in sorted(p for p in attempt.iterdir() if p.is_dir()):
            startup = (arm_dir / 'metrics-startup.txt').read_text()
            for level_dir in sorted(arm_dir.glob('c*')):
                cell = analysis.evaluate_level(level_dir, int(level_dir.name[1:]), startup, TOPOLOGY)
                write_json(level_dir / 'gate-calculation.json', cell)
                cells.setdefault(arm_dir.name, {})[level_dir.name[1:]] = cell
        first = {a: cells[a] for a in ('TP4', 'PP4')}
        retry = {a: cells[a + '-level-retry'] for a in ('TP4', 'PP4') if a + '-level-retry' in cells} or None
        paired, _ = gates.paired_round(0, 0, first, retry)
        write_json(attempt / 'paired.json', {'N': paired['N'], 'Q': paired['Q']})


def capacity_ok():
    return CAPACITY
