"""Pure gate and decision calculations. Missing or malformed evidence raises EvidenceError."""
import math
from pathlib import Path
import re
import statistics

from .spec import (ARMS, CACHE_HIT_LIMIT, CAPACITY, CHUNK_SHARE, CLOCK_FLOOR, CLOCK_SHARE, DROP_LIMIT,
                   FOREIGN_CPU_LIMIT, LEVELS, NODE_SHARE, OUTPUT_TOKENS, PLACEMENT_RERUNS, PP_BETTER_ABOVE,
                   PROMPT_TOKENS, QUIET_RERUNS, ROUNDS, TELEMETRY_GAP_RECORD_S, TP_BETTER_BELOW)


class EvidenceError(ValueError):
    pass


# ------------------------------------------------------------------ Prometheus text

def metric_samples(text, name):
    rows = {}
    for line in text.splitlines():
        match = re.fullmatch(re.escape(name) + r'(\{.*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?', line)
        if match:
            key = match[1] or ''
            value = float(match[2])
            if key in rows or not math.isfinite(value):
                raise EvidenceError(f'Invalid duplicate or non-finite metric: {name}')
            rows[key] = value
    if not rows:
        raise EvidenceError(f'Missing metric {name}')
    return rows


def delta(before, after, name):
    left, right = metric_samples(before, name), metric_samples(after, name)
    if left.keys() != right.keys() or any(right[k] < left[k] for k in left):
        raise EvidenceError(f'Counter reset or label set changed: {name}')
    return sum(right[k] - left[k] for k in left)


def cache_config(metrics):
    """Labels of vLLM's engine-wide `vllm:cache_config_info` gauge."""
    samples = metric_samples(metrics, 'vllm:cache_config_info')
    if len(samples) != 1:
        raise EvidenceError('Expected one engine-wide cache config')
    labels, value = next(iter(samples.items()))
    if value != 1:
        raise EvidenceError('Cache config info gauge must equal one')
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)="((?:[^"\\]|\\[\\"n])*)"(?:,|$)')
    body, pos, result = labels[1:-1], 0, {}
    while pos < len(body):
        match = pattern.match(body, pos)
        if not match or match[1] in result:
            raise EvidenceError('Malformed or duplicate cache config label')
        result[match[1]] = re.sub(r'\\([\\"n])', lambda m: '\n' if m[1] == 'n' else m[1], match[2])
        pos = match.end()
    if result.get('engine') != '0':
        raise EvidenceError('Expected cache configuration for engine 0')
    for name in ('num_gpu_blocks', 'block_size'):
        if not re.fullmatch(r'[1-9][0-9]*', result.get(name, '')):
            raise EvidenceError(f'Missing or invalid cache config {name}')
    return result


def capacity(metrics):
    """KV capacity in tokens of the whole engine: num_gpu_blocks x block_size."""
    config = cache_config(metrics)
    return int(config['num_gpu_blocks']) * int(config['block_size'])


# ------------------------------------------------------------------ gate P (transport)

# NCCL 2.29 prefixes every INFO line with "<host>:<pid>:<tid> [<cudaDev>] NCCL INFO ".
NCCL_PREFIX = re.compile(r'(\S+):(\d+):(\d+) \[(\d+)\] NCCL INFO ')
NCCL_ANY_PREFIX = re.compile(r'(\S+):(\d+):(\d+) \[(\d+)\] (?:\S+ )?NCCL (?:INFO|WARN)')
# SHM: "Channel %02d : %d[%d] -> %d[%d] via SHM/%s/%s" (no connection index);
# P2P: "Channel %02d/%01d : ... via P2P/...". A "[send]"/"[receive]" tag (network
# transports) is parsed so that it fails as non-SHM.
NCCL_CHANNEL = re.compile(r'Channel\s+(\d+)(?:/(\d+))?\s*:\s*(\d+)\[(\d+)\]\s*->\s*(\d+)\[(\d+)\]'
                          r'(?:\s*\[\w+\])?\s+via\s+(\S+)')
# Printed once per communicator during initialisation, before its "Init COMPLETE".
NCCL_P2P_TYPE = re.compile(r'Check P2P Type isAllDirectP2p (\d+) directMode (\d+) isAllCudaP2p (\d+)')
NCCL_COMPLETE = re.compile(r'comm (\S+) rank (\d+) nranks (\d+) cudaDev (\d+) nvmlDev (\d+) .*- Init COMPLETE')
# SHM host-buffer allocation, printed for every SHM send and receive setup; only the cuMem
# branch's "CUMEM Host Alloc" line names a NUMA node.
NCCL_HOST_BUFFER = re.compile(r'(?:CUMEM allocated shareable buffer|MMAP allocated shareable host buffer'
                              r'|CUMEM Host Alloc Size)')
NCCL_CUMEM_NUMA = re.compile(r'CUMEM Host Alloc Size \d+ pointer \S+ handle \S+ numa (-?\d+) dev (\d+)')
# NCCL_DEBUG_FILE=<dir>/nccl.%h.%p.log: one line-buffered file per process; %h has no '.'.
NCCL_FILE = re.compile(r'nccl\.([^.]+)\.(\d+)\.log')
NCCL_VERSION_LINE = re.compile(r'^(?:\[[^\]]*\] )?NCCL version \S+$')


def nccl_files(directory):
    """The per-process NCCL debug files of one transport start: {pid: text}."""
    logs = {}
    for path in sorted(Path(directory).glob('nccl.*.log')):
        match = NCCL_FILE.fullmatch(path.name)
        if not match:
            raise EvidenceError(f'Unexpected NCCL debug file name: {path.name}')
        logs[int(match[2])] = path.read_text()
    return logs


def _transport_file(pid, text):
    out = {'channels': [], 'p2p_lines': [], 'host_lines': [], 'numa': [], 'communicators': []}
    pending = {}
    for line in text.splitlines():
        if not line.strip() or NCCL_VERSION_LINE.match(line):
            continue
        prefix = NCCL_PREFIX.search(line) or NCCL_ANY_PREFIX.search(line)
        if not prefix or int(prefix[2]) != pid:
            raise EvidenceError(f'NCCL line not attributable to file pid {pid}: {line}')
        if re.search(r'\bvia\s+P2P', line):
            out['p2p_lines'].append(line)
        if re.search(r'Channel\s+\d+(?:/\d+)?\s*:', line) and re.search(r'\bvia\b', line):
            match = NCCL_CHANNEL.search(line)
            if not match:
                raise EvidenceError(f'Unparsed NCCL channel line: {line}')
            out['channels'].append({'channel': int(match[1]),
                                    'connection_index': int(match[2]) if match[2] else None,
                                    'source_rank': int(match[3]), 'target_rank': int(match[5]),
                                    'transport': match[7], 'cuda_dev': int(prefix[4]), 'pid': pid,
                                    'line': line})
        if NCCL_HOST_BUFFER.search(line):
            out['host_lines'].append({'cuda_dev': int(prefix[4]), 'pid': pid, 'line': line})
            found = NCCL_CUMEM_NUMA.search(line)
            if found:
                out['numa'].append({'numa': int(found[1]), 'dev': int(found[2]), 'pid': pid})
        check, complete = NCCL_P2P_TYPE.search(line), NCCL_COMPLETE.search(line)
        if check:
            pending.setdefault(prefix[3], []).append(
                {'isAllDirectP2p': int(check[1]), 'directMode': int(check[2]), 'isAllCudaP2p': int(check[3])})
        if complete:
            queue = pending.get(prefix[3])
            if not queue:
                raise EvidenceError(f'Communicator completed without a Check P2P Type line: {line}')
            out['communicators'].append({'comm': complete[1], 'rank': int(complete[2]),
                                         'nranks': int(complete[3]), 'cuda_dev': int(complete[4]),
                                         'pid': pid, **queue.pop(0)})
    if any(pending.values()):
        raise EvidenceError(f'Check P2P Type line without a completed communicator (pid {pid})')
    return out


def transport(logs, arm='TP4', worker_pids=None):
    """Gate P, the same for both arms, from the per-process NCCL debug files.

    Pass requires: every NCCL "Channel ... via" line is SHM/; no "via P2P" line; every
    communicator with more than one rank logged "Check P2P Type isAllDirectP2p 0 ...
    isAllCudaP2p 0", paired with its Init COMPLETE line in the same thread; each worker's
    file holds a multi-rank communicator and an SHM host-buffer line; all four devices own
    a multi-rank communicator. Channel lines are required across files, not per file: only
    the SHM sender prints one, so a pipeline stage that only receives may have none. A
    one-rank communicator has no peers, so its flags stay 1; it is recorded and exempt.
    A shared stdout log is never parsed: NCCL writes stdout without flushing, so lines of
    several workers can be cut and interleaved."""
    if arm not in ARMS:
        raise EvidenceError('Unknown transport arm')
    if not isinstance(logs, dict) or not logs:
        raise EvidenceError('Missing per-process NCCL debug files')
    missing_files = sorted(set(worker_pids or ()) - set(logs))
    if worker_pids is not None and (len(set(worker_pids)) != 4 or missing_files):
        raise EvidenceError(f'Worker NCCL debug file missing for pids {missing_files} '
                            f'(workers {sorted(worker_pids)})')
    per_file = {pid: _transport_file(pid, text) for pid, text in logs.items()}
    channels = [c for f in per_file.values() for c in f['channels']]
    p2p_lines = [x for f in per_file.values() for x in f['p2p_lines']]
    host_lines = [x for f in per_file.values() for x in f['host_lines']]
    numa = [x for f in per_file.values() for x in f['numa']]
    communicators = [x for f in per_file.values() for x in f['communicators']]
    ranks = {r[k] for r in channels for k in ('source_rank', 'target_rank')}
    if not channels or ranks != {0, 1, 2, 3}:
        raise EvidenceError('Missing NCCL channel lines for all four ranks')
    workers = sorted(worker_pids) if worker_pids is not None else sorted(logs)
    multi = [c for c in communicators if c['nranks'] > 1]
    devices = {c['cuda_dev'] for c in multi}
    per_worker = {str(pid): {'multi_rank_communicators': sum(c['nranks'] > 1
                                                             for c in per_file[pid]['communicators']),
                             'host_buffer_lines': len(per_file[pid]['host_lines']),
                             'channel_lines': len(per_file[pid]['channels'])} for pid in workers}
    no_direct = all(c['isAllDirectP2p'] == 0 and c['isAllCudaP2p'] == 0 for c in multi)
    all_shm = all(c['transport'].startswith('SHM/') for c in channels)
    checks = {'all_channels_shm': all_shm, 'no_p2p_line': not p2p_lines,
              'every_multi_rank_communicator_not_p2p': bool(multi) and no_direct,
              'communicator_on_every_device': devices >= {0, 1, 2, 3},
              'every_worker_file_has_multi_rank_communicator':
                  all(v['multi_rank_communicators'] > 0 for v in per_worker.values()),
              'every_worker_file_has_shm_host_buffer':
                  all(v['host_buffer_lines'] > 0 for v in per_worker.values()),
              'shm_host_buffer_logged': bool(host_lines)}
    return {'pass': all(checks.values()), 'arm': arm, 'checks': checks,
            'worker_pids': workers, 'per_worker': per_worker,
            'extra_files': sorted(str(p) for p in set(logs) - set(workers)),
            'channels': channels, 'p2p_lines': p2p_lines, 'communicators': communicators,
            'single_rank_communicators_exempt': [c for c in communicators if c['nranks'] == 1],
            # Informational: SHM lines carry no connection index, so a pair cannot be
            # attributed to the send/receive path versus a ring or tree.
            'adjacent_pairs_observed': [list(p) for p in sorted(
                {(c['source_rank'], c['target_rank']) for c in channels} & {(0, 1), (1, 2), (2, 3)})],
            'host_buffer_lines': host_lines,
            'host_numa': {'cumem_numa_lines': numa, 'numa_identifying_line_present': bool(numa)}}


# ------------------------------------------------------------------ gates T, C, G, K per level

def request_valid(row):
    usage = row.get('usage') or {}
    return (not row.get('error') and row.get('done') is True and row.get('status') == 200
            and usage.get('prompt_tokens') == PROMPT_TOKENS and usage.get('completion_tokens') == OUTPUT_TOKENS
            and row.get('ttft_s') is not None and math.isfinite(row['ttft_s'])
            and row['ttft_s'] >= 0 and len(row.get('content_ns', [])) > 0)


def level(rows, before, after, startup_metrics, concurrency):
    """Gates T, C, G and K and the level's medians, from the measured requests and the
    /metrics text before and after the level and at server start."""
    expected = LEVELS[concurrency]
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise EvidenceError('Missing or duplicate measured request records')
    kept = [r for r in rows if request_valid(r)]
    dropped = len(rows) - len(kept)
    queries = delta(before, after, 'vllm:prefix_cache_queries_total')
    hits = delta(before, after, 'vllm:prefix_cache_hits_total')
    preemptions = delta(before, after, 'vllm:num_preemptions_total')
    metric_samples(after, 'vllm:num_requests_waiting')
    if queries <= 0 or hits > queries:
        raise EvidenceError('Missing positive prefix-cache query accounting')
    gap_rows = [r for r in kept if r['content_chunks'] == OUTPUT_TOKENS and r['token_gap_s'] is not None]
    tpot_values = [r['mean_tpot_s'] for r in kept if r['mean_tpot_s'] is not None]
    for row in gap_rows:
        if not math.isfinite(row['token_gap_s']) or row['token_gap_s'] < 0:
            raise EvidenceError('Bad token gap')
    start, end = min(r['send_ns'] for r in rows), max(r['end_ns'] for r in rows)
    if end <= start:
        raise EvidenceError('Invalid measured window')
    # Gate G counts every measured request, including those dropped by gate T.
    chunks = sum(r.get('content_chunks') == OUTPUT_TOKENS for r in rows)
    kv = capacity(startup_metrics)
    result = {'K': kv >= CAPACITY and preemptions == 0, 'kv_capacity_tokens': kv,
              'T': dropped / expected <= DROP_LIMIT, 'C': hits / queries < CACHE_HIT_LIMIT,
              'G': chunks / expected >= CHUNK_SHARE, 'dropped': dropped,
              'chunk_complete': chunks, 'total': expected,
              'prefix_queries': queries, 'prefix_hits': hits, 'preemptions': preemptions,
              'ttft_s': statistics.median(r['ttft_s'] for r in kept) if kept else None,
              'token_gap_s': statistics.median(r['token_gap_s'] for r in gap_rows) if gap_rows else None,
              'mean_tpot_s': statistics.median(tpot_values) if tpot_values else None,
              'output_tokens_per_second': sum(r['usage']['completion_tokens'] for r in kept) / ((end - start) / 1e9),
              'window_ns': [start, end]}
    result['rerun_level'] = not (result['T'] and result['C'])
    return result


# ------------------------------------------------------------------ gate N

def placement_shm(rows, node):
    nodes = {}
    for row in rows.values():
        for key, pages in (row.get('nccl_shm_pages') or {}).items():
            nodes[key] = nodes.get(key, 0) + pages
    return {'pages_by_node': nodes, 'total_pages': sum(nodes.values()),
            'off_node_pages': sum(v for k, v in nodes.items() if k != str(node))}


def placement(rows, server_cpus, node):
    """Gate N: every server process has at least 95% of its anonymous memory on the
    servers' node and a CPU affinity inside the server CPUs, and NCCL's /dev/shm staging
    pages exist and all lie on that node."""
    if not rows:
        raise EvidenceError('Missing process placement evidence')
    allowed = set(server_cpus)
    if not allowed:
        raise EvidenceError('Missing server CPU set')
    per_process = all(not r.get('error') and r.get('node_fraction_lower_bound') is not None
                      and r['node_fraction_lower_bound'] >= NODE_SHARE
                      and bool(r.get('affinity')) and set(r['affinity']) <= allowed for r in rows.values())
    shm = placement_shm(rows, node)
    return per_process and shm['total_pages'] > 0 and shm['off_node_pages'] == 0


# ------------------------------------------------------------------ gate Q

def gap_record(window):
    """Sampling intervals over 2.5 s inside the level's window (edge intervals included).
    A record, not a gate condition."""
    steps = [(a['monotonic_ns'], b['monotonic_ns']) for a, b in zip(window, window[1:])]
    over = [{'from_monotonic_ns': a, 'to_monotonic_ns': b, 'interval_s': (b - a) / 1e9}
            for a, b in steps if (b - a) / 1e9 > TELEMETRY_GAP_RECORD_S]
    return {'gate': False, 'threshold_s': TELEMETRY_GAP_RECORD_S,
            'max_interval_s': max(((b - a) / 1e9 for a, b in steps), default=None),
            'intervals_over_threshold': len(over), 'over_threshold': over}


def level_scale(window, begin, end):
    """When an edge interval of the sampled window exceeds 2.5 s, time outside the level
    would dilute "at most 2% over the level"; the foreign fraction is then also computed
    with the denominator scaled to the level's own duration (every foreign tick kept) and
    the larger of the two is gated. Returns the scale (<= 1) or None."""
    first = (window[1]['monotonic_ns'] - window[0]['monotonic_ns']) / 1e9
    last = (window[-1]['monotonic_ns'] - window[-2]['monotonic_ns']) / 1e9
    if max(first, last) <= TELEMETRY_GAP_RECORD_S:
        return None
    return (end - begin) / (window[-1]['monotonic_ns'] - window[0]['monotonic_ns'])


SAMPLE_ERROR = 'telemetry sample error'


def quiet(samples, cpus, begin, end, gpus, card_count):
    """Gate Q over one measured level. An errored telemetry sample fails the gate (so the
    round is rerun); the declared conditions are still computed on the other samples when
    possible. Without an errored sample, missing evidence raises EvidenceError."""
    errors = [str(s['error']) for s in samples if s.get('error')]
    if not errors:
        return _quiet_samples(samples, cpus, begin, end, gpus, card_count)
    clean = [s for s in samples if not s.get('error')]
    try:
        result = _quiet_samples(clean, cpus, begin, end, gpus, card_count)
    except EvidenceError as exc:
        result = {'not_computable': str(exc)}
    result.update({'pass': False, 'reasons': [SAMPLE_ERROR] + result.get('reasons', []),
                   'telemetry_sample_errors': errors})
    return result


def _quiet_samples(samples, cpus, begin, end, gpus, card_count):
    if len(samples) < 2:
        raise EvidenceError('Missing telemetry samples')
    if samples[0]['monotonic_ns'] > begin or samples[-1]['monotonic_ns'] < end:
        raise EvidenceError('Telemetry does not enclose the measured window')
    left = max(i for i, s in enumerate(samples) if s['monotonic_ns'] <= begin)
    right = min(i for i, s in enumerate(samples) if s['monotonic_ns'] >= end)
    samples = samples[left:right + 1]
    gaps = gap_record(samples)
    relevant = [s for s in samples if begin <= s['monotonic_ns'] <= end]
    if not relevant:
        raise EvidenceError('No samples within the measured window')
    foreign_gpu = []
    for s in samples:
        if sorted(g['index'] for g in s['gpus']) != list(range(card_count)):
            raise EvidenceError('Telemetry is missing a card')
        foreign_gpu.extend(p for p in s['compute_apps'] if str(p['pid']) not in s['owned'])
    a, b = samples[0], samples[-1]
    busy = sum(b['cpu'][str(c)]['busy'] - a['cpu'][str(c)]['busy'] for c in cpus)
    total = sum(b['cpu'][str(c)]['total'] - a['cpu'][str(c)]['total'] for c in cpus)
    if total <= 0 or busy < 0:
        raise EvidenceError('Bad CPU counters')
    owned = 0
    for pid, last in b['owned'].items():
        first = a['owned'].get(pid)
        if first and first['start'] == last['start']:
            owned += last['ticks'] - first['ticks']
        # A child born inside the interval is not subtracted: conservative.
    unscaled = max(0, busy - owned) / total
    scale = level_scale(samples, begin, end)
    scaled = max(0, busy - owned) / (total * scale) if scale else None
    foreign_fraction = max(unscaled, scaled) if scaled is not None else unscaled
    clocks = {}
    for gpu in gpus:
        values = [next(g['clock_mhz'] for g in s['gpus'] if g['index'] == gpu) for s in relevant]
        med = statistics.median(values)
        if med <= 0:
            raise EvidenceError('Invalid GPU clock')
        clocks[str(gpu)] = sum(v < CLOCK_FLOOR * med for v in values) / len(values)
    reasons = []
    if foreign_gpu:
        reasons.append('foreign compute process')
    if foreign_fraction > FOREIGN_CPU_LIMIT:
        reasons.append('foreign CPU time')
    if any(v > CLOCK_SHARE for v in clocks.values()):
        reasons.append('low GPU clock')
    return {'pass': not reasons, 'reasons': reasons,
            'foreign_gpu_processes': foreign_gpu, 'foreign_cpu_fraction': foreign_fraction,
            'foreign_cpu_fraction_unscaled': unscaled, 'foreign_cpu_fraction_level_scaled': scaled,
            'level_scale': scale, 'low_clock_fraction': clocks, 'telemetry_gaps': gaps}


# ------------------------------------------------------------------ decision rule

def metric_verdict(ratios):
    if len(ratios) != ROUNDS or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise EvidenceError('Five positive finite paired ratios required')
    if all(r < TP_BETTER_BELOW for r in ratios):
        return 'TP better'
    if all(r > PP_BETTER_ABOVE for r in ratios):
        return 'PP better'
    if all(TP_BETTER_BELOW <= r <= PP_BETTER_ABOVE for r in ratios):
        return 'no material difference'
    return 'unresolved'


# Written before measurement: PP4 has the lower TTFT, TP4 the lower token gap (TRADE-OFF).
EXPECTED_ROW = 'TRADE-OFF'


def reversed_verdicts(verdicts):
    return sorted(k for k, v in verdicts.items()
                  if (k.startswith('ttft_s:') and v == 'TP better')
                  or (k.startswith('token_gap_s:') and v == 'PP better'))


def level_valid(pair):
    return all(pair[a]['K'] and pair[a]['T'] and pair[a]['C'] for a in ARMS)


def decision(rounds, terminal_gate=None):
    """The decision row from the accepted rounds. A level gives verdicts only if it is
    valid (K, T, C in both arms) in all five rounds; its token-gap verdict also needs G."""
    if terminal_gate is not None:
        return {'row': 'NOT-MEASURABLE', 'terminal_gate': terminal_gate, 'verdicts': {}, 'ratios': {},
                'valid_levels': 0, 'expectation': {'row': EXPECTED_ROW, 'met': False, 'reversed_verdicts': []}}
    if len(rounds) != ROUNDS:
        raise EvidenceError('Need exactly five accepted rounds')
    verdicts, ratios, valid_levels = {}, {}, 0
    for c in LEVELS:
        pairs = [r['levels'][str(c)] for r in rounds]
        if not all(level_valid(pair) for pair in pairs):
            continue
        valid_levels += 1
        for metric in ('ttft_s', 'token_gap_s'):
            if metric == 'token_gap_s' and not all(all(pair[a]['G'] for a in ARMS) for pair in pairs):
                continue
            values = []
            for pair in pairs:
                tp, pp = pair['TP4'][metric], pair['PP4'][metric]
                if tp is None or pp is None or tp <= 0 or pp <= 0:
                    raise EvidenceError('Missing valid per-round median')
                values.append(tp / pp)
            key = f'{metric}:c{c}'
            ratios[key] = values
            verdicts[key] = metric_verdict(values)
    outcomes = set(verdicts.values())
    if valid_levels < 2:
        row = 'NOT-MEASURABLE'
    elif {'TP better', 'PP better'} <= outcomes:
        row = 'TRADE-OFF'
    elif 'unresolved' in outcomes:
        row = 'UNSTABLE'
    elif outcomes == {'no material difference'}:
        row = 'NO-STORY'
    else:
        row = 'ONE-SIDED'
    flipped = reversed_verdicts(verdicts)
    return {'row': row, 'terminal_gate': None, 'verdicts': verdicts, 'ratios': ratios,
            'valid_levels': valid_levels,
            'expectation': {'row': EXPECTED_ROW, 'met': row == EXPECTED_ROW and not flipped,
                            'reversed_verdicts': flipped}}


# ------------------------------------------------------------------ rerun budgets

class StudyRetries:
    """One gate-N replacement round for the whole study."""
    def __init__(self):
        self.replacement_used = False


class RoundRetries:
    """At most two reruns of a round for gate Q and two for gate N; after the second N
    rerun the round may be replaced once per study; any further failure ends the study."""
    def __init__(self, study=None):
        self.study = study if study is not None else StudyRetries()
        self.n_retries = self.q_retries = 0
        self.replaced = False
        self.terminal_gate = None

    def failed(self, placement_ok, quiet_ok):
        if not quiet_ok:
            if self.q_retries == QUIET_RERUNS:
                self.terminal_gate = 'Q'
                return 'NOT-MEASURABLE'
            self.q_retries += 1
        if not placement_ok:
            if self.replaced:
                self.terminal_gate = 'N'
                return 'NOT-MEASURABLE'
            if self.n_retries == PLACEMENT_RERUNS:
                if self.study.replacement_used:
                    self.terminal_gate = 'N'
                    return 'NOT-MEASURABLE'
                self.study.replacement_used = True
                self.replaced = True
                return 'replace_round'
            self.n_retries += 1
        return 'rerun_round'

    def receipt(self):
        return {'n_retries': self.n_retries, 'q_retries': self.q_retries,
                'replaced': self.replaced, 'terminal_gate': self.terminal_gate,
                'study_replacement_used': self.study.replacement_used}


# ------------------------------------------------------------------ assembling a round

def retry_levels(arms):
    """Levels rerun on bank 1: valid for K in both arms, and T or C failed in either."""
    return [c for c in LEVELS if all(arms[a][str(c)]['K'] for a in ARMS)
            and any(arms[a][str(c)]['rerun_level'] for a in ARMS)]


def merge_retry(original, replacement):
    """A rerun level replaces the first one, except that a K failure of the first stays,
    and a G failure that was reached (T and C passed) stays."""
    replacement = dict(replacement)
    replacement['K'] = original['K'] and replacement['K']
    if original['T'] and original['C'] and not original['G']:
        replacement['G'] = False
    replacement['replaces'] = original.get('directory')
    return replacement


def paired_round(number, attempt, first, retry=None):
    """first, retry: {arm: {str(c): cell}}. Returns the paired round and every cell seen
    (gates N and Q are judged over all of them, including the rerun levels)."""
    arms = {a: dict(first[a]) for a in ARMS}
    history = [cell for a in ARMS for cell in first[a].values()]
    for a in ARMS:
        for c, cell in (retry or {}).get(a, {}).items():
            history.append(cell)
            arms[a][c] = merge_retry(first[a][c], cell)
    paired = {'round': number, 'attempt': attempt,
              'levels': {str(c): {a: arms[a][str(c)] for a in ARMS} for c in LEVELS}}
    paired.update(N=all(cell['N'] for cell in history), Q=all(cell['Q'] for cell in history))
    return paired, history
