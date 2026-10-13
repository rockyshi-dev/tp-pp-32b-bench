"""The phases: start-test, settings, transport and timed.

Each phase writes `<study>/<phase>/` (a new directory) and ends with `result.json`.
"""
import json
import os
from pathlib import Path
import time

from . import analysis, client, gates, hostmem, prompts, server, telemetry
from .spec import (ARMS, LEVELS, MODEL, MODEL_REVISION, REFERENCE_MODEL_FILES, REFERENCE_PROMPTS_SHA256, ROUNDS,
                   digest, read_json, write_json)

PHASES = ('start-test', 'settings', 'transport', 'timed')
PREREQUISITES = {'start-test': (), 'settings': ('start-test',), 'transport': ('start-test',),
                 'timed': ('start-test', 'settings', 'transport')}
REQUESTS_PER_START = {'start-test': 1, 'settings': 1, 'transport': 8}
REHEARSAL_LEVEL = 32
MARGIN_BYTES = hostmem.MARGIN_BYTES


class HostMemoryStop(RuntimeError):
    """The helper refused, or the node's usable free memory stayed below the threshold
    after the helper and one repeat. The run stops before the server starts."""


# ------------------------------------------------------------------ touch-and-free

def host_memory_dir(output):
    return output.parent / f'{output.name}-host-memory'


def host_memory_step(cfg, output):
    """Before one server start: the helper on the servers' node, then the precondition
    usable free >= the threshold; one repeat; otherwise stop before the server starts."""
    directory = host_memory_dir(output)
    directory.mkdir(parents=True, exist_ok=False)
    node = cfg.topology['server_node']
    record = {'node': node, 'touch_bytes': cfg.touch_bytes,
              'precondition_usable_free_bytes': cfg.min_usable_free, 'runs': [], 'units': 'decimal bytes'}
    if cfg.touch_bytes == 0:
        record.update(skipped=True, reading=hostmem.snapshot(node))
        write_json(directory / 'host-memory.json', record)
        return record
    usable = None
    for number in (1, 2):
        try:
            rows = hostmem.run((node,), directory / f'helper-{number}', telemetry.HELPER_PIDS,
                               cpus_of={node: cfg.topology['server_cpus']}, nbytes=cfg.touch_bytes,
                               required=cfg.touch_bytes + MARGIN_BYTES)
        except hostmem.Refusal as exc:
            record.update(stopped=True, reason=f'helper refused: {exc}', refusal=exc.record)
            write_json(directory / 'host-memory.json', record)
            raise HostMemoryStop(record['reason']) from exc
        row = rows[node]
        usable = row['after']['usable_free_bytes']
        record['runs'].append({'helper': number, 'pid': row['pid'], 'seconds': row['seconds'],
                               'before': row['before'], 'after': row['after'],
                               'precondition_met': usable >= cfg.min_usable_free})
        if usable >= cfg.min_usable_free:
            record['precondition'] = 'met'
            write_json(directory / 'host-memory.json', record)
            return record
    record.update(stopped=True, reason=(f'node {node} usable free memory {usable} bytes < {cfg.min_usable_free} '
                                        'bytes after the helper and one repeat; stopped before the server start'))
    write_json(directory / 'host-memory.json', record)
    raise HostMemoryStop(record['reason'])


def host_memory_ready(cfg, output):
    reading = hostmem.snapshot(cfg.topology['server_node'])
    write_json(host_memory_dir(output) / 'server-ready.json', reading)
    return reading


# ------------------------------------------------------------------ one start

def metrics(srv, directory, name):
    text = server.http(srv.url + '/metrics')
    (directory / name).write_text(text)
    gates.metric_samples(text, 'vllm:num_requests_waiting')
    return text


def measured_level(cfg, srv, bank, c, retry, directory):
    directory.mkdir()
    ids = bank['banks'][f'{c}:{retry}']
    rows = bank['requests']
    topo = cfg.topology
    warm = client.phase(srv.url, [rows[i] for i in ids['warmup']], c, topo['client_cpus'], directory / 'warmup')
    if not all(gates.request_valid(r) for r in warm):
        raise server.StartFailure('Warm-up incomplete or wrong token count')
    write_json(directory / 'placement.json',
               telemetry.placement(srv.process.pid, directory / 'numa', topo['server_node']))
    metrics(srv, directory, 'metrics-before.txt')
    monitor = telemetry.Monitor(srv.process.pid, directory, topo['runner_cpus'])
    boundary_before = monitor.boundary()
    with monitor:
        client.phase(srv.url, [rows[i] for i in ids['measured']], c, topo['client_cpus'], directory / 'measured')
    boundary_after = monitor.boundary()
    metrics(srv, directory, 'metrics-after.txt')
    samples = [boundary_before, *(json.loads(x) for x in monitor.path.read_text().splitlines()), boundary_after]
    write_json(directory / 'telemetry-with-boundaries.json', samples)
    startup = (srv.directory / 'metrics-startup.txt').read_text()
    result = analysis.evaluate_level(directory, c, startup, topo)
    result.update(bank_retry=retry, prompt_ids=ids)
    write_json(directory / 'gate-calculation.json', result)
    return result


def arm(cfg, phase, name, output, locks, deviations, bank, levels, retry=0, settings_capacity=None):
    srv = None
    try:
        server.preflight(cfg.port)
        host_memory_step(cfg, output)
        srv = server.Server(cfg, name, output, locks, deviations, phase)
        srv.ready()
        host_memory_ready(cfg, output)
        initial = metrics(srv, output, 'metrics-startup.txt')
        log = (output / 'server.log').read_text(errors='replace')
        fields = server.startup_fields(log, initial, name)
        write_json(output / 'startup-fields.json', fields)
        record = server.start_record(log, initial)
        if phase == 'timed':
            record['settings_comparison'] = server.capacity_note(record, (settings_capacity or {}).get(name))
        write_json(output / 'start-record.json', record)
        problem = server.start_problem(record, phase)
        if problem is None and not fields['allreduce_path']['pass']:
            problem = 'TP4 did not log every faster all-reduce path as disabled and NCCL (pynccl) in use'
        if problem:
            write_json(output / 'start-flag.json', {'problem': problem, 'phase': phase})
            srv.close_completed()
            raise gates.EvidenceError(problem)
        if phase == 'settings' and fields['missing_fields']:
            raise gates.EvidenceError('DEBUG settings evidence missing: ' + ', '.join(fields['missing_fields']))
        results = {}
        if phase in REQUESTS_PER_START:
            count = REQUESTS_PER_START[phase]
            records = client.phase(srv.url, bank['requests'][:count], 8 if phase == 'transport' else 1,
                                   cfg.topology['client_cpus'], output / 'requests')
            answered = all(gates.request_valid(r) for r in records)
            write_json(output / 'request-check.json', {'all_answered': answered, 'count': count})
            if not answered:
                if phase == 'start-test':
                    raise server.StartFailure('Start-test request failed')
                raise gates.EvidenceError(f'{phase} request failed')
            if phase == 'start-test':
                rehearsal(cfg, srv, bank, output, initial)
        else:
            for c in levels:
                results[str(c)] = measured_level(cfg, srv, bank, c, retry, output / f'c{c}')
        srv.close_completed()
        if phase == 'transport':
            results['transport'] = transport_gate(output, name)
            write_json(output / 'transport.json', results['transport'])
        return results
    except BaseException as exc:
        if srv is not None and not srv.completed:
            srv.retain(str(exc))
        raise


def rehearsal(cfg, srv, bank, output, initial):
    """One untimed c = 32 level through the timed code path (several client processes,
    placement, monitor, counters). It must finish its gate calculation; it enters no result."""
    result = measured_level(cfg, srv, bank, REHEARSAL_LEVEL, 0, output / f'rehearsal-c{REHEARSAL_LEVEL}')
    problem = {k: result['quiet'][k] for k in ('not_computable', 'telemetry_sample_errors')
               if result['quiet'].get(k)}
    if (result['quiet'].get('telemetry_gaps') or {}).get('intervals_over_threshold'):
        problem['telemetry_gaps'] = result['quiet']['telemetry_gaps']
    if problem:
        raise gates.EvidenceError(f'Rehearsal telemetry evidence failed: {problem}')
    write_json(output / 'rehearsal.json',
               {'concurrency': REHEARSAL_LEVEL, 'completed': True, 'enters_no_result': True,
                'gates': {k: result[k] for k in ('K', 'T', 'C', 'G', 'N', 'Q')},
                'cache_config': gates.cache_config(initial)})


def transport_gate(directory, name):
    """Gate P from the per-process NCCL debug files; the workers are the server's compute
    processes at readiness."""
    cards = read_json(Path(directory) / 'ready-cards.json')
    workers = sorted({row['pid'] for row in cards['compute_apps']})
    return gates.transport(gates.nccl_files(directory), name, workers)


# ------------------------------------------------------------------ timed rounds

def timed(cfg, output, locks, deviations, bank, settings_capacity):
    rounds = []
    study_retries = gates.StudyRetries()
    for number in range(1, ROUNDS + 1):
        retry_state = gates.RoundRetries(study_retries)
        attempt = 0
        while True:
            directory = output / f'round-{number}-attempt-{attempt}'
            directory.mkdir()
            order = ARMS if number % 2 else tuple(reversed(ARMS))
            first = {name: arm(cfg, 'timed', name, directory / name, locks, deviations, bank, LEVELS,
                               settings_capacity=settings_capacity) for name in order}
            retry = None
            levels = gates.retry_levels(first)
            if levels:
                # Both arms rerun the level on the unused bank, in a fresh start, even if
                # only one arm failed.
                retry = {name: arm(cfg, 'timed', name, directory / (name + '-level-retry'), locks, deviations,
                                   bank, levels, retry=1, settings_capacity=settings_capacity)
                         for name in order}
            paired, history = gates.paired_round(number, attempt, first, retry)
            write_json(directory / 'paired.json', paired)
            if paired['N'] and paired['Q']:
                rounds.append(paired)
                write_json(output / 'accepted-rounds.json', rounds)
                break
            action = retry_state.failed(paired['N'], paired['Q'])
            q_failures = [{'directory': cell.get('directory'), 'reasons': cell['quiet'].get('reasons', []),
                           'not_computable': cell['quiet'].get('not_computable')}
                          for cell in history if not cell['Q']]
            write_json(directory / 'retry-action.json', {'action': action, **retry_state.receipt(),
                                                         'q_failures': q_failures})
            if action == 'NOT-MEASURABLE':
                return rounds, retry_state.terminal_gate
            attempt += 1
    return rounds, None


# ------------------------------------------------------------------ phase driver

def check_prerequisites(cfg, phase, deviations, prompts_sha):
    for prior in PREREQUISITES[phase]:
        path = Path(cfg.study) / prior / 'result.json'
        if not path.exists():
            raise RuntimeError(f'{phase} needs a passing {prior} phase in {cfg.study} ({path} missing)')
        data = read_json(path)
        if data.get('pass') is not True:
            raise RuntimeError(f'{prior} did not pass ({path})')
        if data.get('deviations') != deviations or data.get('prompts_sha256') != prompts_sha:
            raise RuntimeError(f'{prior} ran with other deviations or prompts')
        if data.get('gpus') != list(cfg.gpus) or data.get('model_path') != str(cfg.model):
            raise RuntimeError(f'{prior} ran on other GPUs or with another model path')
        if data.get('cache_dir') != str(cfg.cache_dir):
            raise RuntimeError(f'{prior} used another compile cache ({data.get("cache_dir")}); timed starts '
                               'must load the cache the earlier starts built')


def load_inputs(cfg):
    """Prompt file, model files and versions; returns (bank, prompts sha256, versions)."""
    bank = prompts.load(cfg.prompts)
    sha = digest(cfg.prompts)
    if sha != REFERENCE_PROMPTS_SHA256 and not cfg.allow_other_prompts:
        raise RuntimeError(f'Prompt file sha256 {sha} is not the reference prompt file; '
                           'use --allow-other-prompts to run anyway (recorded)')
    for name, expected in bank['tokenizer_files'].items():
        path = Path(cfg.model) / name
        if not path.exists() or digest(path) != expected:
            raise RuntimeError(f'Model file {name} differs from the one the prompts were built with')
    model_check(cfg.model, cfg.allow_other_model)
    return bank, sha


def model_check(model, allow_other):
    """The model directory's JSON files (config, generation config, weight index, tokenizer)
    must be those of the reference revision. Weight shards are not hashed."""
    found = {name: (digest(Path(model) / name) if (Path(model) / name).exists() else None)
             for name in REFERENCE_MODEL_FILES}
    differ = sorted(n for n, sha in found.items() if sha != REFERENCE_MODEL_FILES[n])
    if differ and not allow_other:
        raise RuntimeError(f'Model files {differ} differ from {MODEL} at {MODEL_REVISION}; '
                           'use --allow-other-model to run anyway (recorded)')
    return {'reference_model_files': not differ, 'differing_model_files': differ}


def settings_capacities(cfg):
    base = Path(cfg.study) / 'settings'
    return {name: gates.capacity((base / name / 'metrics-startup.txt').read_text()) for name in ARMS}


def manifest(cfg, phase, deviations, prompts_sha):
    return {'phase': phase, 'study': str(cfg.study), 'model': str(cfg.model), 'prompts': str(cfg.prompts),
            'prompts_sha256': prompts_sha, 'reference_prompts': prompts_sha == REFERENCE_PROMPTS_SHA256,
            'gpus': list(cfg.gpus), 'port': cfg.port, 'vllm': str(cfg.vllm), 'cache_dir': str(cfg.cache_dir),
            'touch_bytes': cfg.touch_bytes, 'min_usable_free_bytes': cfg.min_usable_free,
            'deviations': deviations, 'versions': cfg.versions, 'model_revision': MODEL_REVISION,
            'model_identity': model_check(cfg.model, True), 'locks': [str(p) for p in cfg.locks],
            'expected_cards': cfg.topology.get('card_count'),
            'topology': {k: v for k, v in cfg.topology.items() if k != 'lscpu'},
            'argv': cfg.argv, 'start_wall_ns': time.time_ns()}


def execute(cfg, phase):
    """Run one phase. Returns True if it passed."""
    deviations = server.deviations_from_evidence(cfg.deviation_evidence)
    bank, prompts_sha = load_inputs(cfg)
    check_prerequisites(cfg, phase, deviations, prompts_sha)
    os.sched_setaffinity(0, cfg.topology['runner_cpus'])
    settings_capacity = settings_capacities(cfg) if phase == 'timed' else None
    output = Path(cfg.study) / phase
    locks = server.Locks(output, cfg.lock_dir, cfg.gpus, cfg.locks)
    result = {'phase': phase, 'pass': False, 'deviations': deviations, 'prompts_sha256': prompts_sha,
              'gpus': list(cfg.gpus), 'model_path': str(cfg.model), 'cache_dir': str(cfg.cache_dir)}
    try:
        write_json(output / 'run-manifest.json', manifest(cfg, phase, deviations, prompts_sha))
        if phase == 'timed':
            rounds, terminal = timed(cfg, output, locks, deviations, bank, settings_capacity)
            calculation = gates.decision(rounds, terminal)
            write_json(output / 'decision.json', calculation)
            result['pass'] = terminal is None
            result['row'] = calculation['row']
        else:
            results = {name: arm(cfg, phase, name, output / name, locks, deviations, bank, [])
                       for name in ARMS}
            result['pass'] = phase != 'transport' or all(results[a]['transport']['pass'] for a in ARMS)
            if phase == 'transport':
                result['gate_P'] = {a: results[a]['transport']['pass'] for a in ARMS}
                if not result['pass']:
                    write_json(output / 'decision.json', gates.decision([], 'P'))
        result['finish_wall_ns'] = time.time_ns()
        write_json(output / 'result.json', result)
        return result['pass']
    except BaseException as exc:
        if isinstance(exc, HostMemoryStop):
            write_json(output / 'STOPPED-host-memory.json',
                       {'reason': str(exc), 'action': 'Stopped before a server start; no server of this '
                        'start exists.'})
        elif isinstance(exc, server.StartFailure):
            write_json(output / 'decision.json', gates.decision([], 'S'))
        write_json(output / 'INCOMPLETE.json', {'error': f'{type(exc).__name__}: {exc}',
                                                'action': 'No result is usable. Check MANUAL_ATTENTION.json '
                                                          'files for a server left running.'})
        raise
    finally:
        locks.close()
