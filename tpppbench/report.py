"""Recompute every gate, ratio and the decision row from a study directory's raw files.

Exit code 0: recomputed values agree with what the run stored. 1: a disagreement (each is
printed as WARNING). 2: the timed phase is missing or incomplete, so there is no decision.
"""
import json
import math
from pathlib import Path

from . import analysis, gates
from .run import transport_gate
from .spec import (ARMS, LEVELS, REFERENCE_PROMPTS_SHA256, ROUNDS, TOUCH_BYTES, USABLE_FREE_AFTER_HELPER,
                   read_json, write_json)


class Report:
    def __init__(self):
        self.lines, self.warnings, self.data = [], [], {}

    def out(self, text=''):
        self.lines.append(text)

    def warn(self, text):
        self.warnings.append(text)
        self.lines.append('WARNING: ' + text)


def fmt(value, digits=3):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def inputs(study, rep):
    manifests = [p / 'run-manifest.json' for p in (study / name for name in
                 ('start-test', 'settings', 'transport', 'timed')) if (p / 'run-manifest.json').exists()]
    if not manifests:
        rep.out('No phase has run in this study directory.')
        return None
    first = read_json(manifests[-1])
    rep.data['inputs'] = {k: first.get(k) for k in ('gpus', 'versions', 'prompts_sha256', 'reference_prompts',
                                                    'deviations', 'touch_bytes', 'min_usable_free_bytes')}
    v = first.get('versions') or {}
    rep.out('## Inputs')
    rep.out(f"vLLM {v.get('vllm')} (reference build: {v.get('reference_versions')}), torch {v.get('torch')}, "
            f"transformers {v.get('transformers')}")
    rep.out(f"GPUs {first.get('gpus')}; server CPUs on socket {first['topology'].get('server_socket')} "
            f"(NUMA node {first['topology'].get('server_node')}), client CPUs {first['topology'].get('client_cpus')}")
    rep.out(f"prompts sha256 {first.get('prompts_sha256')} "
            f"({'the reference prompts' if first.get('prompts_sha256') == REFERENCE_PROMPTS_SHA256 else 'NOT the reference prompts'})")
    deviations = first.get('deviations') or {}
    rep.out('declared deviations applied to both arms: ' + (', '.join(f'{k}={v}' for k, v in deviations.items())
                                                             if deviations else 'none'))
    touch, free = first.get('touch_bytes'), first.get('min_usable_free_bytes')
    if touch == 0:
        rep.out('touch-and-free before server starts: OFF (the reference procedure runs it)')
    elif touch is not None and touch != TOUCH_BYTES:
        rep.out(f'touch-and-free writes {touch / 1e9:g} GB (reference {TOUCH_BYTES / 1e9:g} GB; refusal threshold '
                f'moves with it)')
    if free is not None and free != USABLE_FREE_AFTER_HELPER and touch != 0:
        rep.out(f'usable free memory required after the helper: {free / 1e9:g} GB '
                f'(reference {USABLE_FREE_AFTER_HELPER / 1e9:g} GB)')
    cards = first['topology'].get('card_count')
    rep.out(f'cards watched by gate Q: {cards}')
    identity = first.get('model_identity') or {}
    if identity and not identity.get('reference_model_files'):
        rep.out(f"model files differ from the reference revision: {identity.get('differing_model_files')}")
    for path in manifests:
        other = read_json(path)
        for key in ('prompts_sha256', 'deviations', 'gpus', 'versions', 'cache_dir', 'touch_bytes',
                    'min_usable_free_bytes'):
            if other.get(key) != first.get(key):
                rep.warn(f'{path.parent.name} ran with a different {key} than {manifests[-1].parent.name}')
    rep.out()
    return first


def phase_status(study, name):
    base = study / name
    if (base / 'result.json').exists():
        return read_json(base / 'result.json')
    if (base / 'INCOMPLETE.json').exists():
        return {'pass': False, 'incomplete': read_json(base / 'INCOMPLETE.json')['error']}
    return None


def start_test(study, rep):
    status = phase_status(study, 'start-test')
    rep.out('## Start test (gate S)')
    if status is None:
        rep.out('not run')
        rep.out()
        return
    rep.out(f"pass: {status.get('pass')}" + (f" ({status['incomplete']})" if status.get('incomplete') else ''))
    for arm in ARMS:
        record_path = study / 'start-test' / arm / 'start-record.json'
        if record_path.exists():
            record = read_json(record_path)
            rehearsal = study / 'start-test' / arm / 'rehearsal.json'
            rep.out(f"  {arm}: KV capacity {record['kv_capacity_tokens']} tokens, compile {record['compile_mode']}, "
                    f"c=32 rehearsal {'completed' if rehearsal.exists() else 'missing'}")
    rep.out()


SETTING_KEYS = ('model_runner', 'async_scheduling', 'max_num_seqs', 'max_num_batched_tokens',
                'max_concurrent_batches', 'num_gpu_blocks', 'block_size', 'gpu_memory_utilization',
                'enable_prefix_caching', 'max_seq_len', 'attention_backend')


def settings(study, rep):
    rep.out('## Resolved settings (DEBUG start, untimed)')
    status = phase_status(study, 'settings')
    if status is None:
        rep.out('not run')
        rep.out()
        return
    fields = {}
    for arm in ARMS:
        path = study / 'settings' / arm / 'startup-fields.json'
        if path.exists():
            fields[arm] = read_json(path)
    rep.data['settings'] = {a: {k: f.get(k) for k in SETTING_KEYS} for a, f in fields.items()}
    rep.out(f"{'setting':<26}" + ''.join(f'{a:>12}' for a in fields))
    for key in SETTING_KEYS:
        rep.out(f'{key:<26}' + ''.join(f'{fmt(f.get(key)):>12}' for f in fields.values()))
    if 'TP4' in fields:
        rep.out(f"TP4 all-reduce: faster paths disabled and NCCL in use: {fields['TP4']['allreduce_path']['pass']}")
    runners = {f.get('model_runner') for f in fields.values()}
    if len(runners) > 1:
        rep.out('The two arms run different model-runner versions; that is part of what the mode changes.')
    rep.out()


def transport(study, rep):
    rep.out('## Transport check (gate P)')
    status = phase_status(study, 'transport')
    if status is None:
        rep.out('not run')
        rep.out()
        return None
    result = {}
    for arm in ARMS:
        directory = study / 'transport' / arm
        if not (directory / 'ready-cards.json').exists():
            rep.out(f'  {arm}: no ready server')
            continue
        try:
            gate = transport_gate(directory, arm)
        except gates.EvidenceError as exc:
            rep.out(f'  {arm}: evidence error: {exc}')
            result[arm] = False
            continue
        result[arm] = gate['pass']
        failed = [k for k, v in gate['checks'].items() if not v]
        transports = sorted({c['transport'] for c in gate['channels']})
        rep.out(f"  {arm}: {'PASS' if gate['pass'] else 'FAIL ' + ', '.join(failed)}; channel transports "
                f"{transports}; multi-rank communicators {sum(c['nranks'] > 1 for c in gate['communicators'])}")
        stored = directory / 'transport.json'
        if stored.exists() and read_json(stored).get('pass') != gate['pass']:
            rep.warn(f'{arm} gate P recomputed as {gate["pass"]}, stored {read_json(stored).get("pass")}')
    rep.data['gate_P'] = result
    if result and not all(result.values()):
        rep.out('Gate P failed: the premise (no peer-to-peer, NCCL stages through host memory) does not hold; '
                'no timing claim.')
    rep.out()
    return result


CELL_KEYS = ('K', 'T', 'C', 'G', 'N', 'Q', 'dropped', 'chunk_complete', 'prefix_queries', 'prefix_hits',
             'preemptions', 'kv_capacity_tokens', 'ttft_s', 'token_gap_s', 'mean_tpot_s',
             'output_tokens_per_second', 'rerun_level')
MISMATCHES = []


def compare_cell(directory, fresh):
    """Recomputed level values against the gate-calculation.json the run stored."""
    path = Path(directory) / 'gate-calculation.json'
    if not path.exists():
        MISMATCHES.append(f'{directory}: no stored gate-calculation.json')
        return
    stored = read_json(path)
    for key in CELL_KEYS:
        a, b = stored.get(key), fresh.get(key)
        same = close(a, b) if isinstance(a, float) or isinstance(b, float) else a == b
        if not same:
            MISMATCHES.append(f'{directory}: {key} recomputed {b}, stored {a}')
    if (stored.get('quiet') or {}).get('foreign_cpu_fraction') is not None and not close(
            stored['quiet']['foreign_cpu_fraction'], fresh['quiet'].get('foreign_cpu_fraction')):
        MISMATCHES.append(f'{directory}: foreign_cpu_fraction differs')


def level_cells(attempt_dir, arm_dir_name, levels, topology):
    base = attempt_dir / arm_dir_name
    startup = (base / 'metrics-startup.txt').read_text()
    cells = {}
    for c in levels:
        cells[str(c)] = analysis.evaluate_level(base / f'c{c}', c, startup, topology)
        compare_cell(base / f'c{c}', cells[str(c)])
    return cells


def replay_timed(study, topology):
    """Rebuild the accepted rounds from the attempt directories, applying the same rerun
    budgets as the runner. Returns (rounds, terminal gate, attempts, problem)."""
    timed = study / 'timed'
    rounds, attempts = [], []
    study_retries = gates.StudyRetries()
    for number in range(1, ROUNDS + 1):
        retry_state = gates.RoundRetries(study_retries)
        attempt = 0
        while True:
            directory = timed / f'round-{number}-attempt-{attempt}'
            if not directory.exists():
                return rounds, None, attempts, f'round {number} attempt {attempt} missing'
            try:
                first = {a: level_cells(directory, a, LEVELS, topology) for a in ARMS}
                levels = gates.retry_levels(first)
                retry = ({a: level_cells(directory, a + '-level-retry', levels, topology) for a in ARMS}
                         if levels else None)
            except FileNotFoundError as exc:
                return rounds, None, attempts, f'round {number} attempt {attempt} incomplete ({exc.filename})'
            paired, history = gates.paired_round(number, attempt, first, retry)
            stored_pair = directory / 'paired.json'
            if stored_pair.exists():
                old = read_json(stored_pair)
                if (old.get('N'), old.get('Q')) != (paired['N'], paired['Q']):
                    MISMATCHES.append(f'{directory}: N/Q recomputed {paired["N"]}/{paired["Q"]}, stored '
                                      f'{old.get("N")}/{old.get("Q")}')
            else:
                MISMATCHES.append(f'{directory}: no stored paired.json')
            record = {'round': number, 'attempt': attempt, 'N': paired['N'], 'Q': paired['Q'],
                      'retry_levels': levels, 'directory': str(directory)}
            attempts.append(record)
            if paired['N'] and paired['Q']:
                rounds.append(paired)
                break
            record['action'] = retry_state.failed(paired['N'], paired['Q'])
            record['quiet_reasons'] = sorted({r for cell in history for r in cell['quiet'].get('reasons', [])})
            if record['action'] == 'NOT-MEASURABLE':
                return rounds, retry_state.terminal_gate, attempts, None
            attempt += 1
    extra = timed / f'round-{ROUNDS + 1}-attempt-0'
    if extra.exists():
        return rounds, None, attempts, 'more rounds on disk than the procedure runs'
    return rounds, None, attempts, None


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def compare_stored(stored, decision, rep):
    if stored.get('row') != decision['row']:
        rep.warn(f"decision row recomputed as {decision['row']}, stored {stored.get('row')}")
    if stored.get('verdicts') != decision['verdicts']:
        rep.warn(f"verdicts recomputed as {decision['verdicts']}, stored {stored.get('verdicts')}")
    for key, values in decision['ratios'].items():
        old = (stored.get('ratios') or {}).get(key)
        if old is None or len(old) != len(values) or not all(close(a, b) for a, b in zip(old, values)):
            rep.warn(f'ratios {key} recomputed as {values}, stored {old}')


def timed_section(study, rep):
    rep.out('## Timed rounds')
    timed = study / 'timed'
    if not (timed / 'run-manifest.json').exists():
        rep.out('not run')
        rep.out()
        return None
    topology = read_json(timed / 'run-manifest.json')['topology']
    stored = read_json(timed / 'decision.json') if (timed / 'decision.json').exists() else None
    MISMATCHES.clear()
    rounds, terminal, attempts, problem = replay_timed(study, topology)
    rep.data['attempts'] = attempts
    for text in MISMATCHES:
        rep.warn(text)
    for a in attempts:
        if not (a['N'] and a['Q']):
            rep.out(f"round {a['round']} attempt {a['attempt']}: N {a['N']}, Q {a['Q']} -> {a.get('action')} "
                    f"{a.get('quiet_reasons') or ''}")
    if problem is not None:
        if stored and stored.get('terminal_gate') == 'S':
            decision = stored
            rep.out('A server did not start during the timed phase (gate S): NOT-MEASURABLE.')
        else:
            rep.out(f'Timed phase incomplete: {problem}. No decision.')
            rep.out()
            return None
    else:
        decision = gates.decision(rounds, terminal)
        if stored is None:
            rep.warn('the run stored no decision.json')
        else:
            compare_stored(stored, decision, rep)
    rep.data['decision'] = decision
    if rounds:
        rounds_table(rounds, rep)
    rep.out()
    rep.out('## Decision')
    if decision.get('terminal_gate'):
        rep.out(f"row NOT-MEASURABLE: gate {decision['terminal_gate']} ended the study")
    else:
        rep.out('verdict per metric and level (all five TP4/PP4 ratios below 0.90: TP better; above 1.10: PP better; '
                'within 0.90-1.10: no material difference; otherwise unresolved):')
        for c in LEVELS:
            for metric, label in (('ttft_s', 'TTFT'), ('token_gap_s', 'token gap')):
                key = f'{metric}:c{c}'
                if key in decision['verdicts']:
                    rep.out(f"  {label:<10} c={c:<3} {decision['verdicts'][key]:<24} ratios "
                            + ' '.join(f'{r:.3f}' for r in decision['ratios'][key]))
                else:
                    rep.out(f'  {label:<10} c={c:<3} no verdict (level or gate G not valid in all five rounds)')
        rep.out(f"valid levels: {decision['valid_levels']} of {len(LEVELS)}")
        rep.out(f"row: {decision['row']}")
        expectation = decision['expectation']
        rep.out(f"expectation written before measurement (TRADE-OFF: PP4 lower TTFT, TP4 lower token gap): "
                f"{'met' if expectation['met'] else 'NOT met'}"
                + (f"; verdicts against it: {expectation['reversed_verdicts']}" if expectation['reversed_verdicts'] else ''))
    rep.out()
    return decision


def rounds_table(rounds, rep):
    rep.out()
    rep.out('per accepted round: median TTFT and median token gap per arm (ms), ratio TP4/PP4; '
            'gates K T C G per arm')
    rep.out(f"{'round':<6}{'c':<4}{'TTFT TP4':>10}{'TTFT PP4':>10}{'ratio':>8}{'gap TP4':>9}{'gap PP4':>9}"
            f"{'ratio':>8}{'tok/s TP4':>11}{'tok/s PP4':>11}  gates TP4 | PP4")
    worst = {'max_telemetry_interval_s': 0.0, 'max_foreign_cpu_fraction': 0.0, 'min_node_fraction': 1.0}
    for r in rounds:
        for c in LEVELS:
            pair = r['levels'][str(c)]
            tp, pp = pair['TP4'], pair['PP4']

            def ms(x):
                return None if x is None else x * 1e3

            def ratio(m):
                return tp[m] / pp[m] if tp[m] and pp[m] else None

            def g(cell):
                return ''.join(k if cell[k] else k.lower() for k in ('K', 'T', 'C', 'G'))
            rep.out(f"{r['round']:<6}{c:<4}{fmt(ms(tp['ttft_s']), 1):>10}{fmt(ms(pp['ttft_s']), 1):>10}"
                    f"{fmt(ratio('ttft_s')):>8}{fmt(ms(tp['token_gap_s']), 2):>9}{fmt(ms(pp['token_gap_s']), 2):>9}"
                    f"{fmt(ratio('token_gap_s')):>8}{fmt(tp['output_tokens_per_second'], 0):>11}"
                    f"{fmt(pp['output_tokens_per_second'], 0):>11}  {g(tp)} | {g(pp)}")
            for cell in (tp, pp):
                gaps = cell['quiet'].get('telemetry_gaps') or {}
                worst['max_telemetry_interval_s'] = max(worst['max_telemetry_interval_s'],
                                                        gaps.get('max_interval_s') or 0)
                worst['max_foreign_cpu_fraction'] = max(worst['max_foreign_cpu_fraction'],
                                                        cell['quiet'].get('foreign_cpu_fraction') or 0)
                if cell.get('node_fraction_lower_bound_min') is not None:
                    worst['min_node_fraction'] = min(worst['min_node_fraction'], cell['node_fraction_lower_bound_min'])
    rep.data['accepted_level_extremes'] = worst
    keys = ('ttft_s', 'token_gap_s', 'mean_tpot_s', 'output_tokens_per_second', 'K', 'T', 'C', 'G', 'N', 'Q',
            'dropped', 'chunk_complete', 'prefix_hits', 'prefix_queries', 'preemptions', 'kv_capacity_tokens',
            'node_fraction_lower_bound_min', 'directory')
    rep.data['rounds'] = [{'round': r['round'], 'attempt': r['attempt'], 'levels': {
        c: {a: {**{k: cell.get(k) for k in keys},
                'foreign_cpu_fraction': cell['quiet'].get('foreign_cpu_fraction'),
                'max_telemetry_interval_s': (cell['quiet'].get('telemetry_gaps') or {}).get('max_interval_s')}
            for a, cell in pair.items()} for c, pair in r['levels'].items()}} for r in rounds]
    rep.out('(gate letter in lower case = failed)')
    rep.out(f"over accepted levels: largest telemetry sampling interval {worst['max_telemetry_interval_s']:.2f} s "
            f"(recorded, not a gate); largest foreign CPU share on the servers' socket "
            f"{worst['max_foreign_cpu_fraction'] * 100:.2f}% (gate Q: at most 2%); smallest share of a server "
            f"process's anonymous memory on the servers' node {worst['min_node_fraction'] * 100:.1f}% "
            '(gate N: at least 95%)')


def build(study):
    study = Path(study)
    rep = Report()
    rep.out(f'# TP4 vs PP4, Qwen2.5-32B-Instruct, 1536 -> 128 tokens: {study}')
    rep.out()
    first = inputs(study, rep)
    if first is None:
        return rep, 2
    start_test(study, rep)
    settings(study, rep)
    transport(study, rep)
    decision = timed_section(study, rep)
    code = 2 if decision is None else (1 if rep.warnings else 0)
    return rep, code


def main(args):
    rep, code = build(args.study)
    print('\n'.join(rep.lines))
    if args.json:
        write_json(args.json, {**rep.data, 'warnings': rep.warnings, 'exit_code': code})
    return code
