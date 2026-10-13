"""Gate calculations of one measured level, read back from the files the run wrote.

The runner calls `evaluate_level` right after a level, and `report` calls the same function
on the same files later, so the stored result and the recomputed one come from one code path.
"""
import json
from pathlib import Path

from . import client, gates, telemetry
from .spec import read_json


def placement_rows(directory, node):
    """placement.json, with the memory figures recomputed from the saved numa_maps files."""
    directory = Path(directory)
    rows = read_json(directory / 'placement.json')
    result = {}
    for pid, row in rows.items():
        row = dict(row)
        raw_path = directory / 'numa' / f'{pid}.numa_maps'
        if not row.get('error'):
            if not raw_path.exists():
                raise gates.EvidenceError(f'Missing saved numa_maps for pid {pid}')
            raw = raw_path.read_text()
            row.update(telemetry.anonymous_placement(raw, node), nccl_shm_pages=telemetry.nccl_shm_pages(raw))
        result[pid] = row
    return result


def evaluate_level(directory, concurrency, startup_metrics, topology):
    directory = Path(directory)
    rows = [client.summarize(r) for r in client.read_rows(directory / 'measured')]
    before = (directory / 'metrics-before.txt').read_text()
    after = (directory / 'metrics-after.txt').read_text()
    result = gates.level(rows, before, after, startup_metrics, concurrency)
    node = topology['server_node']
    memory = placement_rows(directory, node)
    result['N'] = gates.placement(memory, topology['server_cpus'], node)
    result['nccl_shm_placement'] = gates.placement_shm(memory, node)
    result['node_fraction_lower_bound_min'] = min(
        (r['node_fraction_lower_bound'] for r in memory.values()
         if r.get('node_fraction_lower_bound') is not None), default=None)
    samples = json.loads((directory / 'telemetry-with-boundaries.json').read_text())
    result['quiet'] = gates.quiet(samples, topology['server_cpus'], *result['window_ns'],
                                  topology['gpus'], topology['card_count'])
    result['Q'] = result['quiet']['pass']
    result['directory'] = str(directory)
    return result
