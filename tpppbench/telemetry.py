"""Read-only evidence from /proc, sysfs, lscpu and nvidia-smi; never imports a GPU library."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from .spec import CLIENT_CORES

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=20)


def parse_cpulist(text):
    cpus = []
    for part in text.strip().split(','):
        if not part:
            continue
        lo, _, hi = part.partition('-')
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(cpus)


def parse_lscpu(text):
    rows = []
    for line in text.splitlines():
        if line and not line.startswith('#'):
            rows.append(tuple(int(x) for x in line.split(',')[:4]))
    return rows


def gpu_affinity(topo_text):
    """{gpu index: (cpu list, numa node or None)} from `nvidia-smi topo -m`."""
    lines = [ANSI.sub('', x) for x in topo_text.splitlines()]
    header = next((x.split('\t') for x in lines if 'CPU Affinity' in x), None)
    if header is None:
        raise RuntimeError('nvidia-smi topo -m has no CPU Affinity column')
    names = [h.strip() for h in header]
    cpu_col = names.index('CPU Affinity')
    numa_col = names.index('NUMA Affinity') if 'NUMA Affinity' in names else None
    result = {}
    for line in lines:
        match = re.match(r'GPU(\d+)\t', line)
        if not match:
            continue
        cells = [c.strip() for c in line.split('\t')]
        numa = cells[numa_col] if numa_col is not None and numa_col < len(cells) else ''
        result[int(match[1])] = (parse_cpulist(cells[cpu_col]), int(numa) if numa.isdigit() else None)
    return result


def derive_topology(lscpu_text, topo_text, gpus):
    """Server CPUs = every logical CPU of the socket the GPUs hang off (all GPUs must share
    one socket and one NUMA node). Client CPUs = one logical CPU on each of 8 physical cores
    of another socket; runner and monitor = one on each of 2 further cores of that socket,
    so neither shares a core (or an SMT sibling) with a client."""
    rows = parse_lscpu(lscpu_text)
    affinity = gpu_affinity(topo_text)
    missing = [g for g in gpus if g not in affinity]
    if missing:
        raise RuntimeError(f'GPUs {missing} not in nvidia-smi topo -m')
    cpu_sets = {tuple(affinity[g][0]) for g in gpus}
    if len(cpu_sets) != 1:
        raise RuntimeError(f'GPUs {list(gpus)} do not share one CPU affinity: {sorted(cpu_sets)}')
    affine = list(cpu_sets.pop())
    sockets = {r[2] for r in rows if r[0] in affine}
    nodes = {r[3] for r in rows if r[0] in affine}
    if len(sockets) != 1 or len(nodes) != 1:
        raise RuntimeError('The GPUs\' CPU affinity spans several sockets or NUMA nodes')
    socket, node = sockets.pop(), nodes.pop()
    server = sorted(r[0] for r in rows if r[2] == socket)
    if server != affine:
        raise RuntimeError(f'GPU CPU affinity {affine} is not exactly the CPUs of socket {socket}')
    numa = {affinity[g][1] for g in gpus}
    if numa != {node}:
        raise RuntimeError(f'nvidia-smi NUMA affinity {numa} differs from lscpu node {node}')
    others = sorted({r[2] for r in rows} - {socket})
    if not others:
        raise RuntimeError('The client needs a second socket')
    client_socket = others[0]
    seen, client, runner = set(), [], []
    for cpu, core, sock, _ in rows:
        if sock == client_socket and core not in seen and len(client) < CLIENT_CORES:
            client.append(cpu)
            seen.add(core)
    for cpu, core, sock, _ in rows:
        if sock == client_socket and core not in seen and len(runner) < 2:
            runner.append(cpu)
            seen.add(core)
    if len(client) != CLIENT_CORES or len(runner) != 2:
        raise RuntimeError('The client socket has fewer than 10 physical cores')
    return {'gpus': list(gpus), 'server_socket': socket, 'server_node': node, 'server_cpus': server,
            'client_socket': client_socket, 'client_cpus': client, 'runner_cpus': runner}


def topology(gpus, expected_cards):
    """Placement and the number of cards gate Q covers. The machine must show exactly
    `expected_cards` cards now; from then on every snapshot must show the same number."""
    global EXPECTED_CARDS
    EXPECTED_CARDS = None
    found = len(gpu_snapshot()['gpus'])
    if found != expected_cards:
        raise RuntimeError(f'nvidia-smi shows {found} cards, expected {expected_cards} (--expect-cards)')
    EXPECTED_CARDS = expected_cards
    lscpu = command(['lscpu', '-p=CPU,CORE,SOCKET,NODE'])
    topo = command(['nvidia-smi', 'topo', '-m'])
    result = derive_topology(lscpu, topo, gpus)
    result.update(lscpu=lscpu, gpu_topology=topo, p2p_read=command(['nvidia-smi', 'topo', '-p2p', 'r']),
                  card_count=expected_cards)
    return result


def process_table(proc=Path('/proc')):
    result = {}
    for path in Path(proc).glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            rest = text[text.rindex(')') + 2:].split()
            result[int(path.parent.name)] = {'state': rest[0], 'ppid': int(rest[1]),
                                            'session': int(rest[3]), 'ticks': int(rest[11]) + int(rest[12]),
                                            'start': int(rest[19])}
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
    return result


def descendants(table, pid):
    found = {pid} if pid in table else set()
    while True:
        more = {p for p, row in table.items() if row['ppid'] in found}
        if more <= found:
            return found
        found |= more


def assert_server_cards(pid, snapshot, gpus, table=None):
    """The server's compute processes must sit on exactly the intended physical cards
    (matched by UUID), and no other process may compute on any card."""
    owned = descendants(table if table is not None else process_table(), pid)
    uuid_to_index = {row['uuid']: row['index'] for row in snapshot['gpus']}
    apps = snapshot['compute_apps']
    if any(row['pid'] not in owned for row in apps):
        raise RuntimeError('Foreign compute process at readiness')
    actual = {uuid_to_index.get(row['uuid']) for row in apps if row['pid'] in owned}
    if actual != set(gpus):
        raise RuntimeError(f'Server compute cards {sorted(actual, key=str)} differ from intended {sorted(gpus)}')
    return {'intended_indices': sorted(gpus), 'observed_indices': sorted(actual),
            'compute_apps': apps, 'uuid_to_index': uuid_to_index, 'owned_pids': sorted(owned)}


def anonymous_placement(raw, node):
    """Anonymous memory of one process and a lower bound of the share on `node`.

    A copy-on-write mapping can hold file and anonymous pages; every page on another node
    is charged to anonymous memory, so the share is a lower bound and never a false pass."""
    total = lower = 0
    for line in raw.splitlines():
        fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
        anon = int(fields.get('anon', '0'))
        if not anon:
            continue
        size = int(fields.get('kernelpagesize_kB', '4'))
        other = sum(int(v) for k, v in fields.items()
                    if re.fullmatch(r'N\d+', k) and int(k[1:]) != node)
        total += anon * size
        lower += max(0, anon - other) * size
    return {'anonymous_kib': total, 'node_lower_bound_kib': lower,
            'node_fraction_lower_bound': lower / total if total else None}


def nccl_shm_pages(raw):
    """Pages per NUMA node of NCCL's /dev/shm staging buffers mapped by one process.

    When NCCL cannot use cuMem host allocation (it needs a driver reporting CUDA 12.6 or
    later), its SHM transport maps files under /dev/shm/nccl-*; their placement is visible
    only in numa_maps. Shared pages are counted in every process that maps them; only the
    split between nodes matters."""
    nodes = {}
    for line in raw.splitlines():
        fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
        if not fields.get('file', '').startswith('/dev/shm/nccl-'):
            continue
        for key, value in fields.items():
            match = re.fullmatch(r'N(\d+)', key)
            if match:
                nodes[match[1]] = nodes.get(match[1], 0) + int(value)
    return nodes


def placement(pid, directory, node):
    """Save numa_maps of every live process of the server tree and summarise each."""
    directory = Path(directory)
    directory.mkdir()
    table = process_table()
    rows = {}
    for p in sorted(descendants(table, pid)):
        if table[p]['state'] == 'Z':
            continue
        try:
            raw = Path(f'/proc/{p}/numa_maps').read_text()
            (directory / f'{p}.numa_maps').write_text(raw)
            rows[str(p)] = {**anonymous_placement(raw, node), 'nccl_shm_pages': nccl_shm_pages(raw),
                            'start': table[p]['start'], 'affinity': sorted(os.sched_getaffinity(p))}
        except (OSError, ValueError) as exc:
            rows[str(p)] = {'error': str(exc)}
    return rows


def cpu_snapshot(text=None):
    rows = {}
    for line in (text if text is not None else Path('/proc/stat').read_text()).splitlines():
        fields = line.split()
        if fields and re.fullmatch(r'cpu\d+', fields[0]):
            values = list(map(int, fields[1:9]))  # guest time is already inside user time
            rows[fields[0][3:]] = {'total': sum(values), 'busy': sum(values) - values[3] - values[4]}
    return rows


# Fixed at start by topology(); a snapshot with another number of cards raises, so inside
# a measured level it becomes an errored telemetry sample and gate Q fails.
EXPECTED_CARDS = None


def parse_gpus(query_text, apps_text, expected=None):
    rows = []
    for parts in csv.reader(query_text.splitlines(), skipinitialspace=True):
        if not parts:
            continue
        idx, uuid, clock, power, temp, util, memory = parts
        rows.append({'index': int(idx), 'uuid': uuid, 'clock_mhz': float(clock),
                     'power_w': float(power), 'temperature_c': float(temp),
                     'utilization_pct': float(util), 'memory_mib': float(memory)})
    if not rows or sorted(r['index'] for r in rows) != list(range(len(rows))):
        raise RuntimeError('Cannot observe every card')
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f'nvidia-smi shows {len(rows)} cards, expected {expected}')
    apps = [{'pid': int(p[0]), 'uuid': p[1], 'name': p[2]}
            for p in csv.reader(apps_text.splitlines(), skipinitialspace=True) if p]
    return {'gpus': rows, 'compute_apps': apps}


def gpu_snapshot():
    fields = 'index,uuid,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used'
    query = command(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'])
    apps = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name',
                    '--format=csv,noheader,nounits'])
    return parse_gpus(query, apps, EXPECTED_CARDS)


# Pids of live touch-and-free helpers. They run before a server starts, outside every
# measured window, but any sample that sees one counts it as the run's own process.
HELPER_PIDS = set()


class Monitor:
    """A 1 Hz sampler on the runner's own CPUs: per-CPU counters, the server tree's CPU
    ticks, and clock, power, temperature, utilisation and compute processes of every card."""

    def __init__(self, pid, directory, cpus):
        self.pid = pid
        self.cpus = cpus
        self.path = Path(directory) / 'telemetry.jsonl'
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        table = process_table()
        owned = descendants(table, self.pid) | {p for p in set(HELPER_PIDS) if p in table}
        return {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns(),
                'cpu': cpu_snapshot(), 'owned': {str(p): table[p] for p in owned},
                **gpu_snapshot()}

    def boundary(self):
        """An enclosing sample; a failed read is recorded as an errored sample (gate Q fails)."""
        try:
            return self.sample()
        except Exception as exc:
            return {'monotonic_ns': time.monotonic_ns(), 'error': str(exc)}

    def run(self):
        os.sched_setaffinity(0, self.cpus)
        with self.path.open('x') as dest:
            while not self.stop.is_set():
                before = time.monotonic()
                try:
                    row = self.sample()
                except Exception as exc:
                    row = {'monotonic_ns': time.monotonic_ns(), 'error': str(exc)}
                dest.write(json.dumps(row) + '\n')
                dest.flush()
                self.stop.wait(max(0, 1 - (time.monotonic() - before)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=45)
        if self.thread.is_alive():
            raise RuntimeError('Telemetry failed to stop')
