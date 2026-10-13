"""Touch-and-free: make the kernel reclaim page cache on one NUMA node before a server starts.

The servers set a preferred-node memory policy for the node their GPUs hang off. On a machine
whose page cache fills that node, a preferred allocation falls back to the other node instead
of reclaiming cache, and the servers' memory (and NCCL's host staging buffers) land on the
wrong node. So before every server start one helper process:

1. starts its own session and sets its ``oom_score_adj`` to 1000 before anything is
   allocated, so that if the kernel's OOM killer has to choose, it chooses the helper;
2. binds its memory with ``MPOL_BIND`` to the named node only (and, if given, its CPUs to
   that node's cores);
3. reads the node's reclaimable memory and refuses, allocating nothing, unless it exceeds
   the bytes to write plus 4e9 bytes;
4. writes the bytes (40e9 by default; 1 GiB anonymous chunks, every byte), keeps them
   mapped until all are written, then unmaps them.

Only clean page cache is reclaimed; no process memory is touched. Other users' cached
files on that node are evicted and their next reads are slower.

Reclaimable memory = (Normal-zone free pages minus the Normal zone's low watermark, floored
at 0, from /proc/zoneinfo) + the node's Active(file) and Inactive(file) (from
/sys/devices/system/node/nodeN/meminfo). Units are decimal bytes everywhere: zoneinfo pages
x 4096, meminfo kB x 1024. "Usable free" is Normal-zone free minus its low watermark, not
floored, in bytes. The helper uses no CUDA.
"""
import ctypes
import json
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import re
import time
import traceback

PAGE_BYTES = 4096
KIB = 1024
TOUCH_BYTES = 40 * 10**9
MARGIN_BYTES = 4 * 10**9
REQUIRED_BYTES = TOUCH_BYTES + MARGIN_BYTES  # strictly exceeded, or the helper refuses
OOM_SCORE_ADJ = 1000
CHUNK = 1 << 30
MPOL_BIND = 2
SYS_SET_MEMPOLICY = 238  # x86_64
SYS_GET_MEMPOLICY = 239  # x86_64


class Refusal(RuntimeError):
    """The helper refused on at least one node: reclaimable memory <= required bytes.
    Nothing was allocated on a refusing node. ``record`` holds every node's row."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


# ------------------------------------------------------------------ readings

ZONE_HEADER = re.compile(r'^Node\s+(\d+),\s+zone\s+(\S+)\s*$')


def parse_zoneinfo(text):
    """{(node, zone): {'free': pages, 'min': pages, 'low': pages, 'high': pages}}.

    In /proc/zoneinfo each zone opens with 'Node N, zone NAME'; the first zone of
    a node may carry a 'per-node stats' block before its 'pages free' line. The
    watermark lines are the bare 'min', 'low', 'high' lines that follow
    'pages free'."""
    zones, current = {}, None
    for line in text.splitlines():
        m = ZONE_HEADER.match(line)
        if m:
            current = zones.setdefault((int(m[1]), m[2]), {})
            continue
        if current is None:
            continue
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ['pages', 'free'] and parts[2].isdigit():
            current['free'] = int(parts[2])
        elif len(parts) == 2 and parts[0] in ('min', 'low', 'high') and parts[1].isdigit() \
                and 'free' in current and parts[0] not in current:
            current[parts[0]] = int(parts[1])
    return zones


def parse_node_meminfo(text, node):
    """{'MemFree': bytes, 'Active(file)': bytes, 'Inactive(file)': bytes, ...} from
    /sys/devices/system/node/nodeN/meminfo ('Node N Key: value kB')."""
    out = {}
    for line in text.splitlines():
        m = re.match(rf'^Node {node} ([^:]+):\s+(\d+)(\s+kB)?\s*$', line)
        if m:
            out[m[1]] = int(m[2]) * (KIB if m[3] else 1)
    return out


def snapshot(node, zoneinfo_text=None, meminfo_text=None):
    """One reading of a node's memory, decimal bytes. Raises ValueError when a
    needed line is missing (missing evidence never passes)."""
    if zoneinfo_text is None:
        zoneinfo_text = Path('/proc/zoneinfo').read_text()
    if meminfo_text is None:
        meminfo_text = Path(f'/sys/devices/system/node/node{node}/meminfo').read_text()
    normal = parse_zoneinfo(zoneinfo_text).get((node, 'Normal'))
    if not normal or 'free' not in normal or 'low' not in normal:
        raise ValueError(f'/proc/zoneinfo has no Normal zone free/low lines for node {node}')
    info = parse_node_meminfo(meminfo_text, node)
    missing = [k for k in ('MemFree', 'Active(file)', 'Inactive(file)') if k not in info]
    if missing:
        raise ValueError(f'node{node}/meminfo lacks {missing}')
    usable = (normal['free'] - normal['low']) * PAGE_BYTES
    files = info['Active(file)'] + info['Inactive(file)']
    return {'node': node, 'time_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
            'normal_free_pages': normal['free'], 'normal_low_pages': normal['low'],
            'usable_free_bytes': usable, 'memfree_bytes': info['MemFree'],
            'active_file_bytes': info['Active(file)'], 'inactive_file_bytes': info['Inactive(file)'],
            'reclaimable_bytes': max(0, usable) + files, 'units': 'decimal bytes; pages x 4096; kB x 1024'}


def admits(reading, required=REQUIRED_BYTES):
    return reading['reclaimable_bytes'] > required


# ------------------------------------------------------------------ process

def set_oom_score_adj(value=OOM_SCORE_ADJ, path='/proc/self/oom_score_adj'):
    Path(path).write_text(f'{value}\n')
    got = int(Path(path).read_text().strip())
    if got != value:
        raise RuntimeError(f'oom_score_adj is {got}, not {value}')
    return got


def bind_node(node):
    """MPOL_BIND to one node for this process, verified with get_mempolicy."""
    if platform.machine() != 'x86_64':
        raise RuntimeError('The set/get_mempolicy syscall numbers are x86_64 only')
    libc = ctypes.CDLL(None, use_errno=True)
    mask = ctypes.c_ulong(1 << node)
    if libc.syscall(SYS_SET_MEMPOLICY, MPOL_BIND, ctypes.byref(mask), ctypes.c_ulong(64)) != 0:
        raise OSError(ctypes.get_errno(), 'set_mempolicy(MPOL_BIND)')
    mode, actual = ctypes.c_int(), ctypes.c_ulong()
    if libc.syscall(SYS_GET_MEMPOLICY, ctypes.byref(mode), ctypes.byref(actual), ctypes.c_ulong(64),
                    ctypes.c_void_p(), 0) != 0:
        raise OSError(ctypes.get_errno(), 'get_mempolicy')
    if mode.value != MPOL_BIND or actual.value != mask.value:
        raise RuntimeError('Memory policy did not take effect')
    return {'mode': 'MPOL_BIND', 'nodemask': actual.value}


def touch(nbytes):
    """Write nbytes of anonymous memory, keep it mapped until all is written, free it."""
    libc = ctypes.CDLL(None)
    chunks, touched = [], 0
    try:
        while touched < nbytes:
            size = min(CHUNK, nbytes - touched)
            m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            chunks.append(m)
            view = ctypes.c_char.from_buffer(m)
            libc.memset(ctypes.c_void_p(ctypes.addressof(view)), 0x5A, ctypes.c_size_t(size))
            del view  # release the buffer export so the map can be closed
            touched += size
    finally:
        for m in chunks:
            m.close()
    return touched


def worker(node, nbytes, cpus, required, result_queue):
    """Child process body. Order: own session, oom_score_adj, CPU affinity,
    MPOL_BIND, reclaimable reading and refusal, touch, free, reading."""
    try:
        os.setsid()
        oom = set_oom_score_adj()
        if cpus:
            os.sched_setaffinity(0, set(cpus))
        policy = bind_node(node)
        before = snapshot(node)
        row = {'node': node, 'pid': os.getpid(), 'oom_score_adj': oom, 'policy': policy,
               'cpus': sorted(os.sched_getaffinity(0)), 'required_bytes': required,
               'touch_bytes': nbytes, 'before': before}
        if not admits(before, required):
            row.update(refused=True, bytes=0,
                       reason=f'node {node}: reclaimable {before["reclaimable_bytes"]} bytes '
                              f'<= {required} bytes; nothing allocated')
            result_queue.put(row)
            return
        start = time.monotonic()
        row['bytes'] = touch(nbytes)
        row['seconds'] = time.monotonic() - start
        row['after'] = snapshot(node)
        row['refused'] = False
        result_queue.put(row)
    except BaseException:
        result_queue.put({'node': node, 'pid': os.getpid(), 'error': traceback.format_exc()})
        raise


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    temp.replace(path)


def two_nodes():
    """The helper only runs on a machine with at least two NUMA nodes, so that a one-node
    workstation never runs the 40e9-byte helper by accident."""
    return Path('/sys/devices/system/node/node1').exists()


def run(nodes, directory, owned, cpus_of=None, nbytes=TOUCH_BYTES, required=REQUIRED_BYTES, timeout=900):
    """One helper process per node in ``nodes``, in parallel. ``owned`` is the
    runner's ownership set (add/discard): each helper pid is in it while the
    helper lives. Writes ``directory/touch.json`` and returns {node: row}.

    Raises Refusal if any node refused (the other nodes' rows are kept), and
    RuntimeError on any helper failure (a harness failure, not a gate result).
    No helper outlives run(): on any exit path (stall timeout, non-zero exit,
    an exception or KeyboardInterrupt in the parent) every helper still alive is
    killed (SIGKILL; it is in its own session, so the runner's group signal would
    not reach it) and joined, and the kill is recorded in ``killed.json``."""
    if not two_nodes():
        raise RuntimeError('The touch-and-free helper needs a machine with two NUMA nodes')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = []
    try:
        for node in nodes:
            p = ctx.Process(target=worker, args=(node, nbytes, (cpus_of or {}).get(node), required, q))
            p.start()
            owned.add(p.pid)
            procs.append(p)
        rows = []
        for _ in procs:
            try:
                rows.append(q.get(timeout=timeout))
            except queue.Empty:
                raise RuntimeError(f'Touch-and-free helper stalled for {timeout} s; it is killed')
        result = {r['node']: r for r in sorted(rows, key=lambda r: r['node'])}
        write_json(directory / 'touch.json', {str(k): v for k, v in result.items()})
        for p in procs:
            p.join(timeout=60)
            if p.is_alive() or p.exitcode != 0:
                raise RuntimeError(f'Touch-and-free helper failed (exit {p.exitcode}); see {directory}')
            owned.discard(p.pid)
        if any('error' in r for r in rows):
            raise RuntimeError(str(rows))
        refused = [r for r in rows if r.get('refused')]
        if refused:
            raise Refusal('; '.join(r['reason'] for r in refused), result)
        return result
    finally:
        reap(procs, owned, directory)
        q.close()


def reap(procs, owned, directory):
    """Kill and join every helper still alive; drop dead helpers from ``owned``.
    A helper that survives SIGKILL and a 60 s join stays in ``owned`` (the runner's
    failure path then waits on it) and is recorded."""
    killed = []
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(timeout=60)
            killed.append({'pid': p.pid, 'alive_after_kill': p.is_alive(), 'exitcode': p.exitcode})
        elif p.exitcode is None:
            p.join(timeout=60)
        if not p.is_alive():
            owned.discard(p.pid)
    if killed:
        write_json(Path(directory) / 'killed.json', killed)
    return killed
