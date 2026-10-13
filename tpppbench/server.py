"""vLLM server lifecycle and the evidence read from each start.

A start that fails or hangs is never killed automatically: the server is left running,
`MANUAL_ATTENTION.json` names its pid, and the operator decides. A completed start is shut
down normally (SIGTERM) and the next start waits until every process of the old server has
exited and the cards' memory is back to its value before the start.
"""
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import telemetry
from .spec import (ARM_FLAGS, ARMS, CAPACITY, DECLARED_DEVIATIONS, HTTP_TIMEOUT_S, MODEL, MODEL_REVISION, NCCL_HANG_S,
                   PP_SIZE, READY_TIMEOUT_S, SHUTDOWN_TIMEOUT_S, VLLM_COMMIT, VLLM_VERSION, digest,
                   write_json)


class StartFailure(RuntimeError):
    """A server did not start or did not answer (gate S), as opposed to missing evidence."""


# Inherited variables with these prefixes would change what is measured; they stop a start.
DENIED_PREFIXES = ('PYTORCH_', 'OMP_', 'VLLM_', 'NCCL_', 'CUDA_', 'TORCH_NCCL_', 'TORCHINDUCTOR_', 'TRITON_')
MANAGED = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')


def cache_environment(cache_dir):
    """Compile caches the launcher sets; an inherited variable with the same value is harmless."""
    cache_dir = Path(cache_dir)
    return {'VLLM_CACHE_ROOT': str(cache_dir / 'vllm'), 'TRITON_CACHE_DIR': str(cache_dir / 'triton'),
            'TORCHINDUCTOR_CACHE_DIR': str(cache_dir / 'inductor')}


def installed_versions():
    """vLLM version and the commit in its _version.py, read without importing vLLM."""
    spec = importlib.util.find_spec('vllm')
    if spec is None or not spec.origin:
        raise RuntimeError('vLLM is not installed in this Python environment')
    version_file = Path(spec.origin).parent / '_version.py'
    text = version_file.read_text() if version_file.exists() else ''
    return {'vllm': importlib.metadata.version('vllm'),
            'vllm_commit_in_version_file': VLLM_COMMIT[:7] in text,
            'torch': importlib.metadata.version('torch'),
            'transformers': importlib.metadata.version('transformers')}


def check_versions(allow_other):
    versions = installed_versions()
    ok = versions['vllm'] == VLLM_VERSION and versions['vllm_commit_in_version_file']
    if not ok and not allow_other:
        raise RuntimeError(f'Needs vllm=={VLLM_VERSION} (commit {VLLM_COMMIT[:7]}); found {versions}. '
                           'Use --allow-other-versions to run anyway (recorded).')
    versions['reference_versions'] = ok
    return versions


def vllm_interpreter(path, allow_other):
    """The server executable must exist and run on this interpreter, so the version check
    is about the vLLM the server actually runs. Returns (interpreter, same)."""
    path = Path(path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f'--vllm {path} is not an executable file')
    first = path.read_bytes()[:512].split(b'\n', 1)[0].decode(errors='replace')
    if not first.startswith('#!'):
        raise RuntimeError(f'--vllm {path} is not a Python console script (no #! line)')
    interpreter = first[2:].strip().split()[0] if first[2:].strip() else ''
    same = bool(interpreter) and os.path.realpath(interpreter) == os.path.realpath(sys.executable)
    if not same and not allow_other:
        raise RuntimeError(f'--vllm {path} runs on {interpreter}, not on this interpreter {sys.executable}; '
                           'use the vllm of this environment or --allow-other-versions (recorded)')
    return interpreter, same


def check_vllm(path, versions, allow_other, run=subprocess.run):
    """vllm_interpreter, then the version the executable itself reports (recorded).
    Any problem is an operator error raised before any start, never a gate result."""
    path = Path(path)
    interpreter, same = vllm_interpreter(path, allow_other)
    result = run([str(path), '--version'], capture_output=True, text=True, timeout=600)
    reported = (result.stdout or '').strip().splitlines()[-1:] if result.returncode == 0 else []
    reported = reported[0].strip() if reported else None
    if reported != versions.get('vllm') and not allow_other:
        raise RuntimeError(f'{path} --version reported {reported!r} (exit {result.returncode}), '
                           f'installed vllm is {versions.get("vllm")}')
    return {'vllm_executable': str(path), 'vllm_interpreter': interpreter, 'same_interpreter': same,
            'vllm_reported_version': reported, 'vllm_version_exit_code': result.returncode}


def environment(cfg, deviations, kind, nccl_debug_dir=None, environ=None):
    """The server's environment. `kind` is start-test, settings, transport or timed."""
    environ = os.environ if environ is None else environ
    if any(DECLARED_DEVIATIONS.get(k) != v for k, v in deviations.items()):
        raise RuntimeError('Undeclared startup deviation')
    caches = cache_environment(cfg.cache_dir)
    unsafe = [k for k, v in environ.items()
              if (k == 'LD_PRELOAD' or k.startswith(DENIED_PREFIXES)) and k not in MANAGED
              and not (k in deviations and deviations[k] == v)
              and not caches.get(k) == v]
    if unsafe:
        raise RuntimeError(f'Inherited engine or transport overrides would change the measurement: {sorted(unsafe)}')
    env = dict(environ)
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, cfg.gpus)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', **caches)
    env.update(deviations)
    if kind == 'transport':
        env['NCCL_DEBUG'] = 'INFO'
        # INIT keeps the channel, Check P2P Type and Init COMPLETE lines; ALLOC and SHM add
        # the host-buffer lines.
        env['NCCL_DEBUG_SUBSYS'] = 'INIT,ALLOC,SHM,P2P'
        if nccl_debug_dir is None:
            raise RuntimeError('A transport start needs its directory for NCCL_DEBUG_FILE')
        # Log destination only: one line-buffered file per process.
        env['NCCL_DEBUG_FILE'] = str(Path(nccl_debug_dir) / 'nccl.%h.%p.log')
    if kind == 'settings':
        env['VLLM_LOGGING_LEVEL'] = 'DEBUG'
    return env


def argv(cfg, arm):
    """Every server flag but the parallelism flag is common to both arms."""
    if arm not in ARMS:
        raise ValueError(arm)
    result = [str(cfg.vllm), 'serve', str(cfg.model), '--served-model-name', MODEL,
              '--host', '127.0.0.1', '--port', str(cfg.port), ARM_FLAGS[arm], '4']
    return result


def http(url, method='GET', timeout=HTTP_TIMEOUT_S):
    with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=timeout) as response:
        return response.read().decode()


def check_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


def preflight(port):
    """Before anything of a start runs: no compute process on any card and the port free."""
    gpu = telemetry.gpu_snapshot()
    if gpu['compute_apps']:
        raise RuntimeError('A compute process is present on a card; it is left untouched')
    check_port(port)
    return gpu


def assert_server_identity(process, start, port, model_data, model_path, proc=Path('/proc')):
    """The answering server is our process tree: same root pid and start time, the
    listening socket owned by the tree, and the expected model name and path."""
    table = telemetry.process_table(proc)
    if process.poll() is not None or table.get(process.pid, {}).get('start') != start:
        raise RuntimeError('Answering server process exited or changed identity')
    owned = telemetry.descendants(table, process.pid)
    listeners = set()
    for line in (proc / 'net/tcp').read_text().splitlines()[1:]:
        fields = line.split()
        address, hexport = fields[1].split(':')
        if int(hexport, 16) == port and fields[3] == '0A' and address in ('0100007F', '00000000'):
            listeners.add(fields[9])
    sockets = set()
    for pid in owned:
        for fd in (proc / str(pid) / 'fd').glob('*'):
            try:
                target = os.readlink(fd)
            except (FileNotFoundError, PermissionError):
                continue
            match = re.fullmatch(r'socket:\[(\d+)\]', target)
            if match:
                sockets.add(match[1])
    if not listeners or not listeners <= sockets:
        raise RuntimeError('The listening socket is not owned by the server tree')
    models = model_data.get('data', [])
    if len(models) != 1 or models[0].get('id') != MODEL or models[0].get('root') != str(model_path):
        raise RuntimeError('The answering server serves another model name or path')
    return {'pid': process.pid, 'start': start, 'owned_pids': sorted(owned),
            'listening_inodes': sorted(listeners), 'models': model_data}


def lock_paths(output, lock_dir, gpus, extra=()):
    """run.lock, one file per GPU in lock_dir, then every --lock PATH with {gpu} expanded
    for each GPU; duplicates (same resolved path) are taken once."""
    paths = [Path(output) / 'run.lock', *(Path(lock_dir) / f'gpu{g}.lock' for g in gpus)]
    for pattern in extra:
        pattern = str(pattern)
        paths += [Path(pattern.format(gpu=g)) for g in gpus] if '{gpu}' in pattern else [Path(pattern)]
    seen, result = set(), []
    for p in paths:
        key = os.path.realpath(p)
        if key not in seen:
            seen.add(key)
            result.append(p)
    for p in result:
        p.parent.mkdir(parents=True, exist_ok=True)
    return result


class Locks:
    """flock on the output directory and on one file per GPU. The descriptors are passed to
    the server, so the locks outlive a runner that dies while its server keeps running."""

    def __init__(self, output, lock_dir, gpus, extra=()):
        self.files = []
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(f'{output} is not empty; every run uses a new directory')
        lock_dir = Path(lock_dir)
        lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            for path in lock_paths(output, lock_dir, gpus, extra):
                file = path.open('a+')
                self.files.append(file)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.close()
            raise

    @property
    def fds(self):
        return tuple(f.fileno() for f in self.files)

    def close(self):
        # Close, not LOCK_UN: a server that inherited the descriptors keeps the locks.
        for f in self.files:
            f.close()
        self.files.clear()


# ------------------------------------------------------------------ declared deviations

def nccl_progress(log):
    marker = log.find('vLLM is using nccl==')
    if marker < 0:
        return False
    later = log[marker:].splitlines()[1:]
    return any(re.search(r'load(?:ing)?.*(?:weight|model|checkpoint)|(?:weight|model).*load|'
                         r'compil|captur.*(?:graph|cuda)|Application startup complete',
                         line, re.I) for line in later)


def startup_signature(log, returncode, timed_out, nccl_observed_seconds=0):
    """The declared deviation a failed default start licenses, or None.

    NCCL_CUMEM_HOST_ENABLE=0: a segfault mentioning cuMemCreate. NCCL_P2P_DISABLE=1: a hang
    after "vLLM is using nccl==" with no later weight-loading or compilation progress line
    for 30 minutes."""
    worker_segv = re.search(r'Worker proc .* died unexpectedly \(exit code: -11\)', log)
    fatal_segv = re.search(r'Fatal Python error: Segmentation fault', log)
    if (returncode in (-signal.SIGSEGV, 139) or worker_segv or fatal_segv) and 'cuMemCreate' in log:
        return 'NCCL_CUMEM_HOST_ENABLE'
    if (timed_out and returncode is None and nccl_observed_seconds >= NCCL_HANG_S
            and 'vLLM is using nccl==' in log and not nccl_progress(log)):
        return 'NCCL_P2P_DISABLE'
    return None


def deviations_from_evidence(start_dirs):
    """Declared deviations licensed by failed default start-test starts (their
    directories hold start-failure.json, manifest.json and server.log)."""
    result = {}
    for directory in start_dirs or ():
        directory = Path(directory)
        failure = json.loads((directory / 'start-failure.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest.get('deviations') != {} or failure.get('deviations') != {}:
            raise RuntimeError('A deviation needs a failed start with no deviation applied')
        if manifest.get('kind') != 'start-test' or manifest.get('arm') not in ARMS:
            raise RuntimeError('A deviation needs a failed start-test start')
        if digest(directory / 'server.log') != failure.get('log_sha256'):
            raise RuntimeError('The failed start\'s server log changed after the failure was recorded')
        key = startup_signature((directory / 'server.log').read_text(), failure['returncode'],
                                failure['timed_out'], failure.get('nccl_observed_seconds', 0))
        if not key:
            raise RuntimeError(f'{directory}: the failure does not match a declared deviation\'s signature')
        result[key] = DECLARED_DEVIATIONS[key]
    return result


# ------------------------------------------------------------------ server

class Server:
    def __init__(self, cfg, arm, directory, locks, deviations, kind):
        self.cfg = cfg
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.process = None
        self.completed = False
        self.url = f'http://127.0.0.1:{cfg.port}'
        self.port = cfg.port
        self.gpus = list(cfg.gpus)
        env = environment(cfg, deviations, kind, nccl_debug_dir=self.directory if kind == 'transport' else None)
        gpu = telemetry.gpu_snapshot()
        self.before_gpus = gpu
        self.known_processes = {}
        write_json(self.directory / 'before-gpus.json', gpu)
        if gpu['compute_apps']:
            raise RuntimeError('A compute process is present on a card; it is left untouched')
        check_port(cfg.port)
        node = cfg.topology['server_node']
        (self.directory / 'node-memory-before.txt').write_text(
            Path(f'/sys/devices/system/node/node{node}/meminfo').read_text())
        command = argv(cfg, arm)
        wrapper = [sys.executable, str(Path(__file__).with_name('placement.py')),
                   '--cpus', ','.join(map(str, cfg.topology['server_cpus'])), '--node', str(node),
                   '--receipt', str(self.directory / 'placement-exec.json'), '--', *command]
        generation = Path(cfg.model) / 'generation_config.json'
        manifest = {'arm': arm, 'kind': kind, 'argv': wrapper, 'model': MODEL, 'model_revision': MODEL_REVISION,
                    'model_path': str(cfg.model),
                    'deviations': deviations, 'versions': cfg.versions,
                    'nvidia_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                                        if (d.metadata['Name'] or '').lower().startswith('nvidia-')},
                    'environment': {k: v for k, v in env.items() if not secret_name(k)},
                    'generation_config': json.loads(generation.read_text()) if generation.exists() else None,
                    'generation_config_sha256': digest(generation) if generation.exists() else None,
                    'topology': {k: v for k, v in cfg.topology.items() if k != 'lscpu'},
                    'start_wall_ns': time.time_ns()}
        write_json(self.directory / 'manifest.json', manifest)
        self.deviations = dict(deviations)
        self.log = (self.directory / 'server.log').open('x')
        try:
            self.process = subprocess.Popen(wrapper, stdin=subprocess.DEVNULL, stdout=self.log,
                                            stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                            pass_fds=locks.fds if locks else ())
            self.identity = telemetry.process_table()[self.process.pid]['start']
            write_json(self.directory / 'server.pid.json', {'pid': self.process.pid, 'start': self.identity})
        except BaseException:
            self.retain('Server constructor failed after a possible launch')
            raise

    def ready(self, timeout=READY_TIMEOUT_S):
        deadline = time.monotonic() + timeout
        nccl_seen = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            log = (self.directory / 'server.log').read_text(errors='replace')
            if nccl_seen is None and 'vLLM is using nccl==' in log:
                nccl_seen = time.monotonic()
                deadline = max(deadline, nccl_seen + NCCL_HANG_S)
            try:
                if http(self.url + '/health') is not None:
                    identity = assert_server_identity(self.process, self.identity, self.port,
                                                      json.loads(http(self.url + '/v1/models')), self.cfg.model)
                    write_json(self.directory / 'ready-identity.json', identity)
                    cards = telemetry.assert_server_cards(self.process.pid, telemetry.gpu_snapshot(), self.gpus)
                    write_json(self.directory / 'ready-cards.json', cards)
                    self.track_processes()
                    return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        code = self.process.poll()
        self.log.flush()
        write_json(self.directory / 'start-failure.json',
                   {'returncode': code, 'timed_out': code is None,
                    'nccl_observed_seconds': time.monotonic() - nccl_seen if nccl_seen is not None else 0,
                    'log_sha256': digest(self.directory / 'server.log'),
                    'deviations': self.deviations,
                    'manifest_sha256': digest(self.directory / 'manifest.json')})
        self.retain('Start failed or timed out; it was not killed')
        raise StartFailure(f'{self.directory}: start failed; the server was not killed automatically')

    def retain(self, reason):
        try:
            write_json(self.directory / 'MANUAL_ATTENTION.json',
                       {'pid': self.process.pid if self.process else None, 'reason': reason,
                        'action': 'Inspect this pid and its run time; stopping it is left to the operator.'})
        finally:
            if getattr(self, 'log', None) and not self.log.closed:
                self.log.flush()

    def close_completed(self, timeout=SHUTDOWN_TIMEOUT_S):
        """Normal end of a completed start: SIGTERM, then wait for every process of the tree
        to exit and for the cards' memory to return to its value before the start."""
        self.track_processes()
        deadline = time.monotonic() + timeout
        if self.process.poll() is not None:
            if self.process.returncode != 0:
                self.retain('Server exited unexpectedly')
                raise RuntimeError('Server exited unexpectedly')
        else:
            table = telemetry.process_table()
            if table[self.process.pid]['start'] != self.identity:
                raise RuntimeError('Server pid changed identity')
            os.kill(self.process.pid, signal.SIGTERM)
        baseline = {g['index']: g['memory_mib'] for g in self.before_gpus['gpus'] if g['index'] in self.gpus}
        while True:
            self.process.poll()
            table = self.track_processes()
            alive = {p: start for p, start in self.known_processes.items()
                     if table.get(p, {}).get('start') == start and table[p]['state'] != 'Z'}
            gpu = telemetry.gpu_snapshot()
            released = all(g['memory_mib'] <= baseline[g['index']] for g in gpu['gpus'] if g['index'] in baseline)
            if not alive and not gpu['compute_apps'] and released:
                write_json(self.directory / 'shutdown.json',
                           {'known_processes': self.known_processes, 'gpus': gpu, 'released': True})
                break
            if time.monotonic() >= deadline:
                self.retain(f'Server tree or GPU memory not released within {timeout} s; no next start')
                raise RuntimeError('Normal shutdown incomplete; no further server is started')
            time.sleep(1)
        self.log.close()
        self.completed = True

    def track_processes(self):
        table = telemetry.process_table()
        # A child reparented after the root exits keeps its session id.
        owned = {p for p, r in table.items() if r['session'] == self.process.pid}
        if table.get(self.process.pid, {}).get('start') == self.identity:
            owned |= telemetry.descendants(table, self.process.pid)
        for p in owned:
            self.known_processes.setdefault(p, table[p]['start'])
        return table


SECRET = re.compile(r'KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|_PAT$|^SSH_', re.I)


def secret_name(name):
    return bool(SECRET.search(name))


# ------------------------------------------------------------------ startup evidence

# Lines vLLM 0.30.0 logs when each faster all-reduce path refuses to run on four PCIe-only
# GPUs without NVLink; TP4 must then fall back to NCCL through pynccl.
ALLREDUCE_DISABLED = {
    'custom_allreduce': r'Custom allreduce is disabled',
    'flashinfer_allreduce': r'FlashInfer All Reduce is disabled|Failed to initialize FlashInfer All Reduce workspace',
    'symmetric_memory': r'SymmMemCommunicator: .*(?:not available|not supported)',
}
PYNCCL_LINE = r'vLLM is using nccl=='
ATTENTION_LINE = r'Using (\S+) attention backend|Using (\S+) backend\.'


def allreduce_path(log, arm):
    lines = {name: [x for x in log.splitlines() if re.search(pattern, x)]
             for name, pattern in ALLREDUCE_DISABLED.items()}
    pynccl = [x for x in log.splitlines() if PYNCCL_LINE in x]
    enabled = [x for x in log.splitlines()
               if re.search(r'Initialized FlashInfer (?:All Reduce|PCIe IPC all-reduce|Allreduce)', x)]
    required = arm == 'TP4'
    passed = (not required) or (all(lines.values()) and bool(pynccl) and not enabled)
    return {'arm': arm, 'assertion_applies': required, 'pass': passed,
            'disabled_lines': lines, 'pynccl_lines': pynccl, 'enabled_lines': enabled}


COMPILE_SCRATCH = r'Compiling a graph for compile range|Compiling model again due to a load failure'
COMPILE_CACHED = (r'Directly load AOT compilation from path|Directly load the compiled graph'
                  r'|reconstructed serializable fn from standalone compile artifacts')


def start_record(log, metrics):
    """KV pool and compile evidence of one start; compile mode 'unknown' never counts as cached."""
    from .gates import capacity
    lines = log.splitlines()
    scratch = [x for x in lines if re.search(COMPILE_SCRATCH, x)]
    cached = [x for x in lines if re.search(COMPILE_CACHED, x)]
    return {'kv_capacity_tokens': capacity(metrics),
            'available_kv_cache_memory_lines': [x for x in lines if 'Available KV cache memory' in x],
            'torch_compile_lines': [x for x in lines if 'torch.compile took' in x],
            'compile_scratch_lines': scratch, 'compile_cached_lines': cached,
            'compile_mode': 'scratch' if scratch else 'cached' if cached else 'unknown'}


def start_problem(record, kind):
    """None, or why the start cannot be used.

    A timed start must have loaded the compile cache made by the earlier starts (a cold
    compile leaves a smaller KV pool). Any start that loaded the cache must have the KV
    capacity gate K needs; a cold start-test start only records its capacity."""
    if kind == 'timed' and record['compile_mode'] != 'cached':
        return f'Timed start compile mode is {record["compile_mode"]}, not a cached load'
    if (kind in ('start-test', 'settings', 'timed') and record['compile_mode'] == 'cached'
            and record['kv_capacity_tokens'] < CAPACITY):
        return f'KV capacity {record["kv_capacity_tokens"]} < {CAPACITY} tokens needed by gate K'
    return None


def capacity_note(record, settings_capacity):
    return {'settings_kv_capacity_tokens': settings_capacity, 'kv_capacity_tokens': record['kv_capacity_tokens'],
            'difference_tokens': None if settings_capacity is None
            else record['kv_capacity_tokens'] - settings_capacity}


def startup_fields(log, metrics, arm=None):
    """Resolved settings from a DEBUG-level start log and the cache-config gauge.
    Deductions are labelled; nothing is filled in from defaults."""
    from .gates import cache_config
    config = cache_config(metrics)
    fields = {}
    for name in ('gpu_memory_utilization', 'max_seq_len', 'enable_prefix_caching',
                 'async_scheduling', 'max_num_batched_tokens', 'max_num_seqs',
                 'max_concurrent_batches', 'num_gpu_blocks'):
        hits = re.findall(r'\b' + name + r'=([^,\s)]+)', log)
        fields[name] = hits[-1] if hits else None
    for name in ('max_num_seqs', 'max_num_batched_tokens'):
        matches = re.findall(r'Defaulting ' + name + r' to (\d+) for', log)
        if matches:
            fields[name] = matches[-1]
    batches = re.findall(r'Batch queue is enabled with size (\d+)', log)
    if batches:
        fields['max_concurrent_batches'] = batches[-1]
    for name in ('gpu_memory_utilization', 'enable_prefix_caching', 'num_gpu_blocks', 'block_size'):
        fields[name] = config.get(name)
    fields['model_runner'] = 'V2' if 'Using V2 Model Runner' in log else None
    fields['deductions'] = {}
    # vLLM 0.30.0 sets the batch queue to pp_size + 1 only with async scheduling, and a
    # PP4 queue of 5 also implies model runner V2.
    pp_size = PP_SIZE.get(arm)
    if pp_size and fields['max_concurrent_batches'] == str(pp_size + 1):
        fields['async_scheduling'] = 'True'
        fields['deductions']['async_scheduling'] = 'batch queue size = pipeline size + 1'
        if arm == 'PP4' and fields['model_runner'] is None:
            fields['model_runner'] = 'V2'
            fields['deductions']['model_runner'] = 'PP4 batch queue size 5'
    if fields['model_runner'] is None and 'gpu_model_runner.py:' in log:
        fields['model_runner'] = 'V1'
        fields['deductions']['model_runner'] = 'the V1 model runner module logged'
    fields['cuda_graph_log_lines'] = [x for x in log.splitlines() if 'cudagraph' in x.lower()]
    fields['cache_config_metric_lines'] = [x for x in metrics.splitlines()
                                           if x.startswith('vllm:cache_config_info')]
    fields['cache_config_scope'] = 'engine-wide, not per card'
    backend = re.findall(ATTENTION_LINE, log)
    fields['attention_backend'] = (backend[-1][0] or backend[-1][1]) if backend else None
    fields['attention_backend_lines'] = [x for x in log.splitlines() if re.search(ATTENTION_LINE, x)]
    fields['allreduce_path'] = allreduce_path(log, arm)
    fields['missing_fields'] = [k for k, v in fields.items() if v is None]
    if not fields['cuda_graph_log_lines']:
        fields['missing_fields'].append('cuda_graph_log_lines')
    return fields
