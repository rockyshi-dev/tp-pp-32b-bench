import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from tpppbench import server, telemetry
from tpppbench.spec import CAPACITY, digest, write_json
from tests.helpers import cache_metric

CFG = SimpleNamespace(gpus=[0, 1, 2, 3], cache_dir=Path('/tmp/cc'), vllm=Path('/env/bin/vllm'),
                      model=Path('/models/qwen32'), port=8140)

INFO_LOG = '\n'.join([
    "(Worker_TP0 pid=1) WARNING custom_all_reduce.py:238] Custom allreduce is disabled because it's not "
    'supported on more than two PCIe-only GPUs.',
    '(Worker_TP0 pid=1) WARNING flashinfer_all_reduce.py:383] FlashInfer All Reduce is disabled because it '
    'is not supported for world_size=4.',
    '(Worker_TP0 pid=1) WARNING symm_mem.py:67] SymmMemCommunicator: Device capability 8.9 not supported, '
    'communicator is not available.',
    '(Worker_TP0 pid=1) INFO pynccl.py:148] vLLM is using nccl==2.29.7',
    "(Worker_TP0 pid=1) INFO cuda.py:538] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN'].",
    '(Worker_TP0 pid=1) INFO decorators.py:313] Directly load AOT compilation from path /cache/model',
    '(Worker_TP0 pid=1) INFO gpu_worker.py:641] Available KV cache memory: 5.12 GiB',
    '(Worker_TP0 pid=1) INFO monitor.py:20] torch.compile took 9.87 s in total'])

DEBUG_LOG = ("Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER'].\n"
             'Defaulting max_num_batched_tokens to 2048 for openai_api_server usage context.\n'
             'Defaulting max_num_seqs to 256 for openai_api_server usage context.\n'
             'Batch queue is enabled with size 5\nUsing V2 Model Runner\n'
             'max_seq_len=32768, compilation_config={"cudagraph_mode":"FULL_AND_PIECEWISE"}\n')


class Environment(unittest.TestCase):
    def test_only_the_parallelism_flag_differs(self):
        tp, pp = server.argv(CFG, 'TP4'), server.argv(CFG, 'PP4')
        diff = [(a, b) for a, b in zip(tp, pp) if a != b]
        self.assertEqual(diff, [('--tensor-parallel-size', '--pipeline-parallel-size')])
        self.assertEqual(len(tp), len(pp))

    def test_inherited_overrides_stop_the_start(self):
        for name in ('VLLM_USE_V1', 'NCCL_P2P_LEVEL', 'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS',
                     'TORCHINDUCTOR_MAX_AUTOTUNE', 'TRITON_CACHE_DIR', 'LD_PRELOAD', 'CUDA_LAUNCH_BLOCKING'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                server.environment(CFG, {}, 'timed', environ={name: '1', 'PATH': '/bin'})
        env = server.environment(CFG, {}, 'timed', environ={'TRITON_CACHE_DIR': '/tmp/cc/triton',
                                                           'CUDA_VISIBLE_DEVICES': '7', 'PATH': '/bin'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
        self.assertEqual(env['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
        self.assertNotIn('NCCL_DEBUG', env)
        self.assertNotIn('VLLM_LOGGING_LEVEL', env)

    def test_phase_specific_variables(self):
        env = server.environment(CFG, {}, 'transport', nccl_debug_dir='/tmp/t', environ={})
        self.assertEqual(env['NCCL_DEBUG'], 'INFO')
        self.assertEqual(env['NCCL_DEBUG_FILE'], '/tmp/t/nccl.%h.%p.log')
        self.assertEqual(server.environment(CFG, {}, 'settings', environ={})['VLLM_LOGGING_LEVEL'], 'DEBUG')
        with self.assertRaises(RuntimeError):
            server.environment(CFG, {'NCCL_P2P_LEVEL': 'SYS'}, 'timed', environ={})
        env = server.environment(CFG, {'NCCL_P2P_DISABLE': '1'}, 'timed', environ={'NCCL_P2P_DISABLE': '1'})
        self.assertEqual(env['NCCL_P2P_DISABLE'], '1')


class Deviations(unittest.TestCase):
    marker = 'vLLM is using nccl==2.29.7'

    def test_signatures(self):
        self.assertEqual(server.startup_signature('cuMemCreate failed\nFatal Python error: Segmentation fault',
                                                  1, False), 'NCCL_CUMEM_HOST_ENABLE')
        self.assertIsNone(server.startup_signature('Fatal Python error: Segmentation fault', 1, False))
        self.assertEqual(server.startup_signature(self.marker, None, True, 1800), 'NCCL_P2P_DISABLE')
        self.assertIsNone(server.startup_signature(self.marker, None, True, 1799))
        self.assertIsNone(server.startup_signature(self.marker + '\nLoading weights took 30 s', None, True, 1900))
        self.assertIsNone(server.startup_signature(self.marker, 1, False, 1900))

    def failed_start(self, tmp, log, deviations=None, kind='start-test', **failure):
        d = Path(tmp) / f'start-{len(list(Path(tmp).iterdir()))}'
        d.mkdir()
        (d / 'server.log').write_text(log)
        write_json(d / 'manifest.json', {'kind': kind, 'arm': 'TP4', 'deviations': deviations or {}})
        write_json(d / 'start-failure.json', {'returncode': None, 'timed_out': True, 'deviations': deviations or {},
                                              'nccl_observed_seconds': 1850, 'log_sha256': digest(d / 'server.log'),
                                              **failure})
        return d

    def test_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            hang = self.failed_start(tmp, self.marker)
            self.assertEqual(server.deviations_from_evidence([hang]), {'NCCL_P2P_DISABLE': '1'})
            (hang / 'server.log').write_text(self.marker + '\nchanged')
            with self.assertRaises(RuntimeError):
                server.deviations_from_evidence([hang])
            with self.assertRaises(RuntimeError):
                server.deviations_from_evidence([self.failed_start(tmp, self.marker, deviations={'NCCL_P2P_DISABLE': '1'})])
            with self.assertRaises(RuntimeError):
                server.deviations_from_evidence([self.failed_start(tmp, self.marker, kind='timed')])
            with self.assertRaises(RuntimeError):
                server.deviations_from_evidence([self.failed_start(tmp, 'other failure')])
        self.assertEqual(server.deviations_from_evidence([]), {})


class StartEvidence(unittest.TestCase):
    def test_allreduce_path(self):
        self.assertTrue(server.allreduce_path(INFO_LOG, 'TP4')['pass'])
        for line in INFO_LOG.splitlines()[:4]:
            self.assertFalse(server.allreduce_path(INFO_LOG.replace(line, ''), 'TP4')['pass'])
        self.assertTrue(server.allreduce_path('', 'PP4')['pass'])
        self.assertFalse(server.allreduce_path(INFO_LOG + '\nInitialized FlashInfer All Reduce', 'TP4')['pass'])

    def test_start_record_and_problem(self):
        record = server.start_record(INFO_LOG, cache_metric(blocks=CAPACITY // 16))
        self.assertEqual(record['compile_mode'], 'cached')
        self.assertIsNone(server.start_problem(record, 'timed'))
        small = server.start_record(INFO_LOG, cache_metric(blocks=CAPACITY // 16 - 1))
        self.assertIn('KV capacity', server.start_problem(small, 'timed'))
        cold = server.start_record('Compiling a graph for compile range (1, 2048)', cache_metric(blocks=100))
        self.assertIsNone(server.start_problem(cold, 'start-test'))
        self.assertIn('not a cached load', server.start_problem(cold, 'timed'))
        unknown = server.start_record('', cache_metric())
        self.assertIn('unknown', server.start_problem(unknown, 'timed'))
        self.assertEqual(server.capacity_note(record, 53000)['difference_tokens'], CAPACITY - 53000)

    def test_settings_fields(self):
        fields = server.startup_fields(DEBUG_LOG, cache_metric(blocks=3328), 'PP4')
        self.assertEqual(fields['missing_fields'], [])
        self.assertEqual(fields['async_scheduling'], 'True')
        self.assertEqual(fields['max_num_seqs'], '256')
        self.assertEqual(fields['max_concurrent_batches'], '5')
        self.assertEqual(fields['model_runner'], 'V2')
        self.assertEqual(fields['num_gpu_blocks'], '3328')
        self.assertEqual(fields['attention_backend'], 'FLASH_ATTN')
        self.assertIn('max_num_seqs', server.startup_fields('', cache_metric())['missing_fields'])
        tp = server.startup_fields(DEBUG_LOG.replace('size 5', 'size 2'), cache_metric(), 'TP4')
        self.assertIn('async_scheduling', tp['deductions'])


NODE_LSCPU = '# CPU,Core,Socket,Node\n' + '\n'.join(
    f'{cpu},{cpu % 52},{(cpu % 52) // 26},{(cpu % 52) // 26}' for cpu in range(104))
NODE_TOPO = ('\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n'
             + '\n'.join(f'GPU{g}\t' + '\t'.join(' X ' if h == g else ('PIX' if h // 4 == g // 4 else 'SYS')
                                                 for h in range(8))
                         + ('\t0-25,52-77\t0\t\tN/A' if g < 4 else '\t26-51,78-103\t1\t\tN/A') for g in range(8))
             + '\n\nLegend:\n\n  X    = Self\n')


class Topology(unittest.TestCase):
    def test_two_socket_layout(self):
        topo = telemetry.derive_topology(NODE_LSCPU, NODE_TOPO, [0, 1, 2, 3])
        self.assertEqual(topo['server_cpus'], list(range(26)) + list(range(52, 78)))
        self.assertEqual(topo['server_node'], 0)
        self.assertEqual(topo['client_cpus'], list(range(26, 34)))
        self.assertEqual(topo['runner_cpus'], [34, 35])
        other = telemetry.derive_topology(NODE_LSCPU, NODE_TOPO, [4, 5, 6, 7])
        self.assertEqual(other['server_node'], 1)
        self.assertEqual(other['client_cpus'], list(range(8)))
        with self.assertRaises(RuntimeError):
            telemetry.derive_topology(NODE_LSCPU, NODE_TOPO, [2, 3, 4, 5])

    def test_cpulist(self):
        self.assertEqual(telemetry.parse_cpulist('0-2,5,7-8'), [0, 1, 2, 5, 7, 8])


class Memory(unittest.TestCase):
    def test_anonymous_lower_bound(self):
        raw = ('1 default anon=100 dirty=100 N0=80 N1=20 kernelpagesize_kB=4\n'
               '2 default file=/lib/x.so mapped=5 N1=5 kernelpagesize_kB=4\n'
               '3 default anon=50 N0=50 kernelpagesize_kB=4\n')
        row = telemetry.anonymous_placement(raw, 0)
        self.assertEqual(row['anonymous_kib'], 600)
        self.assertAlmostEqual(row['node_fraction_lower_bound'], 520 / 600)
        self.assertAlmostEqual(telemetry.anonymous_placement(raw, 1)['node_fraction_lower_bound'], 80 / 600)
        self.assertIsNone(telemetry.anonymous_placement('', 0)['node_fraction_lower_bound'])

    def test_nccl_shm_pages(self):
        raw = ('1 default file=/dev/shm/nccl-AbC mapped=3 N0=2 N1=1\n'
               '2 default file=/dev/shm/other N1=9\n3 default file=/dev/shm/nccl-x N0=4\n')
        self.assertEqual(telemetry.nccl_shm_pages(raw), {'0': 6, '1': 1})


class Cards(unittest.TestCase):
    query = '\n'.join(f'{g}, GPU-{g}, 2520, 60.1, 40, 0, 1' for g in range(8))

    def test_parse_and_assert(self):
        snap = telemetry.parse_gpus(self.query, '10, GPU-0, vllm\n11, GPU-1, vllm\n12, GPU-2, vllm\n13, GPU-3, vllm\n')
        table = {1: {'ppid': 0}, **{p: {'ppid': 1} for p in (10, 11, 12, 13)}}
        cards = telemetry.assert_server_cards(1, snap, [0, 1, 2, 3], table)
        self.assertEqual(cards['observed_indices'], [0, 1, 2, 3])
        wrong = telemetry.parse_gpus(self.query, '10, GPU-4, vllm\n11, GPU-1, vllm\n12, GPU-2, vllm\n13, GPU-3, vllm\n')
        with self.assertRaises(RuntimeError):
            telemetry.assert_server_cards(1, wrong, [0, 1, 2, 3], table)
        foreign = telemetry.parse_gpus(self.query, '10, GPU-0, vllm\n99, GPU-7, other\n')
        with self.assertRaises(RuntimeError):
            telemetry.assert_server_cards(1, foreign, [0, 1, 2, 3], table)
        with self.assertRaises(RuntimeError):
            telemetry.parse_gpus('1, GPU-1, 1, 1, 1, 1, 1', '')


class Identity(unittest.TestCase):
    def proc(self, tmp, listener_inode='555', owner_inode='555'):
        root = Path(tmp)
        for pid, ppid in ((100, 1), (101, 100)):
            (root / str(pid) / 'fd').mkdir(parents=True)
            (root / str(pid) / 'stat').write_text(f'{pid} (x) S {ppid} {pid} 100 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 77 0\n')
        os.symlink(f'socket:[{owner_inode}]', root / '101' / 'fd' / '3')
        (root / 'net').mkdir()
        (root / 'net/tcp').write_text(
            '  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n'
            f'   0: 0100007F:1FCC 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 {listener_inode}\n')
        return root

    def test_owner_and_model(self):
        process = SimpleNamespace(pid=100, poll=lambda: None)
        models = {'data': [{'id': 'Qwen/Qwen2.5-32B-Instruct', 'root': '/models/qwen32'}]}
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.proc(tmp)
            result = server.assert_server_identity(process, 77, 8140, models, Path('/models/qwen32'), proc)
            self.assertEqual(result['listening_inodes'], ['555'])
            with self.assertRaises(RuntimeError):
                server.assert_server_identity(process, 78, 8140, models, Path('/models/qwen32'), proc)
            with self.assertRaises(RuntimeError):
                server.assert_server_identity(process, 77, 8140, models, Path('/models/other'), proc)
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.proc(tmp, owner_inode='999')
            with self.assertRaises(RuntimeError):
                server.assert_server_identity(process, 77, 8140, models, Path('/models/qwen32'), proc)


class LocksAndPlacement(unittest.TestCase):
    def test_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            locks = server.Locks(Path(tmp) / 'out', Path(tmp) / 'locks', [0, 1, 2, 3])
            with self.assertRaises(BlockingIOError):
                server.Locks(Path(tmp) / 'out2', Path(tmp) / 'locks', [3])
            locks.close()
            with self.assertRaises(FileExistsError):
                server.Locks(Path(tmp) / 'out', Path(tmp) / 'locks', [0])

    @unittest.skipUnless(os.uname().machine == 'x86_64', 'syscall number is x86_64')
    def test_placement_wrapper_execs_with_affinity_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / 'r.json'
            cpu = sorted(os.sched_getaffinity(0))[0]
            script = Path(server.__file__).with_name('placement.py')
            out = subprocess.run([sys.executable, str(script), '--cpus', str(cpu), '--node', '0', '--receipt',
                                  str(receipt), '--', sys.executable, '-c',
                                  'import os; print(sorted(os.sched_getaffinity(0)))'],
                                 capture_output=True, text=True, check=True)
            self.assertEqual(out.stdout.strip(), f'[{cpu}]')
            self.assertEqual(json.loads(receipt.read_text())['mempolicy'], 'MPOL_PREFERRED')


if __name__ == '__main__':
    unittest.main()
