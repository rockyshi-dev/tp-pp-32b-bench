"""The phase driver with fake servers, clients and helpers: order, retries, stops."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tpppbench import gates, hostmem, run, server
from tpppbench.spec import CAPACITY, LEVELS, write_json
from tests.helpers import TOPOLOGY, cache_metric, request
from tests.test_server import INFO_LOG


def cfg(tmp, **kw):
    base = dict(study=Path(tmp), topology=dict(TOPOLOGY), port=8140, touch_bytes=40 * 10**9,
                min_usable_free=20 * 10**9, gpus=[0, 1, 2, 3], model=Path('/m'), cache_dir=Path('/cc'))
    base.update(kw)
    return SimpleNamespace(**base)


class FakeServer:
    log = []

    def __init__(self, cfg, arm, directory, locks, deviations, kind):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        (self.directory / 'server.log').write_text(INFO_LOG)
        self.url = 'http://fake'
        self.completed = False
        self.process = SimpleNamespace(pid=1)
        FakeServer.log.append(('start', arm, self.directory.name, kind))

    def ready(self):
        pass

    def retain(self, reason):
        FakeServer.log.append(('retain', reason))

    def close_completed(self, timeout=60):
        self.completed = True
        FakeServer.log.append(('close', self.directory.name))


def cell(**kw):
    base = {'K': True, 'T': True, 'C': True, 'G': True, 'N': True, 'Q': True, 'quiet': {'pass': True},
            'ttft_s': 1.0, 'token_gap_s': 0.05}
    base.update(kw)
    base['rerun_level'] = not (base['T'] and base['C'])
    return base


class Driver(unittest.TestCase):
    def setUp(self):
        FakeServer.log = []
        self.calls = []
        self.patches = [patch.object(server, 'Server', FakeServer),
                        patch.object(server, 'preflight', side_effect=lambda port: self.calls.append('preflight')),
                        patch.object(run, 'host_memory_step', side_effect=lambda c, o: self.calls.append(('helper', o.name))),
                        patch.object(run, 'host_memory_ready', return_value={}),
                        patch.object(run, 'metrics', side_effect=self.metrics)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def metrics(self, srv, directory, name):
        text = cache_metric(blocks=CAPACITY // 16 + 10)
        (directory / name).write_text(text)
        return text

    def test_timed_alternates_order_and_reruns_levels_on_bank_1(self):
        script = {}

        def measured(c, srv, bank, level, retry, directory):
            arm = srv.directory.name.split('-')[0]
            round_dir = srv.directory.parent.name
            self.calls.append(('level', round_dir, srv.directory.name, level, retry))
            if round_dir == 'round-2-attempt-0' and arm == 'PP4' and level == 8 and retry == 0:
                return cell(T=False)
            return script.get((round_dir, arm, level), cell())
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'measured_level', side_effect=measured):
            c = cfg(tmp)
            out = Path(tmp) / 'timed'
            out.mkdir()
            rounds, terminal = run.timed(c, out, None, {}, {}, {'TP4': CAPACITY, 'PP4': CAPACITY})
        self.assertIsNone(terminal)
        self.assertEqual(len(rounds), 5)
        starts = [(e[2]) for e in FakeServer.log if e[0] == 'start']
        self.assertEqual(starts[:2], ['TP4', 'PP4'])
        self.assertEqual(starts[2:6], ['PP4', 'TP4', 'PP4-level-retry', 'TP4-level-retry'])
        self.assertEqual(starts[6:8], ['TP4', 'PP4'])
        retried = [x for x in self.calls if isinstance(x, tuple) and x[0] == 'level' and x[4] == 1]
        self.assertEqual(sorted(x[2] for x in retried), ['PP4-level-retry', 'TP4-level-retry'])
        self.assertTrue(all(x[3] == 8 for x in retried))
        self.assertTrue(rounds[1]['levels']['8']['PP4']['T'])
        # A helper call precedes every server start.
        helpers = [x for x in self.calls if isinstance(x, tuple) and x[0] == 'helper']
        self.assertEqual(len(helpers), len(starts))

    def test_quiet_failures_rerun_then_stop(self):
        def measured(c, srv, bank, level, retry, directory):
            noisy = srv.directory.parent.name.startswith('round-2') and level == 1
            return cell(Q=not noisy, quiet={'pass': not noisy, 'reasons': ['foreign CPU time'] if noisy else []})
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'measured_level', side_effect=measured):
            out = Path(tmp) / 'timed'
            out.mkdir()
            rounds, terminal = run.timed(cfg(tmp), out, None, {}, {}, None)
            self.assertEqual(terminal, 'Q')
            self.assertEqual(len(rounds), 1)
            self.assertTrue((out / 'round-2-attempt-2' / 'retry-action.json').exists())
            self.assertFalse((out / 'round-2-attempt-3').exists())

    def test_cold_timed_start_is_shut_down_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(FakeServer, 'ready', lambda self: (self.directory / 'server.log').write_text(
                    INFO_LOG.replace('Directly load AOT compilation from path', 'Compiling a graph for compile range'))):
            with self.assertRaisesRegex(gates.EvidenceError, 'not a cached load'):
                run.arm(cfg(tmp), 'timed', 'TP4', Path(tmp) / 'TP4', None, {}, {}, LEVELS)
        self.assertIn(('close', 'TP4'), FakeServer.log)

    def test_start_test_rehearsal_and_request(self):
        seen = []

        def client_phase(url, prompts, c, cpus, directory):
            seen.append((len(prompts), c))
            return [request(p['id']) for p in prompts]

        def rehearsal(c, srv, bank, output, initial):
            seen.append('rehearsal')
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.client, 'phase', side_effect=client_phase), \
                patch.object(run, 'rehearsal', side_effect=rehearsal):
            from tpppbench.client import summarize
            with patch.object(run.client, 'phase', side_effect=lambda *a: [summarize(request(p['id'])) for p in a[1]]):
                run.arm(cfg(tmp), 'start-test', 'PP4', Path(tmp) / 'PP4', None, {},
                        {'requests': [{'id': i} for i in range(8)]}, [])
            self.assertTrue((Path(tmp) / 'PP4' / 'request-check.json').exists())
        self.assertIn('rehearsal', seen)

    def test_failed_start_is_retained_not_killed(self):
        def boom(self):
            raise server.StartFailure('no answer')
        with tempfile.TemporaryDirectory() as tmp, patch.object(FakeServer, 'ready', boom):
            with self.assertRaises(server.StartFailure):
                run.arm(cfg(tmp), 'timed', 'TP4', Path(tmp) / 'TP4', None, {}, {}, LEVELS)
        self.assertEqual(FakeServer.log[-1][0], 'retain')
        self.assertNotIn('close', [e[0] for e in FakeServer.log])


class HostMemory(unittest.TestCase):
    def reading(self, usable):
        return {'usable_free_bytes': usable}

    def run_step(self, results, touch=40 * 10**9):
        calls = []

        def fake(nodes, directory, owned, cpus_of=None, nbytes=0, required=0):
            calls.append((nodes, nbytes, required, cpus_of))
            item = results.pop(0)
            if isinstance(item, Exception):
                raise item
            return {0: {'pid': 5, 'seconds': 1.0, 'before': {}, 'after': self.reading(item)}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(hostmem, 'run', side_effect=fake), \
                patch.object(hostmem, 'snapshot', return_value={'usable_free_bytes': 1}):
            c = cfg(tmp, touch_bytes=touch)
            try:
                return run.host_memory_step(c, Path(tmp) / 'TP4'), calls
            except run.HostMemoryStop as exc:
                return exc, calls

    def test_met_first_time_with_declared_bytes(self):
        record, calls = self.run_step([20 * 10**9])
        self.assertEqual(record['precondition'], 'met')
        self.assertEqual(calls[0][1:3], (40 * 10**9, 44 * 10**9))
        self.assertEqual(calls[0][3], {0: TOPOLOGY['server_cpus']})

    def test_repeat_once_then_stop(self):
        record, calls = self.run_step([10 * 10**9, 25 * 10**9])
        self.assertEqual(len(calls), 2)
        self.assertEqual(record['precondition'], 'met')
        stop, calls = self.run_step([10 * 10**9, 19 * 10**9])
        self.assertIsInstance(stop, run.HostMemoryStop)

    def test_refusal_stops_without_repeat(self):
        stop, calls = self.run_step([hostmem.Refusal('low', {})])
        self.assertIsInstance(stop, run.HostMemoryStop)
        self.assertEqual(len(calls), 1)

    def test_off_records_only(self):
        record, calls = self.run_step([], touch=0)
        self.assertTrue(record['skipped'])
        self.assertEqual(calls, [])


class Prerequisites(unittest.TestCase):
    def test_timed_needs_passing_earlier_phases_with_same_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = cfg(tmp)
            with self.assertRaisesRegex(RuntimeError, 'start-test'):
                run.check_prerequisites(c, 'timed', {}, 'sha')
            for phase in ('start-test', 'settings', 'transport'):
                (Path(tmp) / phase).mkdir()
                write_json(Path(tmp) / phase / 'result.json', {'pass': True, 'deviations': {}, 'prompts_sha256': 'sha',
                                                               'gpus': [0, 1, 2, 3], 'model_path': '/m',
                                                               'cache_dir': '/cc'})
            run.check_prerequisites(c, 'timed', {}, 'sha')
            with self.assertRaisesRegex(RuntimeError, 'compile cache'):
                run.check_prerequisites(cfg(tmp, cache_dir=Path('/other')), 'timed', {}, 'sha')
            with self.assertRaisesRegex(RuntimeError, 'deviations or prompts'):
                run.check_prerequisites(c, 'timed', {'NCCL_P2P_DISABLE': '1'}, 'sha')
            write_json(Path(tmp) / 'transport' / 'result.json', {'pass': False})
            with self.assertRaisesRegex(RuntimeError, 'did not pass'):
                run.check_prerequisites(c, 'timed', {}, 'sha')
            run.check_prerequisites(c, 'settings', {}, 'sha')


if __name__ == '__main__':
    unittest.main()
