"""Checks made before any server starts: card count, vllm executable, locks, model files,
and the credential filter of recorded environments."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tpppbench import gates, run, server, telemetry
from tpppbench.spec import REFERENCE_MODEL_FILES
from tests.helpers import BASE_NS, TOPOLOGY, samples_for
from tests.test_server import NODE_LSCPU, NODE_TOPO


def query(n):
    return '\n'.join(f'{g}, GPU-{g}, 2520, 60.1, 40, 0, 1' for g in range(n))


def fake_command(cards):
    def command(argv):
        if argv[0] == 'lscpu':
            return NODE_LSCPU
        if argv[:3] == ['nvidia-smi', 'topo', '-m']:
            return NODE_TOPO
        if argv[:2] == ['nvidia-smi', 'topo']:
            return 'p2p'
        if argv[1].startswith('--query-gpu'):
            return query(cards[0])
        return ''
    return command


class Cards(unittest.TestCase):
    def tearDown(self):
        telemetry.EXPECTED_CARDS = None

    def test_missing_card_at_start_refuses(self):
        with patch.object(telemetry, 'command', side_effect=fake_command([7])):
            with self.assertRaisesRegex(RuntimeError, 'shows 7 cards, expected 8'):
                telemetry.topology([0, 1, 2, 3], 8)
        with patch.object(telemetry, 'command', side_effect=fake_command([8])):
            self.assertEqual(telemetry.topology([0, 1, 2, 3], 8)['card_count'], 8)

    def test_card_lost_later_is_an_errored_sample_and_fails_gate_q(self):
        cards = [8]
        with patch.object(telemetry, 'command', side_effect=fake_command(cards)):
            telemetry.topology([0, 1, 2, 3], 8)
            cards[0] = 7
            with self.assertRaisesRegex(RuntimeError, 'expected 8'):
                telemetry.gpu_snapshot()
            monitor = telemetry.Monitor(os.getpid(), tempfile.gettempdir(), [0])
            row = monitor.boundary()
        self.assertIn('expected 8', row['error'])
        begin, end = BASE_NS, BASE_NS + 10 * 10**9
        samples = samples_for(begin, end)
        samples[5] = {'monotonic_ns': samples[5]['monotonic_ns'], 'error': row['error']}
        result = gates.quiet(samples, TOPOLOGY['server_cpus'], begin, end, TOPOLOGY['gpus'], 8)
        self.assertFalse(result['pass'])
        self.assertEqual(result['reasons'][0], gates.SAMPLE_ERROR)


class Vllm(unittest.TestCase):
    def script(self, tmp, interpreter):
        path = Path(tmp) / 'vllm'
        path.write_text(f'#!{interpreter}\nimport sys\n')
        path.chmod(0o755)
        return path

    @staticmethod
    def runner(stdout='0.30.0+cu129\n', code=0):
        return lambda argv, **kw: SimpleNamespace(returncode=code, stdout=stdout, stderr='')

    def test_same_interpreter_and_reported_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.script(tmp, sys.executable)
            info = server.check_vllm(path, {'vllm': '0.30.0+cu129'}, False, run=self.runner())
            self.assertTrue(info['same_interpreter'])
            self.assertEqual(info['vllm_reported_version'], '0.30.0+cu129')
            with self.assertRaisesRegex(RuntimeError, 'reported'):
                server.check_vllm(path, {'vllm': '0.30.0+cu129'}, False, run=self.runner('0.29.0\n'))
            with self.assertRaisesRegex(RuntimeError, 'reported'):
                server.check_vllm(path, {'vllm': '0.30.0+cu129'}, False, run=self.runner('', 1))

    def test_other_interpreter_or_missing_path_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = self.script(tmp, '/opt/other/bin/python3')
            with self.assertRaisesRegex(RuntimeError, 'not on this interpreter'):
                server.check_vllm(other, {'vllm': '0.30.0+cu129'}, False, run=self.runner())
            info = server.check_vllm(other, {'vllm': '0.30.0+cu129'}, True, run=self.runner('x\n'))
            self.assertFalse(info['same_interpreter'])
            with self.assertRaisesRegex(RuntimeError, 'not an executable file'):
                server.check_vllm(Path(tmp) / 'nope', {}, False)

    def test_wrong_vllm_path_exits_before_any_phase(self):
        """An operator path error stops the command with an error, and no phase directory
        or decision file is written."""
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            study = Path(tmp) / 'study'
            argv = ['timed', '--study', str(study), '--model', str(model), '--prompts', str(Path(tmp) / 'p.json'),
                    '--vllm', str(Path(tmp) / 'missing-vllm'), '--no-setsid-check']
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
            env.pop('TPPP_MODEL', None)
            out = subprocess.run([sys.executable, '-c', 'import sys; from tpppbench.__main__ import main; '
                                  'sys.exit(main(sys.argv[1:]))', *argv], capture_output=True, text=True, env=env)
            self.assertNotEqual(out.returncode, 0)
            self.assertIn('error: --vllm', out.stderr)
            self.assertIn('not an executable file', out.stderr)
            self.assertFalse((study / 'timed').exists())


class Locks(unittest.TestCase):
    def test_extra_locks_expand_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = server.lock_paths(tmp / 'out', tmp / 'locks', [0, 1, 2, 3],
                                      [tmp / 'shared/gpu{gpu}.lock', tmp / 'locks/gpu2.lock', tmp / 'one.lock',
                                       tmp / 'one.lock'])
            names = [str(p.relative_to(tmp)) for p in paths]
            self.assertEqual(names, ['out/run.lock', 'locks/gpu0.lock', 'locks/gpu1.lock', 'locks/gpu2.lock',
                                     'locks/gpu3.lock', 'shared/gpu0.lock', 'shared/gpu1.lock', 'shared/gpu2.lock',
                                     'shared/gpu3.lock', 'one.lock'])
            locks = server.Locks(tmp / 'out', tmp / 'locks', [0, 1, 2, 3], [tmp / 'shared/gpu{gpu}.lock'])
            with self.assertRaises(BlockingIOError):
                server.Locks(tmp / 'out2', tmp / 'other-locks', [5], [tmp / 'shared/gpu1.lock'])
            locks.close()


class ModelFiles(unittest.TestCase):
    def test_reference_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in REFERENCE_MODEL_FILES:
                (Path(tmp) / name).write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'allow-other-model'):
                run.model_check(tmp, False)
            info = run.model_check(tmp, True)
            self.assertFalse(info['reference_model_files'])
            self.assertEqual(info['differing_model_files'], sorted(REFERENCE_MODEL_FILES))

    def test_matching_files_pass(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(run, 'digest', side_effect=lambda p: REFERENCE_MODEL_FILES[Path(p).name]):
            for name in REFERENCE_MODEL_FILES:
                (Path(tmp) / name).write_text('{}')
            self.assertTrue(run.model_check(tmp, False)['reference_model_files'])


class Secrets(unittest.TestCase):
    def test_credential_names_are_not_recorded(self):
        for name in ('HF_TOKEN', 'hf_token', 'AWS_SECRET_ACCESS_KEY', 'OPENROUTER_API_KEY', 'DB_PASSWORD',
                     'FTP_PASSWD', 'GIT_CREDENTIALS', 'PROXY_AUTH', 'SESSION_COOKIE', 'GITHUB_PAT',
                     'SSH_AUTH_SOCK', 'ssh_agent_pid'):
            self.assertTrue(server.secret_name(name), name)
        for name in ('PATH', 'HOME', 'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'VLLM_CACHE_ROOT'):
            self.assertFalse(server.secret_name(name), name)


if __name__ == '__main__':
    unittest.main()
