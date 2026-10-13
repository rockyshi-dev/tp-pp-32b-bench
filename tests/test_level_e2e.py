"""One measured level end to end against a fake streaming server on localhost: real client
processes, real /metrics scrapes, real numa_maps reads and the real 1 Hz monitor (only
nvidia-smi is replaced), then the same gate evaluation the runner and `report` use."""
import asyncio
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tpppbench import run, telemetry
from tpppbench.spec import LEVELS, read_json
from tests.helpers import cache_metric

try:
    from aiohttp import web
except ImportError:  # pragma: no cover
    web = None


class FakeVllm:
    def __init__(self):
        self.queries = 0
        self.loop = asyncio.new_event_loop()
        self.started = threading.Event()
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]
        self.thread = threading.Thread(target=self.serve, daemon=True)

    async def chat(self, request):
        body = await request.json()
        assert body['stream'] and body['max_tokens'] == 128 and body['min_tokens'] == 128
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)

        async def send(obj):
            text = obj if isinstance(obj, str) else json.dumps(obj)
            await response.write(f'data: {text}\n\n'.encode())
        await send({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})
        await asyncio.sleep(0.01)
        for k in range(128):
            await send({'choices': [{'index': 0, 'delta': {'content': f't{k}'},
                                     'finish_reason': 'length' if k == 127 else None}]})
            await asyncio.sleep(0.001)
        self.queries += 1536
        await send({'choices': [], 'usage': {'prompt_tokens': 1536, 'completion_tokens': 128, 'total_tokens': 1664}})
        await send('[DONE]')
        await response.write_eof()
        return response

    async def metrics(self, request):
        text = (cache_metric() +
                f'vllm:prefix_cache_queries_total{{engine="0"}} {float(self.queries)}\n'
                'vllm:prefix_cache_hits_total{engine="0"} 0.0\n'
                'vllm:num_preemptions_total{engine="0"} 0.0\n'
                'vllm:num_requests_waiting{engine="0"} 0.0\n')
        return web.Response(text=text)

    def serve(self):
        asyncio.set_event_loop(self.loop)
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self.chat)
        app.router.add_get('/metrics', self.metrics)
        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        self.loop.run_until_complete(web.TCPSite(self.runner, '127.0.0.1', self.port).start())
        self.started.set()
        self.loop.run_forever()

    def __enter__(self):
        self.thread.start()
        self.started.wait(10)
        return self

    def __exit__(self, *unused):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


def fake_cards():
    return {'gpus': [{'index': g, 'uuid': f'GPU-{g}', 'clock_mhz': 2500.0, 'power_w': 100.0, 'temperature_c': 50.0,
                      'utilization_pct': 50.0, 'memory_mib': 1000.0} for g in range(8)], 'compute_apps': []}


@unittest.skipUnless(web is not None and len(os.sched_getaffinity(0)) >= 2, 'needs aiohttp and two CPUs')
class MeasuredLevel(unittest.TestCase):
    def test_c1_level_through_the_timed_code_path(self):
        cpus = sorted(os.sched_getaffinity(0))
        client = (cpus * 8)[:8]
        topology = {'gpus': [0, 1, 2, 3], 'server_node': 0, 'server_cpus': cpus, 'client_cpus': client,
                    'runner_cpus': cpus[-1:], 'card_count': 8}
        n = LEVELS[1]
        bank = {'requests': [{'id': i, 'messages': [{'role': 'system', 'content': ''},
                                                    {'role': 'user', 'content': f'prompt {i}'}]}
                             for i in range(n + 2)],
                'banks': {'1:0': {'warmup': [0, 1], 'measured': list(range(2, n + 2))}}}
        with tempfile.TemporaryDirectory() as tmp, FakeVllm() as fake, \
                patch.object(telemetry, 'gpu_snapshot', side_effect=fake_cards):
            arm_dir = Path(tmp) / 'TP4'
            arm_dir.mkdir()
            (arm_dir / 'metrics-startup.txt').write_text(cache_metric())
            srv = SimpleNamespace(url=f'http://127.0.0.1:{fake.port}', process=SimpleNamespace(pid=os.getpid()),
                                  directory=arm_dir)
            cfg = SimpleNamespace(topology=topology)
            result = run.measured_level(cfg, srv, bank, 1, 0, arm_dir / 'c1')
            stored = read_json(arm_dir / 'c1' / 'gate-calculation.json')
            rows = [json.loads(x) for x in (arm_dir / 'c1' / 'measured' / 'client-0.jsonl').read_text().splitlines()]
            samples = read_json(arm_dir / 'c1' / 'telemetry-with-boundaries.json')
        self.assertEqual(len(rows), n)
        self.assertTrue(all(r['content_chunks'] == 128 and r['usage']['completion_tokens'] == 128 for r in rows))
        for gate in ('K', 'T', 'C', 'G'):
            self.assertTrue(result[gate], gate)
        self.assertEqual(result['prefix_queries'], n * 1536)
        self.assertGreater(result['ttft_s'], 0.005)
        self.assertGreater(result['token_gap_s'], 0)
        self.assertIsNotNone(result['quiet'].get('foreign_cpu_fraction'))
        self.assertGreaterEqual(len(samples), 3)
        # No NCCL /dev/shm staging pages exist in this process tree, so gate N cannot pass.
        self.assertEqual(result['nccl_shm_placement']['total_pages'], 0)
        self.assertFalse(result['N'])
        self.assertEqual(stored['ttft_s'], result['ttft_s'])


if __name__ == '__main__':
    unittest.main()
