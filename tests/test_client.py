import asyncio
import json
import unittest

from tpppbench import client


class Lines:
    """An async iterator of raw SSE lines, like aiohttp's response.content."""
    def __init__(self, lines):
        self.lines = [x.encode() for x in lines]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.lines:
            raise StopAsyncIteration
        return self.lines.pop(0)


def sse(*events):
    out = []
    for e in events:
        out += [f'data: {e if isinstance(e, str) else json.dumps(e)}\n', '\n']
    return out


def chunk(content=None, finish=None, role=None):
    delta = {}
    if role:
        delta['role'] = role
    if content is not None:
        delta['content'] = content
    return {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


class Stream(unittest.TestCase):
    def read(self, lines):
        row = client.new_row({'id': 7})
        asyncio.run(client.read_stream(Lines(lines), row))
        return client.summarize(row)

    def test_role_empty_and_usage_chunks_are_not_tokens(self):
        usage = {'choices': [], 'usage': {'prompt_tokens': 1536, 'completion_tokens': 3}}
        row = self.read(sse(chunk(role='assistant', content=''), chunk('a'), chunk(''), chunk('b'),
                            chunk('c', finish='length'), usage, '[DONE]'))
        self.assertEqual(row['contents'], ['a', 'b', 'c'])
        self.assertEqual(row['content_chunks'], 3)
        self.assertEqual(row['finish_reason'], 'length')
        self.assertEqual(row['usage']['completion_tokens'], 3)
        self.assertIsNone(row['token_gap_s'])  # not 128 chunks
        self.assertIsNotNone(row['mean_tpot_s'])

    def test_coalesced_content_counts_as_one_chunk(self):
        usage = {'choices': [], 'usage': {'prompt_tokens': 1536, 'completion_tokens': 128}}
        row = self.read(sse(*(chunk('tok') for _ in range(126)), chunk('two tokens'), usage, '[DONE]'))
        self.assertEqual(row['content_chunks'], 127)
        self.assertIsNone(row['token_gap_s'])

    def test_incomplete_or_error_stream_raises(self):
        with self.assertRaises(RuntimeError):
            self.read(sse(chunk('a')))
        with self.assertRaises(RuntimeError):
            self.read(sse({'error': {'message': 'boom'}}))
        with self.assertRaises(RuntimeError):
            self.read(sse({'choices': [{'index': 1, 'delta': {'content': 'x'}}]}))

    def test_multiline_data_event(self):
        row = client.new_row({'id': 1})
        lines = ['data: {"choices": [{"index": 0,\n', 'data:  "delta": {"content": "z"}}]}\n', '\n',
                 'data: {"choices": [], "usage": {"completion_tokens": 1}}\n', '\n', 'data: [DONE]\n', '\n']
        asyncio.run(client.read_stream(Lines(lines), row))
        self.assertEqual(row['contents'], ['z'])


class Summaries(unittest.TestCase):
    def test_gap_is_the_median_and_tpot_the_mean(self):
        stamps = [10**9 + k * 10**7 for k in range(128)]
        stamps[64] += 5 * 10**8  # one stall moves the mean, not the median
        row = client.summarize({'send_ns': 0, 'content_ns': stamps, 'usage': {'completion_tokens': 128}})
        self.assertAlmostEqual(row['ttft_s'], 1.0)
        self.assertAlmostEqual(row['token_gap_s'], 0.01)
        self.assertAlmostEqual(row['mean_tpot_s'], (stamps[-1] - stamps[0]) / 1e9 / 127)

    def test_payload(self):
        body = client.payload({'messages': [{'role': 'user', 'content': 'x'}]})
        self.assertEqual((body['max_tokens'], body['min_tokens'], body['temperature']), (128, 128, 0))
        self.assertTrue(body['stream'])
        self.assertEqual(body['stream_options'], {'include_usage': True})
        self.assertEqual(body['model'], 'Qwen/Qwen2.5-32B-Instruct')


class Plan(unittest.TestCase):
    def test_one_process_per_eight_streams(self):
        prompts = [{'id': i} for i in range(64)]
        cpus = list(range(26, 34))
        for c, processes, share in ((1, 1, 8), (8, 1, 8), (32, 4, 2)):
            work = client.plan(prompts, c, cpus)
            self.assertEqual(len(work), processes)
            self.assertTrue(all(len(a) == share for _, a in work))
            streams = [s for ss, _ in work for s in ss]
            self.assertEqual(len(streams), c)
            self.assertEqual(streams[0], prompts[0::c])
        assigned = [a for _, a in client.plan(prompts, 32, cpus)]
        self.assertEqual(sum(assigned, []), cpus)

    def test_rejects_wrong_shapes(self):
        with self.assertRaises(ValueError):
            client.plan([{'id': 0}] * 10, 8, list(range(8)))
        with self.assertRaises(ValueError):
            client.plan([{'id': 0}] * 16, 16, list(range(8)))
        with self.assertRaises(ValueError):
            client.plan([{'id': 0}] * 8, 8, list(range(4)))


if __name__ == '__main__':
    unittest.main()
