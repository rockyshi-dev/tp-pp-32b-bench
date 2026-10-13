"""Closed-loop streaming client: one process per eight streams; never imports vLLM."""
import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

from .spec import LEVELS, MODEL, OUTPUT_TOKENS, STREAMS_PER_CLIENT


def payload(row):
    return {'model': MODEL, 'messages': row['messages'], 'max_tokens': OUTPUT_TOKENS,
            'min_tokens': OUTPUT_TOKENS, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}}


def summarize(row):
    """Derived per-request values, recomputable from the raw timestamps at any time.

    TTFT: send to the first non-empty content chunk. Token gap: median of the gaps between
    consecutive content chunks, only when the request arrived as exactly 128 chunks (one
    token each). Mean TPOT: (last - first chunk) / (completion tokens - 1)."""
    stamps = row['content_ns']
    row['content_chunks'] = len(stamps)
    row['ttft_s'] = (stamps[0] - row['send_ns']) / 1e9 if stamps else None
    row['token_gap_s'] = (statistics.median(b - a for a, b in zip(stamps, stamps[1:])) / 1e9
                          if len(stamps) == OUTPUT_TOKENS else None)
    n = (row.get('usage') or {}).get('completion_tokens', 0)
    row['mean_tpot_s'] = ((stamps[-1] - stamps[0]) / 1e9 / (n - 1)
                          if len(stamps) > 1 and n > 1 else None)
    return row


def parse_event(row, event, stamp):
    """Apply one server-sent event to a request record. Returns True on [DONE].

    Only events with non-empty `delta.content` count as content chunks: the role chunk,
    empty chunks and the usage chunk are not tokens."""
    if event == '[DONE]':
        row['done'] = True
        return True
    chunk = json.loads(event)
    if chunk.get('error'):
        raise RuntimeError(str(chunk['error']))
    if chunk.get('usage') is not None:
        row['usage'] = chunk['usage']
    choices = chunk.get('choices', [])
    if choices:
        if len(choices) != 1 or choices[0].get('index') != 0:
            raise RuntimeError('Unexpected choices')
        choice = choices[0]
        content = (choice.get('delta') or {}).get('content')
        if content:
            row['content_ns'].append(stamp)
            row['contents'].append(content)
        if choice.get('finish_reason') is not None:
            row['finish_reason'] = choice['finish_reason']
    return False


async def read_stream(lines, row):
    data = []
    async for line in lines:
        line = line.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif line == '' and data:
            stamp = time.monotonic_ns()
            event = '\n'.join(data)
            data = []
            if parse_event(row, event, stamp):
                break
    if not row['done'] or row['usage'] is None or not row['content_ns']:
        raise RuntimeError('Incomplete stream: missing [DONE], usage, or text')


def new_row(prompt):
    return {'id': prompt['id'], 'send_ns': time.monotonic_ns(), 'content_ns': [],
            'contents': [], 'usage': None, 'finish_reason': None, 'done': False, 'error': None}


async def request(session, url, prompt):
    row = new_row(prompt)
    try:
        async with session.post(url + '/v1/chat/completions', json=payload(prompt)) as response:
            row['status'] = response.status
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:1000]}')
            await read_stream(response.content, row)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['end_ns'] = time.monotonic_ns()
    return summarize(row)


def worker(url, streams, cpus, barrier, output):
    os.sched_setaffinity(0, cpus)

    async def run():
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=1800)
        connector = aiohttp.TCPConnector(limit=STREAMS_PER_CLIENT, keepalive_timeout=1.0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            barrier.wait(timeout=90)
            with open(output, 'x') as dest:
                async def loop(items):
                    for row in items:
                        result = await request(session, url, row)
                        dest.write(json.dumps(result, allow_nan=False) + '\n')
                        dest.flush()
                await asyncio.gather(*(loop(items) for items in streams))
    asyncio.run(run())


def plan(prompts, concurrency, cpus):
    """[(streams, cpus)] per client process. Stream i sends prompts[i::c] in order;
    one process per eight streams, the eight client CPUs split evenly among them."""
    if concurrency not in LEVELS or len(cpus) != 8 or len(prompts) % concurrency:
        raise ValueError('Wrong concurrency, client CPU count, or unbalanced closed loop')
    count = max(1, concurrency // STREAMS_PER_CLIENT)
    share = len(cpus) // count
    return [([prompts[i::concurrency] for i in range(k * STREAMS_PER_CLIENT,
                                                       min((k + 1) * STREAMS_PER_CLIENT, concurrency))],
             list(cpus[k * share:(k + 1) * share])) for k in range(count)]


def read_rows(directory):
    return [json.loads(line) for p in sorted(Path(directory).glob('client-*.jsonl'))
            for line in p.read_text().splitlines() if line.strip()]


def phase(url, prompts, concurrency, cpus, directory):
    """Run one closed loop. Fresh client processes synchronise on a barrier, then each
    stream sends its next request as soon as the previous one finishes."""
    work = plan(prompts, concurrency, cpus)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(len(work))
    processes = []
    try:
        for index, (streams, assigned) in enumerate(work):
            proc = ctx.Process(target=worker, args=(url, streams, assigned, barrier,
                                                   directory / f'client-{index}.jsonl'))
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()
        if any(proc.exitcode != 0 for proc in processes):
            raise RuntimeError('Client process failed; the level has incomplete evidence')
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()
    rows = read_rows(directory)
    if sorted(r['id'] for r in rows) != sorted(r['id'] for r in prompts):
        raise RuntimeError('Missing or duplicate requests')
    return rows
