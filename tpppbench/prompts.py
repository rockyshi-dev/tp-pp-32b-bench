"""Prompt file: excerpts of the vLLM source tree rendered to exactly 1536 chat tokens.

Built offline on CPU from a clean vLLM checkout and the model's tokenizer. Each prompt's
text is an exact character slice of one source file; every prompt starts with a different
first text token and a different first 16-token block, so the prefix cache cannot help.
An explicit empty system message stops the chat template from inserting its default
system prompt, which would otherwise be a shared prefix of every request.
"""
import json
import os
from pathlib import Path
import subprocess

from .spec import (LEVELS, MODEL, MODEL_REVISION, PROMPT_TOKENS, VLLM_COMMIT, digest, write_json)

BANK_RETRIES = (0, 1)  # bank 0 is used first; bank 1 only for a level rerun after gate T or C


def total_prompts():
    return len(BANK_RETRIES) * sum(n + 2 * c for c, n in LEVELS.items())


def messages(text):
    return [{'role': 'system', 'content': ''}, {'role': 'user', 'content': text}]


def build(tokenizer, source, count):
    used, first_blocks, rows = set(), set(), []
    for file in sorted((source / 'vllm').rglob('*.py')):
        text = file.read_text()
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets, ids = enc['offset_mapping'], enc['input_ids']
        for start, token in enumerate(ids):
            if token in used or len(ids) - start < PROMPT_TOKENS:
                continue
            lo = offsets[start][0]
            for end in range(start + 1490, min(len(ids), start + 1540)):
                hi = offsets[end][0]
                excerpt = text[lo:hi]
                content_ids = tokenizer.encode(excerpt, add_special_tokens=False)
                if not content_ids or content_ids[0] in used:
                    break
                msg = messages(excerpt)
                rendered = tokenizer.apply_chat_template(msg, tokenize=True,
                                                         add_generation_prompt=True, return_dict=False)
                if len(rendered) == PROMPT_TOKENS:
                    first_block = tuple(rendered[:16])
                    if first_block in first_blocks:
                        break
                    rows.append({'id': len(rows), 'messages': msg,
                                 'prompt_tokens': PROMPT_TOKENS, 'first_text_token': content_ids[0],
                                 'token_ids': rendered, 'source_file': str(file.relative_to(source)),
                                 'source_sha256': digest(file), 'char_start': lo, 'char_end': hi})
                    used.add(content_ids[0])
                    first_blocks.add(first_block)
                    if len(rows) % 64 == 0:
                        print(json.dumps({'constructed': len(rows), 'target': count}), flush=True)
                    if len(rows) == count:
                        return rows
                    break
    raise RuntimeError(f'Only {len(rows)} prompts with distinct first text tokens; need {count}')


def banks():
    """Disjoint warm-up (2c) and measured sets per level, for bank 0 and bank 1."""
    result, pos = {}, 0
    for retry in BANK_RETRIES:
        for c, n in LEVELS.items():
            result[f'{c}:{retry}'] = {'warmup': list(range(pos, pos + 2 * c)),
                                      'measured': list(range(pos + 2 * c, pos + 2 * c + n))}
            pos += 2 * c + n
    return result


def check_source(source):
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != VLLM_COMMIT:
        raise RuntimeError(f'vLLM checkout is at {head}, not {VLLM_COMMIT}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'],
                               text=True).strip():
        raise RuntimeError('vLLM checkout has modified tracked files')


def tokenizer_files(model):
    return {p.name: digest(p) for p in sorted(Path(model).iterdir())
            if p.is_file() and p.suffix == '.json'}


def load_tokenizer(model):
    os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)


def construct(source, model, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    check_source(source)
    tokenizer = load_tokenizer(model)
    rows = build(tokenizer, Path(source), total_prompts())
    artifact = {'model': MODEL, 'model_revision': MODEL_REVISION, 'source_commit': VLLM_COMMIT,
                'system_message': 'explicit empty', 'banks': banks(), 'requests': rows,
                'tokenizer_files': tokenizer_files(model)}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, artifact)
    summary = {'output': str(output), 'sha256': digest(output), 'requests': len(rows),
               'distinct_first_text_tokens': len({r['first_text_token'] for r in rows}),
               'rendered_tokens': PROMPT_TOKENS}
    print(json.dumps(summary), flush=True)
    return summary


def load(path):
    """Read and check a prompt file; any structural problem raises."""
    bank = json.loads(Path(path).read_text())
    if (bank.get('model'), bank.get('model_revision'), bank.get('source_commit')) != (MODEL, MODEL_REVISION,
                                                                                      VLLM_COMMIT):
        raise RuntimeError('Prompt file was built for another model, revision or vLLM commit')
    rows = bank['requests']
    expected = total_prompts()
    if len(rows) != expected or len({r['first_text_token'] for r in rows}) != expected:
        raise RuntimeError('Prompt file lacks distinct first text tokens')
    if any(r['id'] != i or len(r['token_ids']) != PROMPT_TOKENS or r['prompt_tokens'] != PROMPT_TOKENS
           for i, r in enumerate(rows)):
        raise RuntimeError('Prompt token count or id mismatch')
    if len({tuple(r['token_ids'][:16]) for r in rows}) != expected:
        raise RuntimeError('Prompts share a first 16-token block')
    used = []
    for retry in BANK_RETRIES:
        for c, n in LEVELS.items():
            item = bank['banks'][f'{c}:{retry}']
            if len(item['warmup']) != 2 * c or len(item['measured']) != n:
                raise RuntimeError('Wrong bank size')
            used += item['warmup'] + item['measured']
    if sorted(used) != list(range(expected)):
        raise RuntimeError('Overlapping prompt banks')
    return bank


def verify(path, source, model):
    """Replay every stored excerpt and chat token list from the source and tokenizer."""
    tokenizer = load_tokenizer(model)
    data = load(path)
    source = Path(source)
    source_text, source_hash = {}, {}
    for row in data['requests']:
        relative = Path(row['source_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source path')
        name = row['source_file']
        if name not in source_text:
            source_text[name] = (source / name).read_text()
            source_hash[name] = digest(source / name)
        excerpt = source_text[name][row['char_start']:row['char_end']]
        if source_hash[name] != row['source_sha256'] or row['messages'] != messages(excerpt):
            raise ValueError(f'Source excerpt mismatch in prompt {row["id"]}')
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list) or len(ids) != PROMPT_TOKENS or ids != row['token_ids']:
            raise ValueError(f'Rendered token mismatch in prompt {row["id"]}')
        if tokenizer.encode(excerpt, add_special_tokens=False)[0] != row['first_text_token']:
            raise ValueError(f'First text token mismatch in prompt {row["id"]}')
    for name, sha in data['tokenizer_files'].items():
        if digest(Path(model) / name) != sha:
            raise ValueError(f'Tokenizer or config file differs from the one the prompts were built with: {name}')
    return {'prompts_sha256': digest(path), 'prompts': len(data['requests']), 'rendered_tokens': PROMPT_TOKENS,
            'distinct_first_text_tokens': len({r['first_text_token'] for r in data['requests']}),
            'distinct_first_16_token_blocks': len({tuple(r['token_ids'][:16]) for r in data['requests']}),
            'source_files': len(source_text), 'verified': True}
