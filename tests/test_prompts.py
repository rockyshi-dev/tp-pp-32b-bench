import os
from pathlib import Path
import tempfile
import unittest

from tpppbench import prompts
from tpppbench.spec import LEVELS, MODEL, MODEL_REVISION, REFERENCE_PROMPTS_SHA256, VLLM_COMMIT, digest, write_json


class CharTokenizer:
    """One token per character; the chat template adds 10 fixed tokens."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_dict=False):
        assert messages[0] == {'role': 'system', 'content': ''}
        return [1] * 5 + self.encode(messages[1]['content']) + [2] * 5


class Build(unittest.TestCase):
    def test_excerpts_have_distinct_first_tokens_and_exact_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / 'vllm').mkdir()
            # Characters chr(256)..chr(4000) give each start position a distinct first token.
            (source / 'vllm' / 'a.py').write_text(''.join(chr(256 + i) for i in range(4000)))
            rows = prompts.build(CharTokenizer(), source, 20)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r['first_text_token'] for r in rows}), 20)
        self.assertTrue(all(len(r['token_ids']) == 1536 for r in rows))
        self.assertTrue(all(r['char_end'] - r['char_start'] == 1526 for r in rows))
        self.assertEqual(rows[0]['messages'][0], {'role': 'system', 'content': ''})

    def test_too_little_text_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'vllm').mkdir()
            (Path(tmp) / 'vllm' / 'a.py').write_text('x' * 100)
            with self.assertRaises(RuntimeError):
                prompts.build(CharTokenizer(), Path(tmp), 1)


def synthetic_file(path):
    n = prompts.total_prompts()
    rows = [{'id': i, 'messages': prompts.messages(str(i)), 'prompt_tokens': 1536, 'first_text_token': 10_000 + i,
             'token_ids': [10_000 + i] + [0] * 1535, 'source_file': 'vllm/a.py', 'source_sha256': 'x',
             'char_start': 0, 'char_end': 1} for i in range(n)]
    data = {'model': MODEL, 'model_revision': MODEL_REVISION, 'source_commit': VLLM_COMMIT,
            'system_message': 'explicit empty', 'banks': prompts.banks(), 'requests': rows, 'tokenizer_files': {}}
    write_json(path, data)
    return data


class Load(unittest.TestCase):
    def test_bank_layout(self):
        banks = prompts.banks()
        self.assertEqual(prompts.total_prompts(), 704)
        self.assertEqual(banks['1:0'], {'warmup': [0, 1], 'measured': list(range(2, 32))})
        self.assertEqual(len(banks['32:1']['measured']), LEVELS[32])
        used = sorted(i for b in banks.values() for i in b['warmup'] + b['measured'])
        self.assertEqual(used, list(range(704)))

    def test_load_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p.json'
            data = synthetic_file(path)
            self.assertEqual(len(prompts.load(path)['requests']), 704)
            data['requests'][5]['token_ids'] = data['requests'][4]['token_ids']
            data['requests'][5]['first_text_token'] = data['requests'][4]['first_text_token']
            write_json(path, data)
            with self.assertRaises(RuntimeError):
                prompts.load(path)
            data = synthetic_file(path)
            data['model_revision'] = 'other'
            write_json(path, data)
            with self.assertRaises(RuntimeError):
                prompts.load(path)


@unittest.skipUnless(os.environ.get('TPPP_PROMPTS'), 'set TPPP_PROMPTS to check a built prompt file')
class ReferenceFile(unittest.TestCase):
    def test_reference_prompt_file(self):
        path = Path(os.environ['TPPP_PROMPTS'])
        prompts.load(path)
        self.assertEqual(digest(path), REFERENCE_PROMPTS_SHA256)


if __name__ == '__main__':
    unittest.main()
