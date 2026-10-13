import json
import shutil
import tempfile
from pathlib import Path
import unittest

from tpppbench import gates, report
from tpppbench.spec import write_json
from tests.helpers import store_cells, write_manifest, write_round


class ReportTimed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.study = Path(self.tmp.name) / 'study'
        self.timed = self.study / 'timed'
        write_manifest(self.timed, 'timed')

    def tearDown(self):
        self.tmp.cleanup()

    def five_rounds(self, overrides=None):
        for n in range(1, 6):
            write_round(self.timed, n, overrides=(overrides or {}).get(n))

    def store(self, row=None):
        store_cells(self.timed)
        rounds, terminal, _, problem = report.replay_timed(self.study, json.loads(
            (self.timed / 'run-manifest.json').read_text())['topology'])
        self.assertIsNone(problem)
        decision = gates.decision(rounds, terminal)
        if row:
            decision['row'] = row
        write_json(self.timed / 'decision.json', decision)
        return decision

    def test_trade_off_recomputed_and_agrees(self):
        self.five_rounds()
        stored = self.store()
        self.assertEqual(stored['row'], 'TRADE-OFF')
        self.assertEqual(stored['verdicts']['ttft_s:c32'], 'PP better')
        self.assertEqual(stored['verdicts']['token_gap_s:c1'], 'TP better')
        self.assertAlmostEqual(stored['ratios']['ttft_s:c1'][0], 0.60 / 0.45, places=3)
        rep, code = report.build(self.study)
        self.assertEqual(code, 0, rep.lines)
        self.assertIn('row: TRADE-OFF', rep.lines)
        self.assertEqual(len(rep.data['rounds']), 5)
        self.assertAlmostEqual(rep.data['rounds'][0]['levels']['8']['PP4']['ttft_s'], 0.60, places=6)

    def test_tampered_stored_level_value_warns(self):
        self.five_rounds()
        self.store()
        path = self.timed / 'round-3-attempt-0' / 'PP4' / 'c8' / 'gate-calculation.json'
        cell = json.loads(path.read_text())
        cell['ttft_s'] *= 1.01
        write_json(path, cell)
        rep, code = report.build(self.study)
        self.assertEqual(code, 1)
        self.assertTrue(any('PP4/c8: ttft_s' in w for w in rep.warnings), rep.warnings)

    def test_non_default_touch_options_are_printed(self):
        self.five_rounds()
        self.store()
        m = json.loads((self.timed / 'run-manifest.json').read_text())
        m.update(touch_bytes=30 * 10**9, min_usable_free_bytes=15 * 10**9)
        write_json(self.timed / 'run-manifest.json', m)
        rep, _ = report.build(self.study)
        self.assertTrue(any('writes 30 GB' in x for x in rep.lines))
        self.assertTrue(any('15 GB' in x for x in rep.lines))

    def test_tampered_stored_decision_warns(self):
        self.five_rounds()
        self.store(row='ONE-SIDED')
        rep, code = report.build(self.study)
        self.assertEqual(code, 1)
        self.assertTrue(any('decision row' in w for w in rep.warnings))

    def test_incomplete_timed_phase_has_no_decision(self):
        for n in range(1, 4):
            write_round(self.timed, n)
        rep, code = report.build(self.study)
        self.assertEqual(code, 2)
        self.assertTrue(any('incomplete' in x for x in rep.lines))

    def test_quiet_failure_is_rerun_and_the_rerun_counts(self):
        noisy = {('PP4', 8): {'foreign_ticks_per_s': 40}}
        for n in range(1, 6):
            if n == 2:
                write_round(self.timed, 2, attempt=0, overrides=noisy)
                write_round(self.timed, 2, attempt=1)
            else:
                write_round(self.timed, n)
        self.store()
        rep, code = report.build(self.study)
        self.assertEqual(code, 0, rep.lines)
        attempts = rep.data['attempts']
        self.assertEqual([(a['round'], a['attempt']) for a in attempts if not a['Q']], [(2, 0)])
        self.assertEqual(rep.data['rounds'][1]['attempt'], 1)

    def test_three_quiet_failures_end_not_measurable(self):
        noisy = {('TP4', 1): {'foreign_ticks_per_s': 40}}
        write_round(self.timed, 1)
        for k in range(3):
            write_round(self.timed, 2, attempt=k, overrides=noisy)
        rep, code = report.build(self.study)
        self.assertEqual(rep.data['decision']['row'], 'NOT-MEASURABLE')
        self.assertEqual(rep.data['decision']['terminal_gate'], 'Q')
        self.assertEqual(code, 1)  # no stored decision.json

    def test_placement_failure_and_token_retry(self):
        bad_node = {('TP4', 32): {'node0': 900, 'node1': 100}}
        overrides = {1: bad_node}
        for n in range(1, 6):
            write_round(self.timed, n, overrides=overrides.get(n))
        write_round(self.timed, 1, attempt=1)
        rounds, terminal, attempts, problem = report.replay_timed(self.study, json.loads(
            (self.timed / 'run-manifest.json').read_text())['topology'])
        self.assertIsNone(problem)
        self.assertFalse(attempts[0]['N'])
        self.assertEqual(attempts[0]['action'], 'rerun_round')
        self.assertEqual(len(rounds), 5)

    def test_level_retry_directory_is_read(self):
        from tests.helpers import write_level, cache_metric, request
        from tpppbench.spec import LEVELS
        bad_rows = [request(i, completion=100, send_ns=10**12 + i * 10**6) for i in range(LEVELS[1])]
        for n in range(1, 6):
            d = write_round(self.timed, n, overrides={('PP4', 1): {'rows': bad_rows}} if n == 3 else None)
            if n == 3:
                for arm, (ttft, gap) in (('TP4', (0.60, 0.030)), ('PP4', (0.45, 0.090))):
                    base = d / f'{arm}-level-retry'
                    base.mkdir()
                    (base / 'metrics-startup.txt').write_text(cache_metric())
                    write_level(base / 'c1', 1, ttft=ttft, gap=gap)
        stored = self.store()
        self.assertEqual(stored['row'], 'TRADE-OFF')
        self.assertIn('ttft_s:c1', stored['verdicts'])
        shutil.rmtree(self.timed / 'round-3-attempt-0' / 'PP4-level-retry')
        rep, code = report.build(self.study)
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
