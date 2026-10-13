import copy
import unittest

from tpppbench import gates
from tpppbench.spec import CAPACITY, LEVELS
from tests.helpers import BASE_NS, SERVER_PID, TOPOLOGY, cache_metric, metrics, request, sample, samples_for


def summarized(rows):
    from tpppbench.client import summarize
    return [summarize(dict(r)) for r in rows]


def level(rows, c=1, hits=0.0, preemptions=0.0, blocks=4000):
    before = metrics(queries=100.0, hits=0.0, preemptions=1.0)
    after = metrics(queries=100.0 + 1536 * len(rows), hits=hits, preemptions=1.0 + preemptions)
    return gates.level(summarized(rows), before, after, cache_metric(blocks=blocks), c)


class LevelGates(unittest.TestCase):
    def rows(self, n=30, **kw):
        return [request(i, send_ns=BASE_NS + i, **kw) for i in range(n)]

    def test_all_pass_and_medians(self):
        result = level(self.rows(ttft=0.25, gap=0.03))
        for gate in 'KTCG':
            self.assertTrue(result[gate], gate)
        self.assertAlmostEqual(result['ttft_s'], 0.25, places=6)
        self.assertAlmostEqual(result['token_gap_s'], 0.03, places=6)
        self.assertFalse(result['rerun_level'])

    def test_drop_limit_is_two_percent_inclusive(self):
        rows = [request(i, send_ns=BASE_NS + i) for i in range(160)]
        for k in range(3):  # 3 of 160 = 1.875% <= 2%
            rows[k] = request(k, completion=127, chunks=127, send_ns=BASE_NS + k)
        self.assertTrue(level(rows, c=32)['T'])
        rows[3] = request(3, prompt=1535, send_ns=BASE_NS + 3)  # 4 of 160 = 2.5%
        result = level(rows, c=32)
        self.assertFalse(result['T'])
        self.assertTrue(result['rerun_level'])
        self.assertEqual(result['dropped'], 4)

    def test_chunk_gate_counts_every_measured_request(self):
        rows = self.rows(80)
        for k in range(4):  # 76/80 = 95%: pass
            rows[k] = request(k, chunks=100, send_ns=BASE_NS + k)
        self.assertTrue(level(rows, c=8)['G'])
        rows[4] = request(4, chunks=127, send_ns=BASE_NS + 4)
        self.assertFalse(level(rows, c=8)['G'])

    def test_token_gap_needs_exactly_128_chunks(self):
        rows = summarized([request(0, chunks=127)])
        self.assertIsNone(rows[0]['token_gap_s'])
        self.assertIsNotNone(rows[0]['ttft_s'])

    def test_cache_gate_strictly_below_one_percent(self):
        rows = self.rows()
        queries = 1536 * 30
        self.assertTrue(level(rows, hits=queries * 0.0099)['C'])
        self.assertFalse(level(rows, hits=queries * 0.01)['C'])

    def test_capacity_and_preemption(self):
        rows = self.rows()
        self.assertFalse(level(rows, preemptions=1)['K'])
        self.assertFalse(level(rows, blocks=CAPACITY // 16 - 1)['K'])
        self.assertTrue(level(rows, blocks=CAPACITY // 16)['K'])

    def test_missing_counter_or_duplicate_request_raises(self):
        rows = summarized(self.rows())
        with self.assertRaises(gates.EvidenceError):
            gates.level(rows, cache_metric(), cache_metric(), cache_metric(), 1)
        with self.assertRaises(gates.EvidenceError):
            gates.level(rows[:-1] + rows[:1], metrics(), metrics(queries=1.0), cache_metric(), 1)
        with self.assertRaises(gates.EvidenceError):
            gates.level(rows, metrics(queries=10.0), metrics(queries=5.0), cache_metric(), 1)

    def test_cache_config_label_parsing(self):
        text = cache_metric().replace('engine="0"', 'engine="0",note="comma, quote\\" slash\\\\"')
        self.assertEqual(gates.capacity(text), 4000 * 16)
        with self.assertRaises(gates.EvidenceError):
            gates.capacity(cache_metric() + cache_metric().splitlines()[1].replace('engine="0"', 'engine="1"'))


class DecisionRule(unittest.TestCase):
    def test_verdict_bounds(self):
        self.assertEqual(gates.metric_verdict([0.89] * 5), 'TP better')
        self.assertEqual(gates.metric_verdict([0.9] * 5), 'no material difference')
        self.assertEqual(gates.metric_verdict([1.1] * 5), 'no material difference')
        self.assertEqual(gates.metric_verdict([1.11] * 5), 'PP better')
        self.assertEqual(gates.metric_verdict([0.5, 0.5, 0.5, 0.5, 0.95]), 'unresolved')
        with self.assertRaises(gates.EvidenceError):
            gates.metric_verdict([1.0] * 4)

    @staticmethod
    def rounds(tp_ttft, tp_gap, pp_ttft=1.0, pp_gap=1.0, flags=None):
        cell = {'K': True, 'T': True, 'C': True, 'G': True}
        out = []
        for k in range(5):
            levels = {}
            for c in LEVELS:
                tp = dict(cell, ttft_s=tp_ttft[c] if isinstance(tp_ttft, dict) else tp_ttft,
                          token_gap_s=tp_gap[c] if isinstance(tp_gap, dict) else tp_gap)
                pp = dict(cell, ttft_s=pp_ttft, token_gap_s=pp_gap)
                for (arm, cc, rnd), changes in (flags or {}).items():
                    if cc == c and rnd in (k, None):
                        (tp if arm == 'TP4' else pp).update(changes)
                levels[str(c)] = {'TP4': tp, 'PP4': pp}
            out.append({'levels': levels})
        return out

    def test_rows(self):
        self.assertEqual(gates.decision(self.rounds(1.5, 0.5))['row'], 'TRADE-OFF')
        self.assertTrue(gates.decision(self.rounds(1.5, 0.5))['expectation']['met'])
        self.assertEqual(gates.decision(self.rounds(1.0, 1.0))['row'], 'NO-STORY')
        self.assertEqual(gates.decision(self.rounds(1.5, 1.0))['row'], 'ONE-SIDED')
        self.assertEqual(gates.decision(self.rounds(0.5, 1.0))['row'], 'ONE-SIDED')
        self.assertEqual(gates.decision(self.rounds({1: 1.5, 8: 1.5, 32: 1.05}, 1.0))['row'], 'ONE-SIDED')
        unstable = gates.decision(self.rounds(1.5, 1.0, flags={('TP4', 1, 0): {'ttft_s': 1.0}}))
        self.assertEqual(unstable['row'], 'UNSTABLE')
        self.assertEqual(unstable['verdicts']['ttft_s:c1'], 'unresolved')

    def test_trade_off_beats_unresolved_and_reversal_is_flagged(self):
        result = gates.decision(self.rounds(0.5, 1.5, flags={('TP4', 1, 0): {'token_gap_s': 1.0}}))
        self.assertEqual(result['row'], 'TRADE-OFF')
        self.assertFalse(result['expectation']['met'])
        self.assertIn('ttft_s:c1', result['expectation']['reversed_verdicts'])

    def test_invalid_levels(self):
        one_valid = self.rounds(1.5, 0.5, flags={('TP4', 8, 2): {'K': False}, ('PP4', 32, None): {'C': False}})
        result = gates.decision(one_valid)
        self.assertEqual(result['valid_levels'], 1)
        self.assertEqual(result['row'], 'NOT-MEASURABLE')
        g_fail = gates.decision(self.rounds(1.5, 0.5, flags={('PP4', 8, 4): {'G': False}}))
        self.assertIn('ttft_s:c8', g_fail['verdicts'])
        self.assertNotIn('token_gap_s:c8', g_fail['verdicts'])
        self.assertEqual(gates.decision([], 'Q')['row'], 'NOT-MEASURABLE')
        with self.assertRaises(gates.EvidenceError):
            gates.decision(self.rounds(1.5, 0.5)[:4])


class Budgets(unittest.TestCase):
    def test_quiet_two_reruns_then_stop(self):
        r = gates.RoundRetries()
        self.assertEqual(r.failed(True, False), 'rerun_round')
        self.assertEqual(r.failed(True, False), 'rerun_round')
        self.assertEqual(r.failed(True, False), 'NOT-MEASURABLE')
        self.assertEqual(r.terminal_gate, 'Q')

    def test_placement_two_reruns_one_replacement_per_study(self):
        study = gates.StudyRetries()
        r = gates.RoundRetries(study)
        self.assertEqual([r.failed(False, True) for _ in range(3)], ['rerun_round', 'rerun_round', 'replace_round'])
        self.assertEqual(r.failed(False, True), 'NOT-MEASURABLE')
        second = gates.RoundRetries(study)
        self.assertEqual([second.failed(False, True) for _ in range(3)],
                         ['rerun_round', 'rerun_round', 'NOT-MEASURABLE'])
        self.assertEqual(second.terminal_gate, 'N')


class RoundAssembly(unittest.TestCase):
    @staticmethod
    def cell(**kw):
        base = {'K': True, 'T': True, 'C': True, 'G': True, 'N': True, 'Q': True, 'rerun_level': False,
                'directory': 'first'}
        base.update(kw)
        base['rerun_level'] = not (base['T'] and base['C'])
        return base

    def test_retry_levels_and_merge(self):
        first = {a: {str(c): self.cell() for c in LEVELS} for a in ('TP4', 'PP4')}
        first['PP4']['8'] = self.cell(T=False)
        first['TP4']['32'] = self.cell(K=False, C=False)
        self.assertEqual(gates.retry_levels(first), [8])
        retry = {a: {'8': self.cell(directory='retry')} for a in ('TP4', 'PP4')}
        paired, history = gates.paired_round(1, 0, first, retry)
        self.assertTrue(paired['levels']['8']['PP4']['T'])
        self.assertEqual(paired['levels']['8']['PP4']['replaces'], 'first')
        self.assertEqual(len(history), 8)

    def test_reached_g_failure_and_k_failure_survive_a_retry(self):
        original = self.cell(G=False, C=False)
        self.assertTrue(gates.merge_retry(original, self.cell())['G'])
        original = self.cell(G=False, T=False)
        self.assertTrue(gates.merge_retry(original, self.cell())['G'])
        self.assertFalse(gates.merge_retry(self.cell(G=False), self.cell())['G'])
        self.assertFalse(gates.merge_retry(self.cell(K=False), self.cell())['K'])

    def test_n_and_q_are_judged_over_retry_cells_too(self):
        first = {a: {str(c): self.cell() for c in LEVELS} for a in ('TP4', 'PP4')}
        first['TP4']['1'] = self.cell(T=False)
        retry = {'TP4': {'1': self.cell(Q=False)}, 'PP4': {'1': self.cell()}}
        paired, _ = gates.paired_round(1, 0, first, retry)
        self.assertFalse(paired['Q'])
        self.assertTrue(paired['N'])


class Placement(unittest.TestCase):
    def rows(self, fraction=1.0, affinity=None, shm=None):
        return {'1': {'node_fraction_lower_bound': fraction, 'affinity': affinity or [0, 1],
                      'nccl_shm_pages': {'0': 10} if shm is None else shm}}

    def test_thresholds(self):
        self.assertTrue(gates.placement(self.rows(0.95), [0, 1, 2], 0))
        self.assertFalse(gates.placement(self.rows(0.949), [0, 1, 2], 0))
        self.assertFalse(gates.placement(self.rows(affinity=[0, 5]), [0, 1, 2], 0))
        self.assertFalse(gates.placement(self.rows(shm={}), [0, 1, 2], 0))
        self.assertFalse(gates.placement(self.rows(shm={'0': 10, '1': 1}), [0, 1, 2], 0))
        self.assertTrue(gates.placement(self.rows(shm={'1': 10}), [0, 1, 2], 1))
        self.assertFalse(gates.placement({'1': {'error': 'gone'}}, [0], 0))
        with self.assertRaises(gates.EvidenceError):
            gates.placement({}, [0], 0)


class Quiet(unittest.TestCase):
    begin, end = BASE_NS, BASE_NS + 10 * 10**9

    def q(self, samples):
        return gates.quiet(samples, TOPOLOGY['server_cpus'], self.begin, self.end, TOPOLOGY['gpus'], 8)

    def test_clean_level_passes(self):
        result = self.q(samples_for(self.begin, self.end))
        self.assertTrue(result['pass'], result)
        self.assertEqual(result['foreign_cpu_fraction'], 0)
        self.assertEqual(result['telemetry_gaps']['intervals_over_threshold'], 0)

    def test_foreign_cpu_limit(self):
        # 4 CPUs x 100 ticks/s; 7 foreign ticks/s = 1.75%; 9 = 2.25%
        self.assertTrue(self.q(samples_for(self.begin, self.end, foreign_ticks_per_s=7))['pass'])
        result = self.q(samples_for(self.begin, self.end, foreign_ticks_per_s=9))
        self.assertFalse(result['pass'])
        self.assertIn('foreign CPU time', result['reasons'])

    def test_foreign_compute_process_on_any_card(self):
        samples = samples_for(self.begin, self.end)
        samples[3]['compute_apps'] = [{'pid': 777, 'uuid': 'GPU-7', 'name': 'other'}]
        self.assertFalse(self.q(samples)['pass'])
        samples[3]['compute_apps'] = [{'pid': SERVER_PID, 'uuid': 'GPU-0', 'name': 'vllm'}]
        self.assertTrue(self.q(samples)['pass'])

    def test_low_clock_share(self):
        n = len(samples_for(self.begin, self.end))
        ok = samples_for(self.begin, self.end, clocks=[2500.0] * 11 + [2000.0] + [2500.0] * n)
        self.assertTrue(self.q(ok)['pass'])
        bad = samples_for(self.begin, self.end, clocks=[2500.0] * 8 + [2000.0] * 3 + [2500.0] * n)
        self.assertFalse(self.q(bad)['pass'])

    def test_errored_sample_fails_instead_of_raising(self):
        samples = samples_for(self.begin, self.end)
        samples[4] = sample(samples[4]['monotonic_ns'], 0, 0, 0, error='nvidia-smi timed out')
        result = self.q(samples)
        self.assertFalse(result['pass'])
        self.assertEqual(result['reasons'][0], gates.SAMPLE_ERROR)
        for k in (0, 1):  # no clean sample at or before the level's start remains
            samples[k] = sample(samples[k]['monotonic_ns'], 0, 0, 0, error='boundary failed')
        self.assertIn('not_computable', self.q(samples))

    def test_missing_card_or_unenclosed_window_raises(self):
        samples = samples_for(self.begin, self.end)
        samples[2]['gpus'] = samples[2]['gpus'][:7]
        with self.assertRaises(gates.EvidenceError):
            self.q(samples)
        with self.assertRaises(gates.EvidenceError):
            self.q(samples_for(self.begin, self.end)[2:])

    def test_mid_gap_is_recorded_not_gated(self):
        samples = samples_for(self.begin, self.end)
        del samples[5:8]
        result = self.q(samples)
        self.assertTrue(result['pass'])
        self.assertEqual(result['telemetry_gaps']['intervals_over_threshold'], 1)

    def test_edge_gap_scales_the_cpu_denominator(self):
        samples = samples_for(self.begin, self.end, step_s=1.0)
        first = copy.deepcopy(samples[0])
        first['monotonic_ns'] = self.begin - 30 * 10**9  # a 30 s edge interval before the level
        for cpu in first['cpu'].values():
            cpu['busy'], cpu['total'] = -1500, -3000  # 30 s at 50 busy / 100 total ticks per second
        first['owned'][str(SERVER_PID)]['ticks'] = -6000  # the server's 50 ticks/s on each of 4 CPUs
        samples[0:2] = [first]
        foreign = copy.deepcopy(samples)
        for s in foreign[1:]:
            for cpu in s['cpu'].values():
                cpu['busy'] += 60  # 240 foreign ticks over the level's 10 s x 400 = 6%, diluted to ~1.5%
        result = self.q(foreign)
        self.assertLess(result['foreign_cpu_fraction_unscaled'], 0.02)
        self.assertGreater(result['foreign_cpu_fraction'], 0.02)
        self.assertFalse(result['pass'])


if __name__ == '__main__':
    unittest.main()
