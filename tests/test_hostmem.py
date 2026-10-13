"""Touch-and-free helper: readings from real /proc/zoneinfo and node meminfo text of a
two-socket machine (tests/fixtures), the worker's order of operations, and a real helper
process that writes 1 MiB."""
import json
from pathlib import Path
import platform
import queue
import tempfile
import unittest
from unittest.mock import patch

from tpppbench import hostmem

FIX = Path(__file__).resolve().parent / 'fixtures'
ZONEINFO = (FIX / 'zoneinfo.txt').read_text()
MEMINFO = {n: (FIX / f'node{n}_meminfo.txt').read_text() for n in (0, 1)}


def meminfo(node, active_kb, inactive_kb, free_kb=1000):
    return (f'Node {node} MemTotal:       527987716 kB\nNode {node} MemFree:  {free_kb} kB\n'
            f'Node {node} Active(file):   {active_kb} kB\nNode {node} Inactive(file): {inactive_kb} kB\n'
            f'Node {node} HugePages_Total:     0\n')


def zoneinfo(node, free, low):
    return (f'Node {node}, zone    DMA32\n  pages free     410282\n        min      416\n        low      416\n'
            f'Node {node}, zone   Normal\n  per-node stats\n      nr_inactive_file 5\n'
            f'  pages free     {free}\n        min      16192\n        low      {low}\n        high     279350\n'
            f'        protection: (0, 0, 0, 0, 0)\n      nr_free_pages {free}\n')


class ReadingTests(unittest.TestCase):
    def test_node0_reclaimable_from_real_text(self):
        r = hostmem.snapshot(0, ZONEINFO, MEMINFO[0])
        # Node 0 Normal: pages free 577505, low 147771; Active(file) 215211188 kB,
        # Inactive(file) 287414664 kB (tests/fixtures).
        self.assertEqual((r['normal_free_pages'], r['normal_low_pages']), (577505, 147771))
        self.assertEqual(r['usable_free_bytes'], (577505 - 147771) * 4096)
        self.assertEqual(r['active_file_bytes'], 215211188 * 1024)
        self.assertEqual(r['inactive_file_bytes'], 287414664 * 1024)
        self.assertEqual(r['reclaimable_bytes'], (577505 - 147771) * 4096 + (215211188 + 287414664) * 1024)
        self.assertTrue(hostmem.admits(r))

    def test_node1_per_node_stats_block_does_not_confuse_the_parser(self):
        # Node 1's Normal zone opens with a 'per-node stats' block before 'pages free'.
        r = hostmem.snapshot(1, ZONEINFO, MEMINFO[1])
        self.assertEqual((r['normal_free_pages'], r['normal_low_pages']), (7499538, 148373))
        self.assertGreater(r['usable_free_bytes'], 20 * 10**9)
        self.assertTrue(hostmem.admits(r))

    def test_dma32_is_not_counted(self):
        zones = hostmem.parse_zoneinfo(ZONEINFO)
        dma32 = zones[(0, 'DMA32')]['free']
        r = hostmem.snapshot(0, ZONEINFO, MEMINFO[0])
        self.assertEqual(r['normal_free_pages'], zones[(0, 'Normal')]['free'])
        self.assertNotEqual(r['normal_free_pages'], dma32 + zones[(0, 'Normal')]['free'])

    def test_units_are_decimal_and_threshold_is_strict(self):
        self.assertEqual(hostmem.REQUIRED_BYTES, 44 * 10**9)
        self.assertEqual(hostmem.TOUCH_BYTES, 40 * 10**9)
        # Exactly 44e9: free above low 0, file pages 44e9 bytes = 42968750 kB.
        exact = hostmem.snapshot(0, zoneinfo(0, 100, 100), meminfo(0, 42968750, 0))
        self.assertEqual(exact['reclaimable_bytes'], 44 * 10**9)
        self.assertFalse(hostmem.admits(exact))
        above = hostmem.snapshot(0, zoneinfo(0, 101, 100), meminfo(0, 42968750, 0))
        self.assertTrue(hostmem.admits(above))

    def test_free_below_low_watermark_is_floored_in_reclaimable_only(self):
        r = hostmem.snapshot(0, zoneinfo(0, 100, 200), meminfo(0, 10, 20))
        self.assertEqual(r['usable_free_bytes'], -100 * 4096)
        self.assertEqual(r['reclaimable_bytes'], 30 * 1024)
        self.assertFalse(hostmem.admits(r))

    def test_missing_evidence_raises(self):
        with self.assertRaisesRegex(ValueError, 'Normal'):
            hostmem.snapshot(1, zoneinfo(0, 1, 1), meminfo(1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'lacks'):
            hostmem.snapshot(0, zoneinfo(0, 1, 1), 'Node 0 MemFree: 5 kB\n')
        with self.assertRaisesRegex(ValueError, 'lacks'):  # node 1's text read for node 0
            hostmem.snapshot(0, ZONEINFO, MEMINFO[1])


class WorkerTests(unittest.TestCase):
    def run_worker(self, reading):
        calls, q = [], queue.Queue()
        with patch.object(hostmem.os, 'setsid', lambda: calls.append('setsid')), \
                patch.object(hostmem, 'set_oom_score_adj', lambda: calls.append('oom') or 1000), \
                patch.object(hostmem.os, 'sched_setaffinity', lambda pid, cpus: calls.append(('cpus', sorted(cpus)))), \
                patch.object(hostmem.os, 'sched_getaffinity', lambda pid: {25, 77}), \
                patch.object(hostmem, 'bind_node', lambda node: calls.append(('bind', node)) or {'mode': 'MPOL_BIND'}), \
                patch.object(hostmem, 'snapshot', lambda node: calls.append(('read', node)) or reading), \
                patch.object(hostmem, 'touch', lambda n: calls.append(('touch', n)) or n):
            hostmem.worker(0, 123, (25, 77), hostmem.REQUIRED_BYTES, q)
        return calls, q.get_nowait()

    def test_oom_score_adj_and_bind_precede_the_reading_and_the_allocation(self):
        ok = hostmem.snapshot(0, ZONEINFO, MEMINFO[0])
        calls, row = self.run_worker(ok)
        self.assertEqual(calls, ['setsid', 'oom', ('cpus', [25, 77]), ('bind', 0), ('read', 0),
                                 ('touch', 123), ('read', 0)])
        self.assertFalse(row['refused'])
        self.assertEqual((row['oom_score_adj'], row['bytes'], row['touch_bytes']), (1000, 123, 123))

    def test_refusal_allocates_nothing(self):
        low = hostmem.snapshot(0, zoneinfo(0, 100, 100), meminfo(0, 1, 1))
        calls, row = self.run_worker(low)
        self.assertNotIn('touch', [c[0] for c in calls if isinstance(c, tuple)])
        self.assertTrue(row['refused'])
        self.assertEqual(row['bytes'], 0)
        self.assertIn('nothing allocated', row['reason'])

    def test_set_oom_score_adj_writes_and_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'oom_score_adj'
            path.write_text('0\n')
            self.assertEqual(hostmem.set_oom_score_adj(path=path), 1000)
            self.assertEqual(path.read_text(), '1000\n')

    def test_run_refuses_on_a_one_node_machine(self):
        with patch.object(hostmem, 'two_nodes', return_value=False), tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'two NUMA nodes'):
                hostmem.run((0,), Path(d) / 'x', set())


@unittest.skipUnless(platform.system() == 'Linux' and platform.machine() == 'x86_64'
                     and Path('/sys/devices/system/node/node0/meminfo').exists(), 'needs Linux x86_64')
class RealProcessTests(unittest.TestCase):
    """A real spawned helper on node 0 with 1 MiB (not 40e9 bytes)."""

    def run_small(self, required):
        added = []

        class Ledger(set):
            def add(self, pid):
                added.append(pid)
                super().add(pid)
        owned = Ledger()
        with tempfile.TemporaryDirectory() as d, patch.object(hostmem, 'two_nodes', return_value=True):
            try:
                return hostmem.run((0,), Path(d) / 'touch', owned, nbytes=1 << 20, required=required,
                                   timeout=120), owned, added, None
            except hostmem.Refusal as e:
                self.assertTrue((Path(d) / 'touch' / 'touch.json').is_file())
                return None, owned, added, e

    def test_real_helper_sets_oom_score_adj_binds_and_touches(self):
        rows, owned, added, _ = self.run_small(required=0)
        row = rows[0]
        self.assertEqual(row['oom_score_adj'], 1000)
        self.assertEqual(row['policy'], {'mode': 'MPOL_BIND', 'nodemask': 1})
        self.assertEqual(row['bytes'], 1 << 20)
        self.assertEqual(len(added), 1)
        self.assertEqual(set(owned), set())  # discarded after a clean exit
        self.assertIn('after', row)

    def test_real_refusal_raises_with_rows(self):
        _, owned, added, refusal = self.run_small(required=10**30)
        self.assertIsNotNone(refusal)
        self.assertTrue(refusal.record[0]['refused'])
        self.assertEqual(refusal.record[0]['bytes'], 0)
        self.assertEqual(set(owned), set())


    def test_stalled_helper_is_killed_and_joined(self):
        # A helper that never reports (here: sleeps) is killed when run() gives up.
        added = []

        class Ledger(set):
            def add(self, pid):
                added.append(pid)
                super().add(pid)
        owned = Ledger()
        with tempfile.TemporaryDirectory() as d, patch.object(hostmem, 'two_nodes', return_value=True), \
                patch.object(hostmem, 'worker', sleeping_worker):
            with self.assertRaisesRegex(RuntimeError, 'stalled'):
                hostmem.run((0,), Path(d) / 'touch', owned, nbytes=1 << 20, required=0, timeout=1)
            killed = json.loads((Path(d) / 'touch' / 'killed.json').read_text())
        self.assertEqual(len(added), 1)
        self.assertFalse(Path(f'/proc/{added[0]}').exists() and
                         Path(f'/proc/{added[0]}/stat').read_text().split(')')[1].split()[0] != 'Z')
        self.assertEqual([k['pid'] for k in killed], added)
        self.assertFalse(killed[0]['alive_after_kill'])
        self.assertEqual(set(owned), set())


def sleeping_worker(node, nbytes, cpus, required, result_queue):
    import time as _time
    _time.sleep(600)


if __name__ == '__main__':
    unittest.main()
