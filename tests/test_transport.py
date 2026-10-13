import tempfile
from pathlib import Path
import unittest

from tpppbench import gates


def nccl_log(arm):
    """Transport lines in NCCL 2.29's formats, as NCCL_DEBUG_FILE writes them."""
    out = []
    for r in range(4):
        pid = 100 + r
        p = f'host:{pid}:{pid} [{r}] NCCL INFO '
        out += [f'{p}[Rank {r}] ncclCommInitRank comm 0x7f{r} rank {r} nranks 4 cudaDev {r} '
                f'nvmlDev {r} busId {r}1000 commId 0xabc - Init START',
                f'{p}Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 0',
                f'{p}ncclCommInitRank comm 0x7f{r} rank {r} nranks 4 cudaDev {r} nvmlDev {r} '
                f'busId {r}1000 commId 0xabc - Init COMPLETE']
        if arm == 'PP4':
            # A one-rank tensor-parallel group: no peers, flags stay 1 (exempt).
            out += [f'{p}Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1',
                    f'{p}ncclCommSplit comm 0x8f{r} rank 0 nranks 1 cudaDev {r} nvmlDev {r} busId {r}1000 '
                    f'parent 0x7f{r} childCount 1 color {r} key 0 - Init COMPLETE']
        out += [f'host:{pid}:{pid + 5} [{r}] NCCL INFO MMAP allocated shareable host buffer '
                f'/dev/shm/nccl-a{r} size 9637888 ptr 0x7f00000{r}000',
                f'{p}Channel 00 : {r}[{r}] -> {(r + 1) % 4}[{(r + 1) % 4}] via SHM/direct/direct',
                f'{p}Channel 00/02 : 0 1 2 3']
    if arm == 'PP4':
        out += [f'host:{100 + r}:{100 + r} [{r}] NCCL INFO Channel 01 : {r}[{r}] -> {r + 1}[{r + 1}] via SHM/direct/direct'
                for r in range(3)]
    return '\n'.join(out)


def split(text):
    files = {}
    for line in text.splitlines():
        match = gates.NCCL_ANY_PREFIX.search(line)
        files.setdefault(int(match[2]), []).append(line)
    return {pid: '\n'.join(lines) + '\n' for pid, lines in files.items()}


class GateP(unittest.TestCase):
    def test_all_shm_passes_for_both_arms(self):
        for arm in ('TP4', 'PP4'):
            with self.subTest(arm=arm):
                result = gates.transport(split(nccl_log(arm)), arm, [100, 101, 102, 103])
                self.assertTrue(result['pass'], result['checks'])
        pp = gates.transport(split(nccl_log('PP4')), 'PP4')
        self.assertEqual(len(pp['single_rank_communicators_exempt']), 4)
        self.assertEqual(pp['adjacent_pairs_observed'], [[0, 1], [1, 2], [2, 3]])

    def test_p2p_line_or_p2p_capable_communicator_fails(self):
        for arm in ('TP4', 'PP4'):
            log = nccl_log(arm)
            bad = log.replace('Channel 00 : 1[1] -> 2[2] via SHM/direct/direct',
                              'Channel 00/1 : 1[1] -> 2[2] via P2P/CUMEM/CE')
            self.assertFalse(gates.transport(split(bad), arm)['pass'])
            bad = log + '\nhost:101:101 [1] NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/IPC'
            self.assertFalse(gates.transport(split(bad), arm)['pass'])
            bad = log.replace('isAllDirectP2p 0 directMode 0 isAllCudaP2p 0',
                              'isAllDirectP2p 1 directMode 0 isAllCudaP2p 1', 1)
            self.assertFalse(gates.transport(split(bad), arm)['pass'])
            bad = log.replace('via SHM/direct/direct', 'via NET/Socket/0', 1)
            self.assertFalse(gates.transport(split(bad), arm)['checks']['all_channels_shm'])

    def test_unpaired_check_line_fails_closed(self):
        log = nccl_log('TP4')
        first = next(x for x in log.splitlines() if 'Check P2P Type' in x)
        with self.assertRaises(gates.EvidenceError):
            gates.transport(split(log.replace(first + '\n', '')), 'TP4')

    def test_host_buffer_line_required(self):
        log = '\n'.join(x for x in nccl_log('TP4').splitlines() if 'allocated shareable host buffer' not in x)
        self.assertFalse(gates.transport(split(log), 'TP4')['pass'])

    def test_receiver_without_channel_line_passes(self):
        log = '\n'.join(x for x in nccl_log('PP4').splitlines()
                        if not (x.startswith('host:103:') and ' via ' in x))
        self.assertTrue(gates.transport(split(log), 'PP4')['pass'])

    def test_missing_worker_file_or_foreign_line_fails(self):
        files = split(nccl_log('TP4'))
        with self.assertRaises(gates.EvidenceError):
            gates.transport({k: v for k, v in files.items() if k != 103}, 'TP4', [100, 101, 102, 103])
        files[100] += 'host:101:101 [1] NCCL INFO Channel 00 : 1[1] -> 2[2] via SHM/direct/direct\n'
        with self.assertRaises(gates.EvidenceError):
            gates.transport(files, 'TP4')

    def test_files_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            for pid, body in split(nccl_log('TP4')).items():
                (Path(tmp) / f'nccl.host.{pid}.log').write_text(body)
            self.assertEqual(sorted(gates.nccl_files(tmp)), [100, 101, 102, 103])
            (Path(tmp) / 'nccl.bad.name.x.log').write_text('')
            with self.assertRaises(gates.EvidenceError):
                gates.nccl_files(tmp)


if __name__ == '__main__':
    unittest.main()
