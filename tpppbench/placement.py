"""Bind CPUs and set an inherited preferred-node memory policy, then exec the server.

Run by file path (it imports nothing from the package):
    python placement.py --cpus 0,1,2 --node 0 --receipt r.json -- vllm serve ...

MPOL_PREFERRED, not MPOL_BIND: a strict bind fails an allocation when the node is short,
a preferred policy spills to another node (which gate N then detects). Both the affinity
and the policy are inherited by every process vLLM starts.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform

SYS_SET_MEMPOLICY = 238  # x86_64
MPOL_PREFERRED = 1


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cpus', required=True)
    p.add_argument('--node', type=int, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    cpus = {int(x) for x in args.cpus.split(',')}
    if not command or platform.machine() != 'x86_64':
        raise RuntimeError('Needs x86_64 (syscall number) and a command to exec')
    os.sched_setaffinity(0, cpus)
    libc = ctypes.CDLL(None, use_errno=True)
    mask = ctypes.c_ulong(1 << args.node)
    status = libc.syscall(ctypes.c_long(SYS_SET_MEMPOLICY), ctypes.c_int(MPOL_PREFERRED),
                          ctypes.byref(mask), ctypes.c_ulong(64))
    if status:
        raise OSError(ctypes.get_errno(), f'set_mempolicy(MPOL_PREFERRED, node {args.node})')
    args.receipt.write_text(json.dumps({'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
                                       'mempolicy': 'MPOL_PREFERRED', 'node': args.node,
                                       'argv': command}) + '\n')
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
