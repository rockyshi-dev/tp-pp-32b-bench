"""Command line: python -m tpppbench <command> ..."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

from .spec import DEFAULT_GPUS, TOUCH_BYTES, USABLE_FREE_AFTER_HELPER

PHASE_COMMANDS = ('start-test', 'settings', 'transport', 'timed')
DEFAULT_LOCK_DIR = '/tmp/tpppbench-locks'


def add_common(p):
    p.add_argument('--study', type=Path, required=True,
                   help='study directory; each phase writes <study>/<phase>/ and must not exist yet')
    p.add_argument('--model', type=Path, default=os.environ.get('TPPP_MODEL'),
                   help='local directory of Qwen/Qwen2.5-32B-Instruct (env TPPP_MODEL)')
    p.add_argument('--prompts', type=Path, default=os.environ.get('TPPP_PROMPTS'),
                   help='prompt file from the `prompts` command (env TPPP_PROMPTS)')
    p.add_argument('--gpus', default=','.join(map(str, DEFAULT_GPUS)),
                   help='four GPU indices on one CPU socket (default 0,1,2,3)')
    p.add_argument('--port', type=int, default=8140)
    p.add_argument('--vllm', type=Path, default=None,
                   help='vllm executable (default: the one beside this Python interpreter)')
    p.add_argument('--cache-dir', type=Path, default=None,
                   help='vLLM, Triton and Inductor compile caches (default <study>/compile-cache)')
    p.add_argument('--lock-dir', type=Path, default=Path(os.environ.get('TPPP_LOCK_DIR', DEFAULT_LOCK_DIR)),
                   help=f'per-GPU lock files (env TPPP_LOCK_DIR, default {DEFAULT_LOCK_DIR})')
    p.add_argument('--touch-gb', type=float, default=TOUCH_BYTES / 1e9,
                   help='GB (1e9 bytes) the touch-and-free helper writes before every server start; '
                        '0 turns it off (default 40)')
    p.add_argument('--min-usable-free-gb', type=float, default=USABLE_FREE_AFTER_HELPER / 1e9,
                   help='usable free memory on the servers\' node required after the helper (default 20)')
    p.add_argument('--deviation-evidence', type=Path, action='append', default=[],
                   help='a failed default start-test start directory whose failure licenses a declared '
                        'NCCL deviation (repeatable); the deviation then applies to both arms')
    p.add_argument('--lock', type=Path, action='append', default=[],
                   help='an extra lock file held for the whole phase (repeatable); "{gpu}" in the path is '
                        'expanded once per GPU, e.g. /shared/locks/gpu{gpu}.lock')
    p.add_argument('--expect-cards', type=int, default=8,
                   help='number of cards the machine must show; gate Q watches all of them (default 8)')
    p.add_argument('--allow-other-versions', action='store_true')
    p.add_argument('--allow-other-prompts', action='store_true')
    p.add_argument('--allow-other-model', action='store_true',
                   help='run with model JSON files that differ from the reference revision (recorded)')
    p.add_argument('--no-setsid-check', action='store_true',
                   help='run even when not started as a session leader (recorded)')


def parser():
    p = argparse.ArgumentParser(prog='python -m tpppbench',
                                description='TP4 versus PP4 for Qwen2.5-32B-Instruct in vLLM 0.30.0 on four GPUs.')
    sub = p.add_subparsers(dest='command', required=True, metavar='command')
    q = sub.add_parser('prompts', help='build the prompt file (CPU only)')
    q.add_argument('--vllm-source', type=Path, required=True, help='clean git checkout of vLLM v0.30.0')
    q.add_argument('--model', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('verify-prompts', help='replay every prompt from the source and tokenizer (CPU only)')
    q.add_argument('--prompts', type=Path, required=True)
    q.add_argument('--vllm-source', type=Path, required=True)
    q.add_argument('--model', type=Path, required=True)
    for name, text in (('start-test', 'start each arm, one request, one untimed c=32 rehearsal'),
                       ('settings', 'one untimed DEBUG-log start per arm to record resolved settings'),
                       ('transport', 'one NCCL_DEBUG start per arm, 8 requests; gate P'),
                       ('timed', 'five paired rounds at c = 1, 8, 32; gates; decision'),
                       ('all', 'start-test, settings, transport, timed; stops at the first that does not pass')):
        add_common(sub.add_parser(name, help=text))
    q = sub.add_parser('report', help='recompute gates, ratios and the decision from a study directory')
    q.add_argument('study', type=Path)
    q.add_argument('--json', type=Path, help='also write the recomputed values as JSON')
    return p


def config(args, argv):
    from . import server, telemetry
    if args.model is None or args.prompts is None:
        raise SystemExit('--model and --prompts are required (or TPPP_MODEL and TPPP_PROMPTS)')
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise SystemExit('--gpus needs four distinct indices')
    model = args.model.absolute()
    if not (model / 'config.json').exists():
        raise SystemExit(f'{model} has no config.json')
    study = args.study.absolute()
    session_leader = os.getsid(0) == os.getpid()
    if not session_leader and not args.no_setsid_check:
        raise SystemExit('Start the run under setsid (so a closed terminal cannot stop it midway), '
                         'or pass --no-setsid-check')
    cfg = SimpleNamespace(
        study=study, model=model, prompts=args.prompts.absolute(), gpus=gpus, port=args.port,
        vllm=(args.vllm or Path(sys.executable).parent / 'vllm').absolute(),
        cache_dir=(args.cache_dir or study / 'compile-cache').absolute(), lock_dir=args.lock_dir,
        touch_bytes=int(args.touch_gb * 1e9), min_usable_free=int(args.min_usable_free_gb * 1e9),
        deviation_evidence=args.deviation_evidence, allow_other_prompts=args.allow_other_prompts,
        allow_other_model=args.allow_other_model, locks=list(args.lock),
        argv=[sys.executable, '-m', 'tpppbench', *argv], session_leader=session_leader)
    try:
        server.vllm_interpreter(cfg.vllm, args.allow_other_versions)
        cfg.versions = server.check_versions(args.allow_other_versions)
        cfg.versions.update(server.check_vllm(cfg.vllm, cfg.versions, args.allow_other_versions))
        cfg.topology = telemetry.topology(gpus, args.expect_cards)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'error: {exc}')
    study.mkdir(parents=True, exist_ok=True)
    return cfg


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if args.command == 'prompts':
        from . import prompts
        prompts.construct(args.vllm_source, args.model, args.output)
        return 0
    if args.command == 'verify-prompts':
        from . import prompts
        print(json.dumps(prompts.verify(args.prompts, args.vllm_source, args.model)))
        return 0
    if args.command == 'report':
        from . import report
        return report.main(args)
    from . import run
    cfg = config(args, argv)
    phases = ('start-test', 'settings', 'transport', 'timed') if args.command == 'all' else (args.command,)
    for phase in phases:
        print(json.dumps({'phase': phase, 'status': 'starting'}), flush=True)
        passed = run.execute(cfg, phase)
        print(json.dumps({'phase': phase, 'pass': passed}), flush=True)
        if not passed:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
