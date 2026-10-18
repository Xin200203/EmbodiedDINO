#!/usr/bin/env python3
"""Read source hashes, installed package versions and config-load results over SSH."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess


PROBE = r'''
import hashlib, importlib.metadata as metadata, json, pathlib, sys, traceback
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
result = dict(root=str(root), python=sys.version,
              packages={d.metadata['Name']: d.version for d in metadata.distributions()
                        if d.metadata.get('Name')}, configs=[])
try:
    import oneformer3d
    result['model_import'] = dict(ok=True, path=oneformer3d.__file__)
except Exception:
    result['model_import'] = dict(ok=False, error=traceback.format_exc())
from mmengine.config import Config
for name in json.loads(sys.argv[2]):
    path = root / name
    item = dict(path=name)
    try:
        item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = Config.fromfile(str(path), import_custom_modules=False)
        item.update(ok=True, model_type=config.get('model', {}).get('type'),
                    instance_classes=config.get('num_instance_classes'))
    except Exception:
        item.update(ok=False, error=traceback.format_exc())
    result['configs'].append(item)
print('SOURCE_PROBE_JSON=' + json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--jump')
    parser.add_argument('--python', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    configs = sorted(str(p.relative_to(args.snapshot))
                     for p in (args.snapshot / 'configs').rglob('*.py'))
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3']
    if args.jump:
        ssh += ['-J', args.jump]
    command = 'cd ' + shlex.quote(args.source) + ' && PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(args.python)
    command += ' -B -c ' + shlex.quote(PROBE) + ' ' + shlex.quote(args.source) + ' ' + shlex.quote(json.dumps(configs))
    result = subprocess.run(ssh + [args.host, command], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    marker = 'SOURCE_PROBE_JSON='
    report = json.loads(result.stdout.split(marker, 1)[1])
    report['scope'] = 'Read-only source import/config loading; no model construction, training or evaluation.'
    manifest = json.loads((args.snapshot / 'SOURCE_MANIFEST.json').read_text())
    expected = {f['path']: f['sha256'] for f in manifest['files']}
    for item in report['configs']:
        item['matches_collected_source'] = item.get('sha256') == expected[item['path']]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    failures = [c['path'] for c in report['configs'] if not c['ok'] or not c['matches_collected_source']]
    print(json.dumps(dict(source=args.source, model_import=report['model_import'],
                          configs_checked=len(configs), failures=failures)))
    raise SystemExit(1 if failures or not report['model_import']['ok'] else 0)


if __name__ == '__main__':
    main()
