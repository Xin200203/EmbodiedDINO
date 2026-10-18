#!/usr/bin/env python3
"""Compare local research sources with this release without executing models."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


CODE_SUFFIXES = {'.py', '.sh', '.bash', '.patch', '.js', '.mjs', '.c', '.cpp',
                 '.cu', '.h', '.hpp', '.yaml', '.yml', '.toml'}
EXCLUDED = {
    'tools/vpn_keepalive.sh': 'Machine-specific VPN maintenance, not research code.',
    'docs_audit/ssh_184_robust.sh': 'Machine-specific SSH connectivity helper.',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_map(workspace):
    snapshot = workspace / '.codex_worktrees/object_world_encoder_20260719'
    for top in ['oneformer3d', 'configs', 'tools', 'scripts']:
        for path in sorted((snapshot / top).rglob('*')):
            if path.is_file() and path.suffix in CODE_SUFFIXES:
                yield path, path.relative_to(snapshot), 'model_snapshot'
    for path in sorted((snapshot / 'patches').glob('*.patch')):
        yield path, Path('research_archive/snapshot_patches') / path.name, 'history'
    for top in ['tools', 'docs', 'docs_audit', 'external_audit/current_v5_184_snapshot']:
        for path in sorted((workspace / top).rglob('*')):
            if not path.is_file() or path.suffix not in CODE_SUFFIXES:
                continue
            relative = path.relative_to(workspace)
            if str(relative) in EXCLUDED:
                continue
            destination = relative if top == 'tools' else Path('research_archive') / relative
            yield path, destination, 'experiment_tools' if top == 'tools' else 'history'
    # AutoSeg3D is primary; ESAM supplies missing preprocessing helpers/metadata.
    selected = set()
    for checkout in ['AutoSeg3D', 'ESAM']:
        repo = workspace / checkout
        tracked = subprocess.check_output(
            ['git', 'ls-files', '-z', 'data'], cwd=repo).decode().split('\0')
        for relative in sorted(filter(None, tracked)):
            path = repo / relative
            if path.suffix not in {'.py', '.sh', '.md', '.txt', '.tsv'}:
                continue
            destination = Path(relative)
            if destination in selected:
                continue
            selected.add(destination)
            yield path, destination, 'upstream_preprocessing'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--sync', action='store_true', help='Copy missing sources; never overwrite differing files.')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--ref', help='Verify bytes in a fetched Git revision, e.g. origin/main.')
    args = parser.parse_args()
    if args.ref and args.sync:
        parser.error('--ref cannot be combined with --sync')
    root = Path(__file__).resolve().parents[1]
    records = []
    destinations = {}
    for source, relative, category in source_map(args.workspace):
        expected = digest(source)
        if relative in destinations and destinations[relative] != expected:
            raise RuntimeError(f'Conflicting source mapping: {relative}')
        destinations[relative] = expected
        destination = root / relative
        if not destination.exists() and args.sync:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        status = 'missing'
        if args.ref:
            result = subprocess.run(['git', 'show', f'{args.ref}:{relative}'],
                                    cwd=root, capture_output=True)
            if result.returncode == 0:
                status = 'identical' if hashlib.sha256(result.stdout).hexdigest() == expected else 'different'
        elif destination.exists():
            status = 'identical' if digest(destination) == expected else 'different'
        records.append(dict(source=str(source.relative_to(args.workspace)),
                            destination=str(relative), category=category,
                            sha256=expected, status=status))
    counts = Counter(record['status'] for record in records)
    report = dict(scope='Local source roots enumerated in source_map; remote-only code not verified.',
                  checked_revision=args.ref or 'working_tree',
                  excluded=EXCLUDED, counts=dict(counts), files=records)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(counts=counts, categories=Counter(r['category'] for r in records)), indent=2))
    for record in records:
        if record['status'] != 'identical':
            print(record['status'], record['source'], '->', record['destination'])
    raise SystemExit(0 if counts.get('missing', 0) + counts.get('different', 0) == 0 else 1)


if __name__ == '__main__':
    main()
