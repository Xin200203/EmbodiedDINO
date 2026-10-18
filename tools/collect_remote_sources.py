#!/usr/bin/env python3
"""Read bounded research source trees over SSH; preserve and verify snapshots."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


REMOTE = r'''
import hashlib, io, json, os, pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1]).resolve()
suffixes = {'.py', '.sh', '.bash', '.yaml', '.yml', '.toml', '.json',
            '.c', '.cpp', '.h', '.hpp', '.cu', '.js', '.mjs'}
files = []
for folder in ('oneformer3d', 'configs', 'tools', 'scripts'):
    for p in sorted((root / folder).rglob('*')):
        if not p.is_file() or p.is_symlink() or p.suffix not in suffixes:
            continue
        if '__pycache__' in p.parts or '.git' in p.parts:
            continue
        try:
            p.resolve().relative_to(root)
        except ValueError:
            raise RuntimeError('Source resolves outside root: ' + str(p))
        files.append(p)
for name in ('requirements.txt', 'LICENSE'):
    p = root / name
    if p.is_file(): files.append(p)
records = []
payload = []
for p in files:
    data = p.read_bytes()
    if len(data) > 8 * 1024 * 1024:
        raise RuntimeError('Unexpected source size: ' + str(p))
    name = p.relative_to(root).as_posix()
    records.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    payload.append((name, data, p.stat().st_mode & 0o777))
manifest = json.dumps(dict(root=str(root), files=records), indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for name, data, mode in payload + [('SOURCE_MANIFEST.json', manifest, 0o644)]:
        info = tarfile.TarInfo(name)
        info.size = len(data); info.mode = mode; info.mtime = 0
        archive.addfile(info, io.BytesIO(data))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--jump')
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3']
    if args.jump:
        ssh += ['-J', args.jump]
    command = 'python3 -c ' + shlex.quote(REMOTE) + ' ' + shlex.quote(args.source)
    result = subprocess.run(ssh + [args.host, command], capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
        manifest = json.load(archive.extractfile('SOURCE_MANIFEST.json'))
        entries = {r['path']: r for r in manifest['files']}
        payload = []
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise RuntimeError('Unsafe member: ' + member.name)
            data = archive.extractfile(member).read()
            if member.name != 'SOURCE_MANIFEST.json':
                expected = entries.pop(member.name)
                assert len(data) == expected['size']
                assert hashlib.sha256(data).hexdigest() == expected['sha256']
            target = args.destination / member.name
            if target.exists() and target.read_bytes() != data:
                raise RuntimeError('Refusing to overwrite a different snapshot: ' + str(target))
            payload.append((target, data, member.mode))
        assert not entries, 'Manifest has missing archive entries'
    for target, data, mode in payload:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)
    print(json.dumps(dict(host=args.host, source=args.source,
                          destination=str(args.destination), files=len(manifest['files']),
                          verified=True)))


if __name__ == '__main__':
    main()
