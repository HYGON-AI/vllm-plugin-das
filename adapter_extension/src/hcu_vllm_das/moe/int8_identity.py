"""Content identity for a frozen INT8 provider; no exact wheel version pin."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


def _stamp(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class FrozenFiles:
    digest: str
    stamps: tuple
    path_targets: tuple

    @classmethod
    def capture(cls, files, watch_paths=()):
        records = []
        stamps = []
        examined = set()
        targets = {}

        def watch(path):
            targets[path] = _stamp(path.stat())[:2]

        for path in watch_paths:
            watch(Path(path).absolute())
        for label, filename in sorted(files.items()):
            path = Path(filename).absolute()
            for entry in (path, *path.parents):
                if entry in examined:
                    break
                examined.add(entry)
                if entry.is_symlink():
                    watch(entry)
            with path.open('rb') as stream:
                before = _stamp(os.fstat(stream.fileno()))
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                after = _stamp(os.fstat(stream.fileno()))
            if before != after or _stamp(path.stat()) != after:
                raise RuntimeError('INT8 provider changed during identity capture')
            records.append((label, digest))
            stamps.append((str(path), after, digest))
        if not records:
            raise RuntimeError('INT8 provider identity is empty')
        digest = hashlib.sha256(repr(records).encode()).hexdigest()
        return cls(digest, tuple(stamps), tuple(targets.items()))

    def validate_paths(self):
        for path, expected in self.path_targets:
            if _stamp(path.stat())[:2] != expected:
                raise RuntimeError('INT8 provider path changed; restart the process')

    def validate(self):
        self.validate_paths()
        for filename, _, digest in self.stamps:
            with Path(filename).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise RuntimeError('INT8 provider files changed; restart the process')
