"""Content identity for a frozen INT8 provider; no exact wheel version pin."""

from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class FrozenFiles:
    digest: str
    stamps: tuple

    @classmethod
    def capture(cls, files):
        records = []
        stamps = []
        for label, filename in sorted(files.items()):
            path = Path(filename).resolve(strict=True)
            before = path.stat()
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = path.stat()
            stamp = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            before_stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            if before_stamp != stamp:
                raise RuntimeError('INT8 provider changed during identity capture')
            records.append((label, digest))
            stamps.append((str(path), stamp, digest))
        if not records:
            raise RuntimeError('INT8 provider identity is empty')
        digest = hashlib.sha256(repr(records).encode()).hexdigest()
        return cls(digest, tuple(stamps))

    def validate(self):
        for filename, expected, digest in self.stamps:
            with Path(filename).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise RuntimeError('INT8 provider files changed; restart the process')
