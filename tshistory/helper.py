import tempfile
import shutil
from contextlib import contextmanager

try:
    from pathlib import Path
except:
    from pathlib2 import Path


@contextmanager
def tempdir(suffix='', prefix='tmp'):
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
    try:
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp)
