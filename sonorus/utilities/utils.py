from pathlib import Path
import uuid
import tarfile, zipfile

from sonorus import CACHE_DIR


def create_random_dir(work_dir=CACHE_DIR, prefix="sonorus"):
    random_dir = Path(work_dir) / f"{prefix}_{uuid.uuid4().hex}"
    random_dir.mkdir(parents=True, exist_ok=True)
    return random_dir


def unpack_archive(filename, unpack_dir=None):

    unpacked = list()
    archive_opener = None

    if tarfile.is_tarfile(filename):
        archive_opener = tarfile.open
        names_method = "getnames"

    elif zipfile.is_zipfile(filename):
        archive_opener = zipfile.ZipFile
        names_method = "namelist"

    if archive_opener:
        unpack_dir = Path(unpack_dir) if unpack_dir else Path(filename).parent

        with archive_opener(filename) as f:
            f.extractall(path=unpack_dir)

            unpacked = [
                (unpack_dir / fn).resolve()
                for fn in getattr(f, names_method)()
                if str(Path(fn).parent) == "." 
                # i.e only top level directory and filenames
            ]

    return unpacked[0] if len(unpacked) == 1 else unpacked
