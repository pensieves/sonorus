from pathlib import Path
import uuid


def create_random_dir(work_dir=".", prefix="sonorus"):
    random_dir = Path(work_dir) / f"{prefix}_{uuid.uuid4().hex}"
    random_dir.mkdir(parents=True, exist_ok=True)
    return random_dir
