import gzip
import json
from pathlib import Path
from typing import Callable, Generator, List


def _is_valid_filetype(file_path: Path) -> bool:
    """Check whether a file is .ndjson or .ndjson.gz."""
    suffixes = file_path.suffixes
    if len(suffixes) == 0 or len(suffixes) > 2:
        return False
    if suffixes[0] != ".ndjson":
        return False
    if len(suffixes) == 2:
        if suffixes[-1] != ".gz":
            return False
    return True


def _get_file_list(data_path: Path) -> List[Path]:
    """Return a list of valid files (.ndjson or .ndjson.gz) from a data_path"""
    file_list: List[Path] = []
    if not data_path.exists():
        return file_list
    if data_path.is_dir():
        file_list = list(data_path.glob("*.ndjson")) + list(
            data_path.glob("*.ndjson.gz")
        )
    else:
        if _is_valid_filetype(data_path):
            file_list = [data_path]
    return file_list


class NdjsonLoader:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self._file_list = _get_file_list(data_path)
        if len(self._file_list) == 0:
            raise ValueError(
                f"""The data_path `{str(data_path)}` is not valid """
                """(must be a .ndjson or .ndjson.gz file, or a folder which """
                """contain at least 1 such file)"""
            )

    def __call__(self) -> Generator[dict, None, None]:
        for file_path in self._file_list:
            is_gzip = file_path.suffixes[-1] == ".gz"
            open_func: Callable = gzip.open if is_gzip else open  # type: ignore
            with open_func(file_path, "rt") as f:
                for line in f:
                    item = json.loads(line)
                    yield item
