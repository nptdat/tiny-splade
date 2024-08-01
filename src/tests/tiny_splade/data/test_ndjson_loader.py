from pathlib import Path
from typing import List

import pytest

from tiny_splade.data.ndjson_loader import _get_file_list, _is_valid_filetype


@pytest.mark.parametrize(
    "file_path, exp",
    [
        (Path("d"), False),
        (Path("path/to/data/a.ndjson"), True),
        (Path("path/to/data/a.ndjson.gz"), True),
        (Path("path/to/data/a.ndjson.gzip"), False),
        (Path("path/to/data/a.txt.gz"), False),
    ],
)
def test__is_valid_filetype(file_path: Path, exp: bool) -> None:
    assert _is_valid_filetype(file_path) == exp


@pytest.mark.parametrize(
    "init_files, exp_files_list",
    [
        ([], []),
        ([Path("a.txt")], []),
        ([Path("folder"), Path("folder/a.ndjson")], [Path("folder/a.ndjson")]),
        (
            [
                Path("folder"),
                Path("folder/a.ndjson"),
                Path("folder/b.ndjson.gz"),
                Path("folder/c"),
            ],
            [Path("folder/a.ndjson"), Path("folder/b.ndjson.gz")],
        ),
    ],
)
def test__get_file_list(
    init_files: List[Path], exp_files_list: List[Path]
) -> None:
    if len(init_files) == 0:
        return
    if len(init_files) == 1:
        init_files[0].touch()
    else:
        init_files[0].mkdir(exist_ok=False)
        for file_path in init_files[1:]:
            # breakpoint()
            file_path.touch()
    file_list = _get_file_list(init_files[0])

    # tear down
    for file_path in init_files[::-1]:
        if file_path.is_dir():
            file_path.rmdir()
        else:
            file_path.unlink()

    assert file_list == exp_files_list
