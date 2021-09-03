from pathlib import Path
from typing import NoReturn


def path_check(path_to_check: Path, create_new: bool = False) -> NoReturn:

    """
    Check if a path exists. If it doesn't, create the path

    :param path_to_check: Path to check if it exists
    :param create_new: True if new path should be created
    """

    if not path_to_check.exists() and create_new:
        path_to_check.mkdir(exist_ok=True, parents=True)
    elif not path_to_check.exists() and not create_new:
        raise FileNotFoundError


def folder_delete(folder_to_delete: Path) -> NoReturn:

    """
    Delete folder and it's contents

    :param folder_to_delete: Path to the folder to be deleted
    """

    path_check(folder_to_delete)

    for files in folder_to_delete.glob(r"**/*"):
        files.unlink()

    folder_to_delete.rmdir()
