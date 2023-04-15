"""Utility functions that are needed for the entire package."""

import errno
import json
import os
import sys
import time
import urllib.error
from itertools import product
from typing import Dict, Optional, Union

from shifthappens.task_data import task_metadata
from shifthappens.tasks.task_result import TaskResult


def dict_product(d):
    """Computes the product of a dict of sequences."""
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


# taken from https://stackoverflow.com/a/34102855/3157209
def is_pathname_valid(pathname: str) -> bool:
    """
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    """

    # Sadly, Python fails to provide the following magic number for us.
    # Windows-specific error code indicating an invalid pathname.
    # See also:
    # https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
    #     Official listing of all such codes.
    ERROR_INVALID_NAME = 123

    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = (
            os.environ.get("HOMEDRIVE", "C:")
            if sys.platform == "win32"
            else os.path.sep
        )
        assert os.path.isdir(root_dirname)  # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, "winerror"):
                    if exc.winerror == ERROR_INVALID_NAME:  # type: ignore
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?


# from https://github.com/pytorch/vision/blob/0cba9b7845795d6be7b164037461ea6e9265f6a2/torchvision/datasets/utils.py
def download_and_extract_archive(
    url: str,
    data_folder: str,
    md5: str,
    filename: Optional[str],
    remove_finished: bool = False,
    n_retries: int = 3,
) -> None:
    """
    Downloads and extracts and archive using torchvision.

    Args:
        url (str): URL to download.
        data_folder (str): Where to save the downloaded file.
        md5 (str): MD5 hash of the archive.
        filename (str, optional): Name under which the archive will be saved locally.
        remove_finished (bool, optional): Remove archive after extraction?
        n_retries (int): How often to retry the download in case of connectivity issues.
    """

    if not filename:
        filename = os.path.basename(url)

    import torchvision.datasets.utils as tv_utils

    for _ in range(n_retries):
        try:
            tv_utils.download_url(url, data_folder, filename, md5)
            break
        except urllib.error.URLError:
            print(f"Download of {url} failed; wait 5s and then try again.")
            time.sleep(5)

    archive = os.path.join(data_folder, filename)
    print(f"Extracting {archive} to {data_folder}")
    tv_utils.extract_archive(archive, data_folder, remove_finished)

def serialize_model_results(results: Dict[task_metadata.TaskMetadata, Union[TaskResult, None]]) -> str:
    return json.dumps({key.serialize_task_metadata():value.serialize_task_result() for (key,value) in results.items() if value != None})

def deserialize_model_results(results_str) -> Dict[task_metadata.TaskMetadata, TaskResult]:
    results_json_dict = json.loads(results_str)
    results = {}
    for (key,value) in results_json_dict.items():
        results[task_metadata.TaskMetadata.deserialize_task_metadata(key)] = TaskResult.deserialize_task_result(value)
    return results
