"""Some utils for extracting ObjectNet and mapping the classes."""
import os
import time
import urllib.error
import zipfile
from typing import Optional


# udapted from https://github.com/pytorch/vision/blob/0cba9b7845795d6be7b164037461ea6e9265f6a2/torchvision/datasets/utils.py
def download_and_extract_zip_with_pwd(
    url: str,
    data_folder: str,
    md5: str,
    filename: Optional[str],
    password: Optional[str],
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
        password (str, optional): Archive's password.
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
    with zipfile.ZipFile(archive, "r") as zip_ref:
        if password is not None:
            zip_ref.extractall(data_folder, pwd=bytes(password, "utf-8"))
        else:
            zip_ref.extractall(data_folder)
    if remove_finished:
        os.remove(archive)
