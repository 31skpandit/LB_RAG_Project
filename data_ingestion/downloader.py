"""Fetch source documents used by the RAG pipeline."""
import os
import urllib.request

from config import settings


def download_pdf(url: str = None, dest_path: str = None) -> str:
    """Download a PDF from `url` to `dest_path`, creating parent dirs as needed.

    Returns the local file path. Skips the download if the file already exists.
    """
    url = url or settings.PDF_URL
    dest_path = dest_path or settings.PDF_PATH

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    if os.path.exists(dest_path):
        return dest_path

    urllib.request.urlretrieve(url, dest_path)
    return dest_path
