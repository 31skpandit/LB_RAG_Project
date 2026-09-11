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

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    # Some hosts (e.g. sgp.fas.org) reject requests without a browser-like User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    return dest_path
