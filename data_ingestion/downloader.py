"""Fetch source documents used by the RAG pipeline."""
import glob
import os
import urllib.parse
import urllib.request
from typing import List

from config import settings


# Formats routed through data_ingestion.document_loader (via Unstructured's
# auto-detecting partition()). Image extensions are included as standalone
# documents (OCR'd by Unstructured), separate from images *extracted from*
# PDFs during partitioning.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".html", ".htm", ".png", ".jpg", ".jpeg"}


def discover_documents(data_dir: str = None) -> List[str]:
    """Return every supported document directly inside `data_dir`, sorted for deterministic order.

    This is main.py's default (no --pdf-path/--pdf-url) way to pick up
    whatever documents have been placed in the data folder.
    """
    data_dir = data_dir or settings.DATA_DIR
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
    return sorted(paths)


def download_pdf(url: str = None, dest_path: str = None) -> str:
    """Download a PDF from `url` to `dest_path`, creating parent dirs as needed.

    Returns the local file path. Skips the download if the file already exists.
    """
    url = url or settings.PDF_URL

    if dest_path is None:
        # Only use the hardcoded sample path for the actual default sample URL;
        # otherwise derive the filename from the URL so a custom --pdf-url
        # doesn't silently overwrite the sample file at PDF_PATH.
        if url == settings.PDF_URL:
            dest_path = settings.PDF_PATH
        else:
            filename = os.path.basename(urllib.parse.urlparse(url).path) or "downloaded.pdf"
            dest_path = os.path.join(settings.DATA_DIR, filename)

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    # Some hosts (e.g. sgp.fas.org) reject requests without a browser-like User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    return dest_path
