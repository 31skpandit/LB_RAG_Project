"""Summarize images with a multimodal LLM (vision) for retrieval indexing."""
import base64
import glob
import os
from typing import List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from config import settings
from config.llm_factory import get_summary_model
from summarization.cache import get_cached_summary, set_cached_summary

IMAGE_SUMMARY_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
            Remember these images could potentially contain graphs, charts or tables also.
            These summaries will be embedded and used to retrieve the raw image for question answering.
            Give a detailed summary of the image that is well optimized for retrieval.
            Do not add additional words like Summary, This image represents, etc.
         """


def encode_image(image_path: str) -> str:
    """Return the base64-encoded string for an image file."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def summarize_image(img_base64: str, prompt: str = IMAGE_SUMMARY_PROMPT, llm: BaseChatModel = None) -> str:
    llm = llm or get_summary_model()
    msg = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(figures_dir: str = None, llm: BaseChatModel = None) -> Tuple[List[str], List[str]]:
    """Encode every .jpg under `figures_dir` (recursively) and generate a retrieval-oriented summary for each.

    Recursive so this also picks up the per-PDF subdirectories main.py creates
    when indexing multiple PDFs (e.g. figures_dir/<pdf_stem>/*.jpg), not just
    files directly inside figures_dir. Skips re-summarizing images already
    cached from a previous run (see summarization/cache.py).

    Returns (base64_images, image_summaries), both in path-sorted order.
    """
    figures_dir = figures_dir or settings.FIGURES_DIR

    img_base64_list = []
    image_summaries = []

    if not os.path.isdir(figures_dir):
        return img_base64_list, image_summaries

    provider = settings.SUMMARY_PROVIDER
    model = settings.SUMMARY_GEMINI_MODEL if provider == "gemini" else settings.SUMMARY_OPENAI_MODEL
    cached_count = 0

    for img_path in sorted(glob.glob(os.path.join(figures_dir, "**", "*.jpg"), recursive=True)):
        base64_image = encode_image(img_path)
        img_base64_list.append(base64_image)

        cached = get_cached_summary(base64_image, provider, model)
        if cached is not None:
            cached_count += 1
            image_summaries.append(cached)
        else:
            summary = summarize_image(base64_image, llm=llm)
            set_cached_summary(base64_image, provider, model, summary)
            image_summaries.append(summary)

    if cached_count:
        print(f"  {cached_count} image summaries reused from cache, {len(img_base64_list) - cached_count} newly summarized")

    return img_base64_list, image_summaries
