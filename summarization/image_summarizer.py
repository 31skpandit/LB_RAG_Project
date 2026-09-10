"""Summarize images with a multimodal LLM (vision) for retrieval indexing."""
import base64
import os
from typing import List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from config import settings
from config.llm_factory import get_chat_model

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
    llm = llm or get_chat_model()
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
    """Encode every .jpg in `figures_dir` and generate a retrieval-oriented summary for each.

    Returns (base64_images, image_summaries), both in filename-sorted order.
    """
    figures_dir = figures_dir or settings.FIGURES_DIR

    img_base64_list = []
    image_summaries = []

    for img_file in sorted(os.listdir(figures_dir)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(figures_dir, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(summarize_image(base64_image, llm=llm))

    return img_base64_list, image_summaries
