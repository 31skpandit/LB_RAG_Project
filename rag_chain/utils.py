"""Display/decoding helpers for base64 images returned by the retriever."""
import base64
from io import BytesIO

from PIL import Image


def plt_img_base64(img_base64: str) -> Image.Image:
    """Decode a base64 string into a PIL Image and display it (in notebooks/IPython)."""
    img_data = base64.b64decode(img_base64)
    img_buffer = BytesIO(img_data)
    img = Image.open(img_buffer)
    try:
        from IPython.display import display

        display(img)
    except ImportError:
        img.show()
    return img
