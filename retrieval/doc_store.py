"""Redis doc store setup — stores raw text/table/image content for retrieval."""
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client

from config import settings


def get_doc_store(redis_url: str = None) -> RedisStore:
    client = get_client(redis_url or settings.REDIS_URL)
    return RedisStore(client=client)
