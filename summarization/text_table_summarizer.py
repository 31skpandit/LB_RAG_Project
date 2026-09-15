"""Summarize text and table chunks with an LLM for retrieval indexing."""
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import settings
from config.llm_factory import get_summary_model
from summarization.cache import get_cached_summary, set_cached_summary

SUMMARY_PROMPT = """
You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
These summaries will be embedded and used to retrieve the raw text or table elements
Give a detailed summary of the table or text below that is well optimized for retrieval.
For any tables also add in a one line description of what the table is about besides the summary.
Do not add redundant words like Summary.
Just output the actual summary content.

Table or text chunk:
{element}
"""


def build_summarize_chain(llm: BaseChatModel = None):
    llm = llm or get_summary_model()
    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    return {"element": RunnablePassthrough()} | prompt | llm | StrOutputParser()


def _summarize_with_cache(chain, docs: List[str], max_concurrency: int) -> List[str]:
    """Summarize `docs`, reusing cached summaries for content already seen
    before (see summarization/cache.py) and only calling the LLM for
    genuinely new/changed content.
    """
    if not docs:
        return []

    provider = settings.SUMMARY_PROVIDER
    model = settings.SUMMARY_GEMINI_MODEL if provider == "gemini" else settings.SUMMARY_OPENAI_MODEL

    results = [None] * len(docs)
    to_summarize, to_summarize_idx = [], []
    for i, doc in enumerate(docs):
        cached = get_cached_summary(doc, provider, model)
        if cached is not None:
            results[i] = cached
        else:
            to_summarize.append(doc)
            to_summarize_idx.append(i)

    cached_count = len(docs) - len(to_summarize)
    if cached_count:
        print(f"  {cached_count} summaries reused from cache, {len(to_summarize)} newly summarized")

    if to_summarize:
        new_summaries = chain.batch(to_summarize, {"max_concurrency": max_concurrency})
        for idx, doc, summary in zip(to_summarize_idx, to_summarize, new_summaries):
            results[idx] = summary
            set_cached_summary(doc, provider, model, summary)

    return results


def summarize_texts_and_tables(
    text_docs: List[str], table_docs: List[str], llm: BaseChatModel = None, max_concurrency: int = 1
):
    """Return (text_summaries, table_summaries) generated via batched LLM calls.

    Skips re-summarizing content already cached from a previous run (see
    summarization/cache.py) -- restarting the pipeline on unchanged documents
    won't re-burn tokens re-summarizing the same chunks every time.
    """
    summarize_chain = build_summarize_chain(llm)
    text_summaries = _summarize_with_cache(summarize_chain, text_docs, max_concurrency)
    table_summaries = _summarize_with_cache(summarize_chain, table_docs, max_concurrency)
    return text_summaries, table_summaries
