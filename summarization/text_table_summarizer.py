"""Summarize text and table chunks with an LLM for retrieval indexing."""
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import settings

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


def build_summarize_chain(llm: ChatOpenAI = None):
    llm = llm or ChatOpenAI(model_name=settings.CHATGPT_MODEL, temperature=settings.LLM_TEMPERATURE)
    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    return {"element": RunnablePassthrough()} | prompt | llm | StrOutputParser()


def summarize_texts_and_tables(
    text_docs: List[str], table_docs: List[str], llm: ChatOpenAI = None, max_concurrency: int = 5
):
    """Return (text_summaries, table_summaries) generated via batched LLM calls."""
    summarize_chain = build_summarize_chain(llm)
    text_summaries = summarize_chain.batch(text_docs, {"max_concurrency": max_concurrency}) if text_docs else []
    table_summaries = summarize_chain.batch(table_docs, {"max_concurrency": max_concurrency}) if table_docs else []
    return text_summaries, table_summaries
