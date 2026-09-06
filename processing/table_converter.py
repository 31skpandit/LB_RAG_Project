"""Convert HTML tables (as extracted by Unstructured) into Markdown tables."""
from typing import List

import htmltabletomd
from langchain_core.documents import Document


def html_table_to_markdown(html: str) -> str:
    return htmltabletomd.convert_table(html)


def convert_tables_to_markdown(tables: List[Document]) -> List[Document]:
    """Mutate each table Document's `page_content` in place to Markdown, returning the list."""
    for table in tables:
        table.page_content = html_table_to_markdown(table.metadata["text_as_html"])
    return tables
