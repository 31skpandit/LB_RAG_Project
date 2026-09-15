"""Convert HTML tables (as extracted by Unstructured) into Markdown tables."""
from typing import List

import htmltabletomd
from langchain_core.documents import Document


def html_table_to_markdown(html: str) -> str:
    return htmltabletomd.convert_table(html)


def convert_tables_to_markdown(tables: List[Document]) -> List[Document]:
    """Mutate each table Document's `page_content` in place to Markdown, returning the list.

    `text_as_html` is reliably populated for PDF tables; table support on
    newer formats (docx/pptx/html) added alongside broad multi-format
    ingestion is less battle-tested, so fall back to the existing plain-text
    `page_content` if it's missing rather than raising.
    """
    for table in tables:
        html = table.metadata.get("text_as_html")
        if html:
            table.page_content = html_table_to_markdown(html)
    return tables
