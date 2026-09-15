"""End-to-end multimodal RAG chain: retrieve context, then synthesize an answer with the configured LLM."""
from operator import itemgetter

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.llm_factory import get_chat_model
from processing import split_image_text_types
from rag_chain.prompts import ANALYST_SYSTEM_INSTRUCTIONS
from rag_chain.utils import plt_img_base64


def multimodal_prompt_function(data_dict):
    """Build a HumanMessage with image parts + a text part combining context and question."""
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    for image in data_dict["context"]["images"]:
        messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})

    text_message = {
        "type": "text",
        "text": ANALYST_SYSTEM_INSTRUCTIONS.format(question=data_dict["question"], context=formatted_texts),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def build_multimodal_rag_chain(retriever, llm: BaseChatModel = None):
    """Return a runnable that accepts {'input': question} and yields {..., 'context', 'answer', 'citations'}.

    `citations` is assembled deterministically in code from retrieved-chunk
    metadata (see processing.split_image_text_types / processing.citations) --
    not left to the LLM to remember to include -- so it's always present.
    """
    llm = llm or get_chat_model()

    multimodal_rag = (
        {"context": itemgetter("context"), "question": itemgetter("input")}
        | RunnableLambda(multimodal_prompt_function)
        | llm
        | StrOutputParser()
    )

    retrieve_docs = itemgetter("input") | retriever | RunnableLambda(split_image_text_types)

    return RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=multimodal_rag,
        citations=itemgetter("context") | RunnableLambda(lambda ctx: ctx.get("citations", [])),
    )


def multimodal_rag_qa(chain, query: str, display_output: bool = True):
    """Invoke the chain and optionally pretty-print the answer + sources (for notebooks)."""
    response = chain.invoke({"input": query})

    if display_output:
        try:
            from IPython.display import Markdown, display
        except ImportError:
            display = None

        print("==" * 50)
        print("Answer:")
        if display:
            display(Markdown(response["answer"]))
        else:
            print(response["answer"])

        print("--" * 50)
        print("Citations:")
        for citation in response["citations"]:
            print(f"- {citation}")

        print("--" * 50)
        print("Sources:")
        for text in response["context"]["texts"]:
            if display:
                display(Markdown(text))
            else:
                print(text)
            print()
        for img in response["context"]["images"]:
            plt_img_base64(img)
            print()
        print("==" * 50)

    return response
