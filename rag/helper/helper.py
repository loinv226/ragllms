from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from IPython.display import Markdown
import textwrap
from openai import OpenAI

from llama_index.core.schema import BaseNode
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import PromptTemplate

from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def to_markdown(text: str):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def get_text_embedding(input, index=None):
    if index is not None:
        print(f">> Getting text embedding at index {index}")

    embeddings_batch_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=input
    )
    return embeddings_batch_response.data[0].embedding


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_qeury(query_engine, question):
    response = query_engine.query(question)
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_llm(prompt, model="gpt-3.5-turbo"):
    messages = [
        dict(role="user", content=prompt)
    ]
    chat_response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=messages
    )
    return (chat_response.choices[0].message.content)


def get_router_query_engine(vector_index: VectorStoreIndex, nodes: list[BaseNode] = None, prompt: str = None):
    # reader = ChromaReader(
    #     collection_name=db_collection,
    #     persist_directory=db_path,
    # )

    # query_vector = get_text_embedding("")

    # documents = reader.load_data(
    #     collection_name="demo", query_vector=query_vector, limit=5
    # )
    # summary_index = SummaryIndex.from_documents(documents)

    summary_index = SummaryIndex(
        nodes if nodes else vector_index.storage_context.vector_store.get_nodes())
    # create the summary query engines
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to any llamaindex blog"
        ),
    )

    text_qa_template = PromptTemplate(prompt) if prompt else None
    vector_query_engine = vector_index.as_chat_engine(
        similarity_top_k=5,
        text_qa_template=text_qa_template
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the llamaindex blog."
        ),
    )
    print("Created the tools.")

    # create the router query engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine
