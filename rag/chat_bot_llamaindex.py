from helper.data_store_llamaindex import prepare_data
from helper.helper import get_router_query_engine, run_qeury
from config import OPENAI_API_KEY

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptTemplate
from llama_index.core import Settings

Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(
    model_name='text-embedding-ada-002', api_key=OPENAI_API_KEY)


def start_bot():
    # whitelist_urls = [
    #     "https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems",
    #     "https://www.llamaindex.ai/blog/openai-cookbook-evaluating-rag-systems-fe393c61fb93",
    #     "https://www.llamaindex.ai/blog/improving-retrieval-performance-by-fine-tuning-cohere-reranker-with-llamaindex-16c0c1f9b33b"
    # ]
    whitelist_urls = None

    vector_index = prepare_data(whitelist_urls)

    # create chat bot
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            print("Exiting.")
            break

        # create the vector query engines
        prompt = """
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query with Blog Url, title and detail answer (note: show detail in code if exist).
        Question: {query_str}
        Answer:
        """

        # query_engine = get_router_query_engine(vector_index, prompt=prompt)
        text_qa_template = PromptTemplate(prompt)

        query_engine = vector_index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=text_qa_template,
        )

        # Query the index
        response = run_qeury(query_engine, question)

        # Display the answer
        print("Answer:\n")
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    start_bot()
