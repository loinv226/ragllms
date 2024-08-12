from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import numpy as np
import faiss
from helper.data_store import prepare_data
from helper.helper import get_text_embedding, run_llm


def start_bot():
    # whitelist_urls = [
    #     "https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems",
    #     "https://www.llamaindex.ai/blog/openai-cookbook-evaluating-rag-systems-fe393c61fb93",
    #     "https://www.llamaindex.ai/blog/improving-retrieval-performance-by-fine-tuning-cohere-reranker-with-llamaindex-16c0c1f9b33b"
    # ]
    whitelist_urls = None

    chunks, text_embeddings = prepare_data(whitelist_urls)

    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # create chat bot
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            print("Exiting.")
            break

        question_embeddings = np.array([get_text_embedding(question)])
        D, I = index.search(question_embeddings, k=5)

        retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_chunk}
        ---------------------
        Given the context information and not prior knowledge, answer the query with Blog Url, title, Section and detail answer.
        Query: {question}
        Answer:
        """

        print(run_llm(prompt))
        print("-" * 50)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query(query_engine, question):
    response = query_engine.query(question)
    return response


if __name__ == "__main__":
    start_bot()
