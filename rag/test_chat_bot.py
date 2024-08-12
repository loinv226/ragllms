
from llama_index.core import PromptTemplate
from datetime import datetime, timezone
import faiss
import numpy as np
import unittest

from helper.data_store_llamaindex import prepare_data as prepare_data_llamaindex
from helper.data_store import get_list_blog, get_blog_content, prepare_data
from helper.helper import get_text_embedding, run_llm


class TestContainsValidField(unittest.TestCase):
    target_blog_url = 'https://www.llamaindex.ai/blog/improving-retrieval-performance-by-fine-tuning-cohere-reranker-with-llamaindex-16c0c1f9b33b'

    def test_contains_blog(self):
        list_blogs = get_list_blog()
        target_blog_url = 'https://www.llamaindex.ai/blog/llamaindex-newsletter-2024-07-30'

        self.assertTrue(
            any(blog.url == target_blog_url for blog in list_blogs))

    def test_should_extract_blog_content(self):
        target_blog_url = 'https://www.llamaindex.ai/blog/improving-retrieval-performance-by-fine-tuning-cohere-reranker-with-llamaindex-16c0c1f9b33b'
        blog_sections = get_blog_content(target_blog_url)

        self.assertTrue(len(blog_sections) > 0)
        # self.assertTrue(
        #     any("Structured Extraction for LLM-powered Pipelines" in section.get('header') for section in blog_sections))
        # self.assertTrue(
        #     any("Feature Releases and Enhancements" in section.get('header') for section in blog_sections))

    def test_custom_chunking(self):
        # clear cache
        # if os.path.exists('chunks.pkl'):
        #     os.remove('chunks.pkl')
        # if os.path.exists('embeddings.pkl'):
        #     os.remove('embeddings.pkl')

        # log start time for performance testing
        start_time = datetime.now(timezone.utc)

        chunks, text_embeddings = prepare_data(
            [
                "https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems",
                "https://www.llamaindex.ai/blog/openai-cookbook-evaluating-rag-systems-fe393c61fb93",
                "https://www.llamaindex.ai/blog/improving-retrieval-performance-by-fine-tuning-cohere-reranker-with-llamaindex-16c0c1f9b33b"
            ])

        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)

        question = "What are the two main metrics used to evaluate the performance of the different rerankers in the RAG system?"

        question_embeddings = np.array([get_text_embedding(question)])
        D, I = index.search(question_embeddings, k=5)

        retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
        self.assertTrue(len(retrieved_chunk) > 0)

        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_chunk}
        ---------------------
        Given the context information and not prior knowledge, answer the query with Blog Url, title, Section and detail answer.
        Query: {question}
        Answer:
        """

        response = run_llm(prompt)
        print('response: ', response)

        # log end time for performance testing
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        print(f'Time taken: {duration.seconds} seconds')

        self.assertTrue(response is not None)
        self.assertTrue("hit_rate" in response or "Hit Rate" in response)
        self.assertTrue("mrr" in response or "MRR" in response)

    def test_chatbot_use_llamaindex(self):
        # log start time for performance testing
        start_time = datetime.now(timezone.utc)

        vector_index = prepare_data_llamaindex(
            [
                "https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems",
                "https://www.llamaindex.ai/blog/openai-cookbook-evaluating-rag-systems-fe393c61fb93",
                "https://www.llamaindex.ai/blog/improving-retrieval-performance-by-fine-tuning-cohere-reranker-with-llamaindex-16c0c1f9b33b"
            ])

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

        question = "What are the two main metrics used to evaluate the performance of the different rerankers in the RAG system?"
        # Query the index
        response = query_engine.query(question)
        # print('response: ', response)

        # log end time for performance testing
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        print(f'Time taken: {duration.seconds} seconds')

        self.assertTrue(response is not None)
        self.assertTrue("Hit Rate" in str(response))
        self.assertTrue("MRR" in str(response))
