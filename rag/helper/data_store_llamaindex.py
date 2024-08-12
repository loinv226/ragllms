import os
import requests
from markdownify import markdownify
import chromadb

from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode
from llama_index.core import Document
from llama_index.core import StorageContext, VectorStoreIndex

from config import OPENAI_API_KEY
from helper.blog import Blog
from helper.data_store import get_list_blog

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
db_path = "./llamaindex_db"
db_collection = "llamaindex_blogs"


def prepare_data(blog_urls: list[str] = None, qeury_vector: list[int] = None):
    """
    Prepare data, load list blog and cache if not exist.

    Args:
        blog_urls: The blog urls to filter, if None then get all.
    return:
        vector_index: VectorStoreIndex object
    raises:
        requests.RequestException: If there's an error loading the list of blogs.
    """

    print(">> Preparing data...")

    is_exist_data = os.path.exists(db_path)

    nodes: list[BaseNode] = None

    if not is_exist_data:
        try:
            print(">> Loading list blogs...")
            list_blogs = get_list_blog(blog_urls)
            print(f">> Loaded {len(list_blogs)} blogs")

        except requests.RequestException as e:
            print(f"An error occurred: {e}")

        print(">> Extract list blogs content...")

        documents = extract_documents(list_blogs)

        # get nodes from documents
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        print(">> [Debug] nodes: ", len(nodes))
        print('>> [Debug] nodes[-1]: ', nodes[-1].text)

    # set chromadb to store data
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context) if not is_exist_data else VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    # print(">> Saving to file...")
    # index.storage_context.persist()

    print(">> Done prepare data")
    # return get_router_query_engine(vector_index, nodes)

    return vector_index


def extract_documents(blogs: list[Blog]) -> list[Document]:
    """
    Extract documents from list blog using SimpleWebPageReader.

    Args:
        blogs (List[Blog]): List of Blog objects.

    Returns:
        List[Document]: List of Document objects created from the blogs.

    Raises:
        Exception: If there's an error extracting content from a URL.
    """
    reader = SimpleWebPageReader()
    documents: list[Document] = []

    for blog in blogs:
        try:
            # section_contents = get_blog_content(blog.url)
            # for section_content in section_contents:
            #     documents.append(Document(text=section_content['content'], metadata={
            #         'Title': section_content['header'],
            #         'Date': blog.date,
            #         'URL': blog.url,
            #     }))

            loaded_documents = reader.load_data([blog.url])
            if loaded_documents:
                doc = loaded_documents[0]
                doc.text = to_markdown(doc.text)
                doc.metadata.update({
                    'Blog_Title': blog.title,
                    'Blog_Date': blog.date,
                    'Blog_URL': blog.url,
                })
                documents.append(doc)
            else:
                print(f"No content extracted from {blog.url}")
        except Exception as e:
            print(f"Error extracting content from {blog.url}: {e}")

    return documents


def to_markdown(html_text: str) -> str:
    """
    Convert HTML text to Markdown format only for the content part.

    Args:
        html_text (str): The HTML text to convert.

    Returns:
        str: The blog markdown content.
    """

    markdown_text = markdownify(html_text, heading_style="ATX")

    # Split the text into lines to find the content part
    lines = markdown_text.split('\n')

    # Find the index of the first line starting with '#'
    start_content_index = next((i for i, line in enumerate(lines)
                                if line.startswith('#')), 0)

    # end_content_index at the line "## Related articles" or the end of the text
    end_content_index = next((i for i, line in enumerate(
        lines) if line.strip() == "## Related articles"), len(lines))

    # Content lines are the lines between start_content_index and end_content_index
    content_lines = []
    for line in lines[start_content_index:end_content_index]:
        line = line.strip()
        if line:
            content_lines.append(line)

    blog_md_content = '\n'.join(content_lines)

    return blog_md_content
