import numpy as np
import requests
import pickle
import re
import os
from typing import List
from bs4 import BeautifulSoup
from markdownify import markdownify
from llama_index.core import Document
from helper.helper import get_text_embedding
from helper.blog import Blog

# Set up headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
base_url = "https://www.llamaindex.ai"


def prepare_data(blog_urls: list[str] = None):
    """
    Prepare data, load list blog and cache if not exist.

    Args:
        blog_urls: The blog urls to filter, if None then get all.
    raises:
        requests.RequestException: If there's an error loading the list of blogs.
    """

    print(">> Preparing data...")

    # create folder dump if not exist
    if not os.path.exists('dump'):
        os.makedirs('dump')

    chunk_path = 'dump/chunks.pkl'
    embedding_path = 'dump/embeddings.pkl'

    is_exist_data = os.path.exists(
        chunk_path) and os.path.exists(embedding_path)

    if is_exist_data:
        # Load data from a file
        with open(chunk_path, 'rb') as f:
            chunks = pickle.load(f)

        with open(embedding_path, 'rb') as f:
            text_embeddings = pickle.load(f)

        print(">> Done prepare data")

        return chunks, text_embeddings

    try:
        print(">> Loading list blogs...")
        list_blogs = get_list_blog(blog_urls)
        print(f">> Loaded {len(list_blogs)} blogs")

    except requests.RequestException as e:
        print(f"An error occurred: {e}")

    chunks = []

    print(">> Extract list blogs content...")
    for blog in list_blogs:
        section_contents = get_blog_content(blog.url)
        blog.section_contents = section_contents
        chunks += blog.to_chunks()

    print(">> Saving to file...")
    # Save chunks to a file
    with open(chunk_path, 'wb') as f:
        pickle.dump(chunks, f)

    print(f">> Processing {len(chunks)} embeddings...")
    text_embeddings = np.array(
        [get_text_embedding(chunk, index) for index, chunk in enumerate(chunks)])
    # Save embeddings to a file
    print(">> Save embeddings to file...")
    with open(embedding_path, 'wb') as f:
        pickle.dump(text_embeddings, f)

    print(">> Done prepare data")
    return chunks, text_embeddings


def get_list_blog(filter_urls: list[str] = None):
    """
    Extracts list blog

    Args:
        filter_urls: The blog url to filter, if None then get all.
    Returns:
        A list of blog.
    """

    response = requests.get('https://www.llamaindex.ai/blog',
                            headers=headers, timeout=10)
    response.raise_for_status()

    return extract_list_blog(response.content, filter_urls)


def extract_list_blog(html_content: str, filter_urls: list[str] = None):
    """
    Extracts blog from HTML content.

    Args:
        html_content: The HTML content to extract information from.
        filter_urls: The blog url to filter, if None then get all.

    Returns:
        A list of blogs.
    """

    response = requests.get('https://www.llamaindex.ai/blog',
                            headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(html_content, 'html.parser')

    # declare blog_list type list Blog
    blog_list: List[Blog] = []
    cards = soup.find_all('div', class_='CardBlog_card__mm0Zw')

    for card in cards:
        title_element = card.find(
            'p', class_='CardBlog_title__qC51U').find('a')
        name = title_element.text.strip()
        title = card.find('a', href=True).text.strip()
        url = base_url + title_element['href']
        date = card.find(
            'p', class_='Text_text__zPO0D Text_text-size-16__PkjFu').text.strip()
        if filter_urls and len(filter_urls) > 0:
            if url in filter_urls:
                blog_list.append(
                    Blog(title=title, name=name, date=date, url=url))
        else:
            blog_list.append(Blog(title=title, name=name, date=date, url=url))

    return blog_list


def get_blog_content(url: str):
    """
    Extracts blog content from URL.
    Args:
        url: The URL to extract blog content from.
    Returns:
        Blog information.
    """
    request_blog = requests.get(
        url if url.startswith('http') else base_url + url, headers=headers, timeout=10)
    request_blog.raise_for_status()
    return extract_blog_content(request_blog.content)


def extract_blog_content(html_content: str):
    """
    Extracts blog content from HTML content.

    Args:
        html_content: The HTML content to extract information from.

    Returns:
        Detail blog.
    """

    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.find('h1').get_text(strip=True)
    content_section = soup.find('div', class_=re.compile(r'BlogPost_htmlPost'))

    section_contents = []

    sections = content_section.find_all(['h1', 'h2', 'h3', 'strong'])
    not_exist_child = len(sections) == 0

    if not_exist_child:
        # get content_section string to use in markdownify
        content_section_str = str(content_section)
        markdown_content = markdownify(content_section_str)

        section_contents.append(
            {'header': title, 'content': markdown_content})

    list_items = content_section.find_all('li')
    for li in list_items:
        strong_tag = li.find('strong')
        if strong_tag:
            header = strong_tag.get_text()

            content_li_str = str(li)
            markdown_content = markdownify(content_li_str)
            section_contents.append(
                {'header': header, 'content': markdown_content})

    for section in content_section.find_all(['h1', 'h2', 'h3', 'strong']):
        header = section.get_text()
        section_content = []
        next_sibling = section.find_next_sibling()

        # Extract section contents
        while next_sibling and next_sibling.name not in {'h1', 'h2', 'h3', 'strong'}:
            content_sibling_str = str(next_sibling)
            markdown_content = markdownify(content_sibling_str)

            section_content.append(markdown_content)
            next_sibling = next_sibling.find_next_sibling()

        section_contents.append(
            {'header': header, 'content': '\n'.join(section_content)})

    return section_contents
