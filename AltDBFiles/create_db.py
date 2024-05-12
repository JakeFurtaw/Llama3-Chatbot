import os
import trafilatura
import httpx
import torch

from tqdm import tqdm
from multiprocessing import Pool
from parsel import Selector
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document, StorageContext, VectorStoreIndex
from bs4 import BeautifulSoup

#Change this to the sitemap link of the schools sitemap you want to scrape
SITEMAP_LINK = "https://www.towson.edu/sitemap.xml"


def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    return device

def set_embed_model(device):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=device)
    Settings.embed_model = embed_model

def parse_sitemap(output_dir):
    sitemap_url_path = SITEMAP_LINK
    response = httpx.get(sitemap_url_path)
    selector = Selector(response.text)
    urls = []
    pdfs = []
    for url in selector.xpath('//url'):
        location = url.xpath('loc/text()').get()
        modified = url.xpath('lastmod/text()').get()
        if ".pdf" not in location:
            urls.append(location)
        else:
            pdfs.append(location)
    write_list_to_file(output_dir=output_dir, file_name="urls.txt", data=urls)
    write_list_to_file(output_dir=output_dir, file_name="pdfs.txt", data=pdfs)

def write_list_to_file(output_dir, file_name, data):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, file_name), 'w') as f:
        for item in data:
            f.write(str(item) + '\n')

def read_web_pages(path):
    urls = []
    with open(path) as f:
        for url in f.readlines():
            urls.append(url.strip())
    documents = load_data(urls=urls)
    return documents

def load_data(urls,
              include_comments=True,
              output_format="txt",
              include_tables=True,
              include_images=False,
              include_formatting=False,
              include_links=False):
    documents = []
    pool_args = [
        (url, include_comments, output_format, include_tables, include_images, include_formatting, include_links) for
        url in urls]

    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(process_url, pool_args), total=len(urls), desc="Fetching documents..."):
            if result is not None:
                documents.append(result)
    return documents

def process_url(args):
    # Unpack all arguments from the single tuple argument
    url, include_comments, output_format, include_tables, include_images, include_formatting, include_links = args
    downloaded = trafilatura.fetch_url(url)

    soup = BeautifulSoup(downloaded, "html.parser")
    selectors = ['div#skip-to-main', 'div.row', 'div.utility', 'div.main', 'div.mobile', 'div.links', 'div.secondary', 'div.bottom', 'div.sidebar', 'nav.subnavigation', 'div#subnavigation', 'div.subnavigation', 'div.sidebar']
    for div in soup.find_all(' , '.join(selectors)) + soup.find_all('noscript'):
        div.decompose()
    souped_text = soup.get_text(strip=True, separator=" ")
    downloaded = str(souped_text)

    response = trafilatura.extract(
        downloaded,
        include_comments=include_comments,
        output_format=output_format,
        include_tables=include_tables,
        include_images=include_images,
        include_formatting=include_formatting,
        include_links=include_links,
    )
    if response is None:
        print(f"{url} is empty")
        return None
    else:
        return Document(text=response, id_=url)

def create_db():
    documents = read_web_pages(path="data/urls.txt")
    vector_store = ChromaVectorStore(collection_name="TowsonData", device=device, overwrite=True)
    context = StorageContext.from_defaults(vector_stores=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=context)
    return index

if __name__ == "__main__":
    device = set_device()
    set_embed_model(device)
    if len(os.listdir("data")) == 0:
        parse_sitemap(output_dir="data")
    else:
        print("Data already exists")
    create_db()