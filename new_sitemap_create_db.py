from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
from multiprocessing import Pool
import os
import shutil
import torch

SITEMAP_URL = 'https://www.towson.edu/sitemap.xml'
CHROMA_PATH = 'TowsonDBAlt'
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
PATH_TO_URLS = './URLList/urls.txt'

def main():
    urls= get_urls(PATH_TO_URLS)
    documents = load_docs(urls)
    cleaned_docs = parse_docs(documents)
    write_cleaned_docs_to_file(cleaned_docs)
    chunks = split_pages(cleaned_docs)
    save_to_db(chunks)

def get_urls(file_path):
    urls = []
    with open(file_path, 'r') as file:
        urls= file.read().splitlines()
    print(f"Number of URLs loaded: {len(urls)}")
    return urls

def load_docs_worker(urls):
    loader = SitemapLoader(SITEMAP_URL, filter_urls=urls, continue_on_failure=True, parsing_function=parse_docs)
    documents = loader.load()
    return documents

def load_docs(urls):
    print("Loading documents from " + SITEMAP_URL)
    chunks = [urls[i::20] for i in range(20)]
    pool = Pool(processes=20)
    results = pool.map(load_docs_worker, chunks)
    documents = [doc for result in results for doc in result]
    print("Number of documents loaded: " + str(len(documents)))
    return documents

def parse_docs(content: BeautifulSoup) -> str:
    selectors = ['div#skip-to-main', 'div.row', 'div.utility', 'div.main', 'div.mobile', 'div.links', 'div.secondary', 'div.bottom', 'div.sidebar', 'nav.subnavigation', 'div#subnavigation', 'div.subnavigation', 'div.sidebar']
    for div in content.find_all(' , '.join(selectors)) + content.find_all('noscript'):
        div.decompose()
    souped_text = content.get_text(strip=True, separator=" ")
    print(souped_text)
    return str(souped_text)

def write_cleaned_docs_to_file(cleaned_docs):
    with open("cleaned_docs.txt", "w", encoding="utf-8") as file:
        for doc in cleaned_docs:
            file.write(doc + "\n\n")
    print(f"Cleaned documents written to 'cleaned_docs.txt'.")

def split_pages(cleaned_docs):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.create_documents(cleaned_docs)
    print("Number of chunks created: " + str(len(chunks)))
    return chunks

def save_to_db(chunks):
    # Clear the database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = HuggingFaceEmbeddings(
                 model_name=EMBEDDING_MODEL, 
                 model_kwargs={"device": device}
    )
    print("Creating Chroma database....")
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print("Chroma database created at " + CHROMA_PATH)

if __name__ == '__main__':
    main()