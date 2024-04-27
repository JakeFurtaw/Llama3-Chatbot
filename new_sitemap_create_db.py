from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import os
import shutil
import torch
import re
import fnmatch

SITEMAP_URL = 'https://www.towson.edu/sitemap.xml'
CHROMA_PATH = 'TowsonDBAlt'
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

def main():
    documents = load_docs()
    cleaned_docs_at_load = parse_docs_at_load(documents)
    cleaned_docs = parse_docs(cleaned_docs_at_load)
    chunks = split_pages(cleaned_docs)
    save_to_db(chunks)
    
def load_docs():
    print("Loading documents from " + SITEMAP_URL)
    loader = SitemapLoader(SITEMAP_URL, continue_on_failure=True, parsing_function=parse_docs_at_load)
    documents = loader.load()
    print("Number of documents loaded: " + str(len(documents)))
    return documents

def parse_docs_at_load(documents):
    cleaned_docs_at_load = []
    for doc in documents:
        soup = BeautifulSoup(doc, 'html.parser')
        for div in soup.select('div#skip-to-main, div.row, div.utility, div.main, div.mobile, div.links, div.secondary, div.bottom, div.sidebar, nav.subnavigation, div#subnavigation, div.subnavigation, div.sidebar'):
            div.decompose()
        for noscript_tag in soup.find_all('noscript'):
            noscript_tag.decompose()
        cleaned_text = soup.get_text(strip=True, separator=" ")
        cleaned_docs_at_load.append(cleaned_text)
    return cleaned_docs_at_load

def parse_docs(cleaned_docs_at_load):
    print("Cleaning documents...")
    cleaned_docs = []
    for doc in cleaned_docs_at_load:
        content = doc.page_content
        cleaned_text = re.sub(r'[\s\n\r\t]+', ' ', content)
        soup = BeautifulSoup(cleaned_text, 'html.parser')
        cleaned_text = soup.get_text(strip=True, separator=" ")
        cleaned_docs.append(cleaned_text)
    print("Number of documents cleaned: " + str(len(cleaned_docs)))
    return cleaned_docs

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