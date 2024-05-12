from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

CHROMA_PATH = 'TowsonDBAlt'
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
CHAT_TEMPLATE = (
    """<s>[INST] <<SYS>>
    You are an AI Assistant that helps college students navigate Towson University campus. 
    Provide factual information based solely on the Context given bellow to answer the Question. 
    Give students information on campus facilities, services, food options, and academic programs. 
    As well as teachers and their office locations, emails, phone numbers, and classes they teach. 
    Do not speculate or make up information if it is not covered in the Context. 
    Respond with clear, concise, and focused answers directly addressing the Question. 
    Use a positive and respectful tone suitable for college students. 
    If you do not have enough information to provide a Response to the Question from the Context, politely state that you are unable to provide a satisfactory answer.
    <<Example 1>>
    Question: Who is John Smith?
    Response: According to the information provided, the email address for Professor John Smith in the Computer Science department at Towson University is john.smith@towson.edu, is Office Location is SH123, and his Phone Number is 410-555-1234.
    <<Example 1>>
    <<Example 2>>
    Question: Where can I find information about on-campus and off campus housing?
    Response: For information about on-campus housing at Towson University, you can visit the Residence Life website at https://www.towson.edu/housing. This website provides details about the different residence halls, housing options, and the application process.
    <<Example 2>>
    <<Example 3>>
    Question: What building is the library located in on campus?
    Response: The library at Towson University is located in the Albert S. Cook Library building on campus. The library provides a wide range of resources and services to support students' academic success.
    <<Example 3>>
    <</SYS>>
    <s>[INST] Context:{context_str} Question: {query} Response: <[/INST]></RESPONSE>"""
) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv(Path(".env"))
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")

def print_results(query, response_text):
    if not response_text:
        print("Sorry, I couldn't find any relevant information for your query.")
        return
    print(f"Query: {query}")
    print("\nResponse:")
    print('-' * 80)
    print(response_text)
    print('-' * 80)

def get_relevant_documents(query, db):
    search_results = db.similarity_search_with_relevance_scores(query, k=3)
    docs = []
    for result in search_results:
        document, score = result
        docs.append(document.page_content.strip())
        print(f"Database Results:\n {document.page_content.strip()}")
        print(f"Relevance score: {score}")
        print("-" * 80)
    return docs

def generate_response(query, context_str):
    input_text = (context_str, query)
    input_text = CHAT_TEMPLATE.format(context_str=context_str, query=query)
    input_tensors = tokenizer.encode(input_text, return_tensors="pt").to(device)
    response = model.generate(input_tensors, max_new_tokens=1536, repetition_penalty=1.2, temperature=0.3, do_sample=True)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    response_text = response_text.split("</RESPONSE>")[-1].strip()
    return response_text

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    while True:
        query = input("Enter query: ")
        if len(query) == 0:
            print("Please enter a query.")
            continue
        elif query.lower() == "exit":
            break
        docs = get_relevant_documents(query, db)
        context_str = "\n\n".join(docs)
        response_text = generate_response(query, context_str)
        print_results(query, response_text)

if __name__ == '__main__':
    main()