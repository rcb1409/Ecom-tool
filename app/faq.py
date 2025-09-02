import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

csv_path = Path(__file__).parent / "resources/faq_data.csv"
chroma_client = chromadb.Client()
collection_name_faq = 'faqs'
groq_client = Groq()
ef = SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.get_or_create_collection(
        name=collection_name_faq,
        embedding_function=ef
    )

        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]

        ids = [f"id_{i}" for i in range(len(docs))]
        collection.add(
        documents=docs,
        metadatas=metadata,
        ids=ids
    )
        print(f"FAQ Data Succesfully ingesteds into Chroma Collection: {collection_name_faq}")
    else:
        print(f"Collection {collection_name_faq} already exists")    


def get_relevant_qa(query):
    collection = chroma_client.get_collection(name=collection_name_faq)
    result = collection.query(
        query_texts= [query],
        n_results=2
    )
    return result


def faq_chain(query):
    result = get_relevant_qa(query)
    context = ''.join([r.get('answer') for r in result['metadatas'][0]])

    prompt = f'''
    Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {query}

    '''
    completion = groq_client.chat.completions.create(
    model= os.environ['GROQ_MODEL'],
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ],
)
    return completion.choices[0].message.content


if __name__ == "__main__":
    ingest_faq_data(csv_path)
    query = "what's your policy on defective products"
    answer = faq_chain(query)
    print("Answer:", answer)