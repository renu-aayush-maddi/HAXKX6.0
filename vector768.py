import logging

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



from pinecone import ServerlessSpec

def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "hackx3072"  # New name
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,  # For new embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    logging.info(f"Index '{index_name}' ready.")

# Call at top of app.py
create_pinecone_index()