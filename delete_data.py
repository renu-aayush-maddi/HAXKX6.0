

from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


# Connect to the index
index_name = "hackx3072"
index = pc.Index(index_name)

# Delete all vectors
index.delete(delete_all=True)

print(f"All data deleted from index: {index_name}")
