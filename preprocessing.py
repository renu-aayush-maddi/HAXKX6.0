import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "test"  # Your existing index

# Download the fixed document (run once)
document_url = "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D"
import requests, tempfile
response = requests.get(document_url)
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    temp_file.write(response.content)
    temp_filepath = temp_file.name

# Process and embed (using your functions)
embeddings = download_hugging_face_embeddings()
extracted_data = load_pdf_file(temp_filepath)
chunks = text_split(extracted_data, temp_filepath)  # Your semantic chunker

# Upsert to Pinecone (persistent)
pc = Pinecone(api_key=PINECONE_API_KEY)
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
)

# Clean up
os.remove(temp_filepath)
print("Embeddings pre-stored in Pinecone!")
