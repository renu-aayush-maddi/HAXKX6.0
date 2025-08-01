import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "hackx3072"  # Your existing index

# Download the fixed document (run once)
document_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
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
