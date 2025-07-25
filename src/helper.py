# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings


# from langchain.text_splitter import RecursiveCharacterTextSplitter


# #Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)

#     documents=loader.load()

#     return documents



# #Split the Data into Text Chunks
# def text_split(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks



# #Download the Embeddings from HuggingFace 
# def download_hugging_face_embeddings():
#     embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
#     return embeddings


# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# import pdfplumber
# import re , os
# from unstructured.partition.pdf import partition_pdf  # New import
# def load_pdf_file(data):
#     documents = []
#     if os.path.isfile(data):  # Check if it's a single file
#         with pdfplumber.open(data) as pdf:
#             for page_num, page in enumerate(pdf.pages, start=1):
#                 page_text = page.extract_text() or ""
#                 tables = page.extract_tables()
#                 table_text = ""
#                 for table in tables:
#                     markdown_rows = [" | ".join(str(cell) if cell is not None else "" for cell in row).strip() for row in table if any(row)]
#                     table_text += "\n".join(markdown_rows) + "\n"
#                 combined_content = f"{page_text.strip()}\n\n{table_text.strip()}".strip()
#                 documents.append(Document(
#                     page_content=combined_content,
#                     metadata={"source": os.path.basename(data), "page": page_num}
#                 ))
#     else:  # Fallback to directory mode if needed
#         for filename in os.listdir(data):
#             if filename.endswith(".pdf"):
#                 filepath = os.path.join(data, filename)
#                 with pdfplumber.open(filepath) as pdf:
#                     for page_num, page in enumerate(pdf.pages, start=1):
#                         page_text = page.extract_text() or ""
#                         tables = page.extract_tables()
#                         table_text = ""
#                         for table in tables:
#                             markdown_rows = [" | ".join(str(cell) if cell is not None else "" for cell in row).strip() for row in table if any(row)]
#                             table_text += "\n".join(markdown_rows) + "\n"
#                         combined_content = f"{page_text.strip()}\n\n{table_text.strip()}".strip()
#                         documents.append(Document(
#                             page_content=combined_content,
#                             metadata={"source": filename, "page": page_num}
#                         ))
#     return documents
# # Helper to extract clause or section references (your existing function, unchanged)
# def extract_clause(text):
#     match = re.search(r'(Clause|Section|Article)\s+[\w.-]+', text, re.IGNORECASE)
#     return match.group(0) if match else None
# # Updated text_split: Structure-aware splitting
# def text_split(extracted_data, pdf_path):  # Now takes pdf_path for unstructured partitioning
#     # Step 1: Use unstructured to partition the PDF into elements (headings, paragraphs, tables, etc.)
#     try:
#         elements = partition_pdf(
#             filename=pdf_path,
#             strategy="hi_res",  # High-res for better table/structure detection
#             infer_table_structure=True,  # Extracts tables accurately
#             languages=["eng"],  # Assuming English docs
#         )
#     except Exception as e:
#         raise ValueError(f"Error partitioning PDF: {str(e)}")
    
#     # Step 2: Convert elements to LangChain Documents, preserving metadata
#     documents = []
#     for el in elements:
#         metadata = el.metadata.to_dict() if hasattr(el, 'metadata') else {}
#         metadata["type"] = el.category  # e.g., 'Title', 'NarrativeText', 'Table'
#         documents.append(Document(page_content=str(el), metadata=metadata))
    
#     # Step 3: Split large elements recursively, respecting policy structures
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,  # Slightly larger to fit full clauses (adjust based on doc length)
#         chunk_overlap=200,  # More overlap to prevent mid-clause breaks
#         separators=[
#             "\n\n",  # Paragraph breaks
#             "\n",    # Line breaks
#             "Clause ", "Section ", "Article ",  # Policy-specific separators
#             ".",     # Sentence ends
#             " "      # Fallback
#         ],
#         keep_separator=True  # Keeps separators in chunks for context
#     )
#     chunks = text_splitter.split_documents(documents)
    
#     # Step 4: Enhance metadata for better retrieval and explainability
#     for chunk in chunks:
#         clause = extract_clause(chunk.page_content)
#         chunk.metadata["clause"] = clause if clause else "Unknown"
#         chunk.metadata["contains_table"] = ("|" in chunk.page_content or "-" in chunk.page_content)  # Detects Markdown tables
        
#         # Add page and source if available from unstructured
#         if "page_number" in chunk.metadata:
#             chunk.metadata["page"] = chunk.metadata["page_number"]
    
#     return chunks
# # Download the Embeddings from HuggingFace 
# def download_hugging_face_embeddings():
    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # returns 384-dim embeddings
    # return embeddings
    
    
    
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pdfplumber
import re
import os
import json
import io

# Google Drive API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request


def load_pdf_file(data):
    """Load PDF file(s) and extract text with tables"""
    documents = []
    if os.path.isfile(data):  # Check if it's a single file
        with pdfplumber.open(data) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    markdown_rows = [" | ".join(str(cell) if cell is not None else "" for cell in row).strip() for row in table if any(row)]
                    table_text += "\n".join(markdown_rows) + "\n"
                combined_content = f"{page_text.strip()}\n\n{table_text.strip()}".strip()
                documents.append(Document(
                    page_content=combined_content,
                    metadata={"source": os.path.basename(data), "page": page_num}
                ))
    else:  # Fallback to directory mode if needed
        for filename in os.listdir(data):
            if filename.endswith(".pdf"):
                filepath = os.path.join(data, filename)
                with pdfplumber.open(filepath) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        page_text = page.extract_text() or ""
                        tables = page.extract_tables()
                        table_text = ""
                        for table in tables:
                            markdown_rows = [" | ".join(str(cell) if cell is not None else "" for cell in row).strip() for row in table if any(row)]
                            table_text += "\n".join(markdown_rows) + "\n"
                        combined_content = f"{page_text.strip()}\n\n{table_text.strip()}".strip()
                        documents.append(Document(
                            page_content=combined_content,
                            metadata={"source": filename, "page": page_num}
                        ))
    return documents


def extract_clause(text):
    """Helper to extract clause or section references"""
    match = re.search(r'(Clause|Section|Article)\s+[\w.-]+', text, re.IGNORECASE)
    return match.group(0) if match else None


def get_google_drive_service():
    """Create and return authenticated Google Drive service"""
    # Get credentials from environment variable (deployment-safe)
    credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not credentials_json:
        raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set")
    
    try:
        credentials_info = json.loads(credentials_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in GOOGLE_CREDENTIALS_JSON: {str(e)}")
    
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    # Check if we have stored credentials
    creds = None
    token_file = 'token.json'  # This will be created after first auth
    
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(credentials_info, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)


def extract_text_with_google_ocr(pdf_path):
    """Extract text from PDF using Google Drive OCR"""
    try:
        service = get_google_drive_service()
        
        # Step 1: Upload PDF to Drive and convert to Google Doc (applies OCR)
        file_metadata = {
            'name': f'temp_ocr_{os.path.basename(pdf_path)}',
            'mimeType': 'application/vnd.google-apps.document'  # Converts to Doc with OCR
        }
        media = MediaFileUpload(pdf_path, mimetype='application/pdf', resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_id = file.get('id')
        
        # Step 2: Export the OCR'd text as plain text
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        extracted_text = fh.getvalue().decode('utf-8')
        
        # Step 3: Clean up (delete temp file from Drive)
        service.files().delete(fileId=file_id).execute()
        
        return extracted_text
        
    except Exception as e:
        raise ValueError(f"Error with Google Drive OCR: {str(e)}")


def text_split(extracted_data, pdf_path):
    """
    Updated text_split using Google Drive OCR for cloud-based text extraction
    This replaces the unstructured library approach to avoid Tesseract dependency
    """
    try:
        # Step 1: Extract text using Google Drive OCR (cloud-based, high accuracy)
        extracted_text = extract_text_with_google_ocr(pdf_path)
        
        # Step 2: Convert to LangChain Document
        documents = [Document(
            page_content=extracted_text, 
            metadata={"source": os.path.basename(pdf_path), "ocr_method": "google_drive"}
        )]
        
    except Exception as e:
        # Fallback to basic pdfplumber extraction if Google OCR fails
        print(f"Google OCR failed ({str(e)}), falling back to pdfplumber...")
        fallback_docs = load_pdf_file(pdf_path)
        combined_text = "\n\n".join([doc.page_content for doc in fallback_docs])
        documents = [Document(
            page_content=combined_text,
            metadata={"source": os.path.basename(pdf_path), "ocr_method": "pdfplumber_fallback"}
        )]
    
    # Step 3: Split large text into chunks, respecting policy structures
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Larger chunks to fit full clauses
        chunk_overlap=200,  # More overlap to prevent mid-clause breaks
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            "Clause ", "Section ", "Article ",  # Policy-specific separators
            ".",     # Sentence ends
            " "      # Fallback
        ],
        keep_separator=True  # Keeps separators in chunks for context
    )
    chunks = text_splitter.split_documents(documents)
    
    # Step 4: Enhance metadata for better retrieval and explainability
    for i, chunk in enumerate(chunks):
        clause = extract_clause(chunk.page_content)
        chunk.metadata["clause"] = clause if clause else "Unknown"
        chunk.metadata["contains_table"] = ("|" in chunk.page_content or "-" in chunk.page_content)
        chunk.metadata["chunk_id"] = i
        
        # Estimate page number based on chunk position (rough approximation)
        estimated_page = (i // 3) + 1  # Assuming ~3 chunks per page on average
        chunk.metadata["estimated_page"] = estimated_page
    
    return chunks


def download_hugging_face_embeddings():
    """Download the Embeddings from HuggingFace"""
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # returns 384-dim embeddings
    return embeddings

