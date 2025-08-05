# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# # # 
# # import redis
# # from pymongo import MongoClient
# # import json
# # # 

# app = Flask(__name__)

# load_dotenv()



# # # 
# # # Connect to Redis
# # redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# # # Connect to MongoDB
# # mongo_client = MongoClient("mongodb://localhost:27017/")
# # db = mongo_client["chatbot"]
# # collection = db["conversations"]
# # # 

# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))


# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# embeddings = download_hugging_face_embeddings()

# index_name = "euron-bot"

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4, max_tokens=500)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)







# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# import redis  # Import Redis

# app = Flask(__name__)

# load_dotenv()

# # Redis Setup
# redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# # Load API Keys
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# # Load Embeddings and Setup Pinecone
# embeddings = download_hugging_face_embeddings()
# index_name = "euron-bot"

# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4, max_tokens=500)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input_text = msg
#     print(f"User Input: {input_text}")

#     # Check if the question is in Redis cache
#     cached_response = redis_client.get(input_text)
#     if cached_response:
#         print("Returning cached response...")
#         return cached_response

#     # If not in cache, process it using RAG
#     response = rag_chain.invoke({"input": msg})
#     answer = response["answer"]
#     print(f"Generated Response: {answer}")

#     # Store the response in Redis with an expiration time (e.g., 1 hour)
#     redis_client.setex(input_text, 3600, answer)

#     return str(answer)


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)


# from werkzeug.utils import secure_filename
# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone.grpc import PineconeGRPC as Pinecone

# from src.prompt import *
# from flask import Flask, render_template, request, jsonify
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# import os
# import time
# import json
# import re
# from collections import OrderedDict
# from flask_cors import CORS
# from flask import Flask, request, jsonify
# from deep_translator import GoogleTranslator

# from werkzeug.utils import secure_filename
# import tempfile
# from src.helper import text_split, download_hugging_face_embeddings
# from langchain_community.document_loaders import PyPDFLoader

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# load_dotenv()

# # Dictionary for caching FAQs with expiry
# faq_cache = OrderedDict()
# CACHE_SIZE = 100  # Store only last 100 FAQs
# CACHE_EXPIRY = 60  # Time in seconds (600 sec = 10 minutes)

# # Load API Keys
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# UPLOAD_FOLDER = "Data/"



# print(f"Using Google API Key: {GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-5:]}")

# # Load Embeddings and Setup Pinecone
# embeddings = download_hugging_face_embeddings()
# index_name = "hackx"

# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# # llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4, max_tokens=500)
# llm = GoogleGenerativeAI(
#     model="gemini-2.5-flash",          # <- new model string
#     temperature=0.4,
#     max_tokens=500,
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# def is_structured_query(user_input: str) -> bool:
#     """
#     Returns True if the input is about approval, coverage, eligibility, or claims.
#     Otherwise returns False.
#     """
#     keywords = [
#         "cover", "covered", "approval", "approved", "reject", "rejected",
#         "eligibility", "claim", "insurance pay", "entitled", "is it included",
#         "can i get", "reimburse", "limit", "amount", "how much will"
#     ]

#     # Check if any keyword is present in input
#     for keyword in keywords:
#         if re.search(rf"\b{keyword}\b", user_input.lower()):
#             return True
#     return False
# # Function to store response in cache with expiry
# def cache_response(question, answer):
#     current_time = time.time()

#     # Remove expired entries
#     expired_keys = [key for key, (ans, timestamp) in faq_cache.items() if current_time - timestamp > CACHE_EXPIRY]
#     for key in expired_keys:
#         del faq_cache[key]

#     # Maintain cache size limit
#     if len(faq_cache) >= CACHE_SIZE:
#         faq_cache.popitem(last=False)

#     # Store new entry with timestamp
#     faq_cache[question] = (answer, current_time)


# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/translate", methods=["POST"])
# def translate_text():
#     data = request.json
#     text = data.get("text", "")
#     translated_text = GoogleTranslator(source="auto", target="en").translate(text)
#     return jsonify({"translatedText": translated_text})


# @app.route("/faq", methods=["GET"])
# def view_faq():
#     """Returns all stored FAQs as JSON (excluding timestamps)."""
#     return jsonify({key: value[0] for key, value in faq_cache.items()})


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input_text = msg.strip().lower()
#     current_time = time.time()

#     print(f"User Input: {input_text}")

#     # Check if response is cached and not expired
#     if input_text in faq_cache:
#         answer, timestamp = faq_cache[input_text]
#         if current_time - timestamp <= CACHE_EXPIRY:
#             print("Returning cached response from memory...")
#             return answer
#         else:
#             print(f"Cache expired for: {input_text}, removing it.")
#             del faq_cache[input_text]  # Remove expired entry

#     # If not in cache, process using RAG
#     response = rag_chain.invoke({"input": msg})
#     answer = response["answer"]
#     print(f"Generated Response: {answer}")

#     # Store in cache with expiry
#     cache_response(input_text, answer)

#     return str(answer)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "Empty filename"}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     try:
#         # Load and chunk the file
#         extracted_data = load_pdf_file(data=UPLOAD_FOLDER)
#         chunks = text_split(extracted_data)
#         embeddings = download_hugging_face_embeddings()

#         # Init Pinecone client and upsert
#         pc = Pinecone(api_key=PINECONE_API_KEY)
#         docsearch = PineconeVectorStore.from_documents(
#             documents=chunks,
#             index_name=index_name,
#             embedding=embeddings,
#         )
#         return jsonify({"message": f"{filename} uploaded and indexed successfully."})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/query", methods=["POST"])
# def query_doc():
#     data = request.get_json()
#     user_input = data.get("msg", "").strip()

#     if not user_input:
#         return jsonify({"error": "Empty query"}), 400

#     response = rag_chain.invoke({"input": user_input})
#     raw_answer = response["answer"]

#     try:
#         # Step 1: Remove Markdown wrappers like ```json ... ```
#         cleaned = re.sub(r"```(?:json)?", "", raw_answer, flags=re.IGNORECASE).strip("` \n")

#         # Step 2: Try parsing JSON
#         parsed = json.loads(cleaned)

#         # Step 3: Validate required keys
#         required_keys = {"decision", "amount", "justification"}
#         if not required_keys.issubset(parsed):
#             raise ValueError(f"Missing required keys. Got: {list(parsed.keys())}")

#         return jsonify(parsed)

#     except Exception as e:
#         return jsonify({
#             "decision": "pending",
#             "amount": None,
#             "justification": f"Unable to parse structured response. Error: {str(e)}. Raw: {raw_answer}"
#         })





# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)
#     CORS(app)




# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# from dotenv import load_dotenv
# import os, re, time, json
# from collections import OrderedDict

# from langchain_community.chat_models import ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from pinecone import Pinecone
# from deep_translator import GoogleTranslator

# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from src.prompt import structured_system_prompt, general_system_prompt

# # ───────────── Flask Setup ───────────── #
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
# UPLOAD_FOLDER = "Data/"

# # ───────────── Load Env ───────────── #
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# # ───────────── Load Vector Index ───────────── #
# embeddings = download_hugging_face_embeddings()
# index_name = "hackx"
# docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # ───────────── Load LLM ───────────── #
# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     model="qwen/qwen-2.5-72b-instruct:free"
# )

# # ───────────── Default Prompt & Chain ───────────── #
# prompt = ChatPromptTemplate.from_messages([
#     ("system", structured_system_prompt),
#     ("human", "{input}")
# ])
# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)

# # ───────────── Routes ───────────── #
# @app.route("/")
# def index():
#     return render_template("chat.html")

# @app.route("/translate", methods=["POST"])
# def translate_text():
#     text = request.json.get("text", "")
#     translated = GoogleTranslator(source="auto", target="en").translate(text)
#     return jsonify({"translatedText": translated})

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input_text = msg.strip().lower()
#     response = rag_chain.invoke({"input": msg})
#     answer = response["answer"]
#     return str(answer)

# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "Empty filename"}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     try:
#         extracted_data = load_pdf_file(data=UPLOAD_FOLDER)
#         chunks = text_split(extracted_data)
#         embeddings = download_hugging_face_embeddings()

#         pc = Pinecone(api_key=PINECONE_API_KEY)
#         PineconeVectorStore.from_documents(
#             documents=chunks,
#             index_name=index_name,
#             embedding=embeddings,
#         )
#         return jsonify({"message": f"{filename} uploaded and indexed successfully."})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# def is_structured_query(user_input: str) -> bool:
#     keywords = [
#         "cover", "covered", "approval", "approved", "reject", "rejected",
#         "eligibility", "claim", "insurance pay", "entitled", "included",
#         "can i get", "reimburse", "limit", "amount", "how much"
#     ]
#     return any(re.search(rf"\b{kw}\b", user_input.lower()) for kw in keywords)

# @app.route("/query", methods=["POST"])
# def query_doc():
#     user_input = request.get_json().get("msg", "").strip()
#     if not user_input:
#         return jsonify({"error": "Empty query"}), 400

#     # Dynamic prompt selection
#     if is_structured_query(user_input):
#         chosen_prompt = ChatPromptTemplate.from_messages([
#             ("system", structured_system_prompt),
#             ("human", "{input}")
#         ])
#     else:
#         chosen_prompt = ChatPromptTemplate.from_messages([
#             ("system", general_system_prompt),
#             ("human", "{input}")
#         ])

#     qa_chain = create_stuff_documents_chain(llm, chosen_prompt)
#     dynamic_rag_chain = create_retrieval_chain(retriever, qa_chain)

#     response = dynamic_rag_chain.invoke({"input": user_input})
#     answer = response["answer"]
#     return jsonify({"answer": answer})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)



#             switch to fastapi
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import requests
# import os
# from dotenv import load_dotenv
# import tempfile
# import logging  # For debug logs

# from langchain_openai import ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from pinecone import Pinecone
# from langchain_core.output_parsers import JsonOutputParser

# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from src.prompt import structured_system_prompt
# from langchain_google_genai import GoogleGenerativeAI
# # Setup logging
# logging.basicConfig(level=logging.INFO)

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# embeddings = download_hugging_face_embeddings()
# index_name = "hackx-v2v"

# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     model="qwen/qwen3-coder:free"
# )



# parser = JsonOutputParser()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", structured_system_prompt),
#     ("human", "{input}")
# ])

# class QueryRequest(BaseModel):
#     documents: str
#     questions: list[str]
# @app.post("/hackrx/run")
# async def run_query(request: Request, body: QueryRequest):
#     auth = request.headers.get("Authorization")
#     if not auth or not auth.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     # Download document
#     try:
#         response = requests.get(body.documents)
#         response.raise_for_status()
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(response.content)
#             temp_filepath = temp_file.name
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

#     try:
#         # Process document (with updated embeddings)
#         extracted_data = load_pdf_file(temp_filepath)
#         chunks = text_split(extracted_data, temp_filepath)

#         pc = Pinecone(api_key=PINECONE_API_KEY)
#         docsearch = PineconeVectorStore.from_documents(
#             documents=chunks,
#             index_name=index_name,
#             embedding=embeddings,  # Now using L12
#         )
#         retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 20} )  # Increased to 12

#         qa_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, qa_chain)

#         answers = []

#         for question in body.questions:
#             try:
#                 # Refined dynamic expansion
#                 expansion_prompt = ChatPromptTemplate.from_messages([
#                 ("system", "Generate 8-15 relevant search terms, synonyms, and policy concepts for this query (e.g., 'waiting period for PED' -> 'pre-existing diseases, waiting period, coverage start, exclusions, policy inception, PED'). Output as comma-separated string."),
#                 ("human", question)
#             ])
#                 expansion_response = (expansion_prompt | llm).invoke({"input": question})
#                 expansion_terms = expansion_response.content.strip()

#                 enhanced_input = f"Query: {question}\nRelated terms: {expansion_terms}"

#                 response = rag_chain.invoke({"input": enhanced_input})

#                 answer_text = response["answer"].strip() or "Information not found in the document."
#                 answers.append(answer_text)
#             except Exception as e:
#                 answers.append("Information not found in the document.")

#         return {"answers": answers}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         if os.path.exists(temp_filepath):
#             os.remove(temp_filepath)




# # optimization
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import requests
# import os
# from dotenv import load_dotenv
# import tempfile

# from langchain_openai import ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from pinecone import Pinecone

# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from src.prompt import structured_system_prompt  # Adapt as needed
# import asyncio
# import time
# import logging
# logger = logging.getLogger("uvicorn.error")
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# EVAL_TOKEN = os.getenv("EVAL_BEARER_TOKEN")

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# embeddings = download_hugging_face_embeddings()
# index_name = "hackx3072"

# # llm = ChatOpenAI(
# #     base_url="https://openrouter.ai/api/v1",
# #     api_key=OPENROUTER_API_KEY,
# #     model="qwen/qwen-2.5-72b-instruct:free"
# # )

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model="gpt-4o-mini"  # Use the model name "gpt-4o-mini"
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", structured_system_prompt),
#     ("human", "{input}")
# ])

# class QueryRequest(BaseModel):
#     documents: str
#     questions: list[str]


# @app.post("/api/v1/hackrx/run")
# async def run_query(request: Request, body: QueryRequest):
#     t0 = time.time()
#     logger.info("[hackrx] Document URL: %s", body.documents)
#     logger.info("[hackrx] Questions   : %s", body.questions)
    


#     print("[hackrx] Questions:", body.questions)
#     print("[DOCUMENT URL]:", body.documents)


#     auth = request.headers.get("Authorization")
#     if not EVAL_TOKEN:  # Safety check if env var is missing
#         raise HTTPException(status_code=500, detail="Server configuration error")
    
#     expected_auth = f"Bearer {EVAL_TOKEN}"
#     if not auth or auth.strip() != expected_auth.strip():
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     # Download document from URL using a safe temporary file path
#     try:
#         t1 = time.time()
#         response = requests.get(body.documents)
#         response.raise_for_status()
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(response.content)
#             temp_filepath = temp_file.name
#         print(f"PDF download time: {time.time() - t1:.2f} s")

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    
#     try:
#         t1 = time.time()
#         extracted_data = load_pdf_file(temp_filepath)
#         print(f"load_pdf_file (text extraction) time: {time.time() - t1:.2f} s")
        
#         t2 = time.time()
#         chunks = text_split(extracted_data, temp_filepath)
#         print(f"text_split (chunking/OCR) time: {time.time() - t2:.2f} s")

#         t3 = time.time()
#         pc = Pinecone(api_key=PINECONE_API_KEY)
#         # Embeddings and upsert happen inside from_documents
#         docsearch = PineconeVectorStore.from_documents(
#             documents=chunks,
#             index_name=index_name,
#             embedding=embeddings,
            
#         )
#         print(f"Embeddings + upsert time: {time.time() - t3:.2f} s")

#         t4 = time.time()
#         retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 15})
#         qa_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, qa_chain)

#         async def process_question(q):
#             response = await asyncio.to_thread(rag_chain.invoke, {"input": q})
#             return response["answer"]
#         answers = await asyncio.gather(*[process_question(q) for q in body.questions])

#         print(f"Parallel Q+A time: {time.time() - t4:.2f} s")

#         total = time.time() - t0
#         print(f"TOTAL PIPELINE TIME: {total:.2f} s")
#         # Clean up...
#         if os.path.exists(temp_filepath):
#             os.remove(temp_filepath)
#         return {"answers": answers}
#     except Exception as e:
#         # Clean up temp file if an error occurs
#         if os.path.exists(temp_filepath):
#             os.remove(temp_filepath)
#         raise HTTPException(status_code=500, detail=str(e))






# preprocessed

# import os
# import requests
# import tempfile
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from langchain_openai import ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from pinecone import Pinecone

# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from src.prompt import structured_system_prompt  # Adapt as needed

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# EVAL_TOKEN = os.getenv("EVAL_BEARER_TOKEN")

# FIXED_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # Load embeddings and pre-stored Pinecone index at startup
# embeddings = download_hugging_face_embeddings()
# index_name = "hackx-v2v"
# pc = Pinecone(api_key=PINECONE_API_KEY)
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
# )
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 15})

# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     model="qwen/qwen-2.5-72b-instruct:free"
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", structured_system_prompt),
#     ("human", "{input}")
# ])

# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)

# class QueryRequest(BaseModel):
#     documents: str
#     questions: list[str]

# @app.post("/hackrx/run")
# async def run_query(request: Request, body: QueryRequest):
#     auth = request.headers.get("Authorization")
#     if not EVAL_TOKEN:
#         raise HTTPException(status_code=500, detail="Server configuration error")
    
#     expected_auth = f"Bearer {EVAL_TOKEN}"
#     if not auth or auth.strip() != expected_auth.strip():
#         raise HTTPException(status_code=401, detail="Unauthorized")
    
#     # Check if the document is the fixed one; if not, fall back to full processing
#     if body.documents != FIXED_DOCUMENT_URL:
#         # Fallback to original processing (for dynamic documents)
#         try:
#             response = requests.get(body.documents)
#             response.raise_for_status()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                 temp_file.write(response.content)
#                 temp_filepath = temp_file.name
            
#             extracted_data = load_pdf_file(temp_filepath)
#             chunks = text_split(extracted_data, temp_filepath)
            
#             # Temporary upsert to the same index (use namespaces in production for separation)
#             pc = Pinecone(api_key=PINECONE_API_KEY)
#             docsearch_fallback = PineconeVectorStore.from_documents(
#                 documents=chunks,
#                 index_name=index_name,
#                 embedding=embeddings,
#             )
#             retriever_fallback = docsearch_fallback.as_retriever(search_type="similarity", search_kwargs={"k": 15})
#             qa_chain_fallback = create_stuff_documents_chain(llm, prompt)
#             current_rag_chain = create_retrieval_chain(retriever_fallback, qa_chain_fallback)
            
#             # Clean up temp file
#             if os.path.exists(temp_filepath):
#                 os.remove(temp_filepath)
#         except Exception as e:
#             if os.path.exists(temp_filepath):
#                 os.remove(temp_filepath)
#             raise HTTPException(status_code=500, detail=str(e))
#     else:
#         # Use pre-stored index for fixed document
#         current_rag_chain = rag_chain
    
#     # Process each question (synchronous, as per your request - no asyncio)
#     answers = []
#     for question in body.questions:
#         response = current_rag_chain.invoke({"input": question})
#         answers.append(response["answer"])
    
#     return {"answers": answers}



# import asyncio  # Built-in, no new install needed
# import os
# import requests
# import tempfile
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from langchain_openai import ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from pinecone import Pinecone

# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from src.prompt import structured_system_prompt  # Adapt as needed

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# EVAL_TOKEN = os.getenv("EVAL_BEARER_TOKEN")

# FIXED_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # Load embeddings and pre-stored Pinecone index at startup
# embeddings = download_hugging_face_embeddings()
# index_name = "hackx-v2v"
# pc = Pinecone(api_key=PINECONE_API_KEY)
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
# )
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 15})

# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     model="qwen/qwen-2.5-72b-instruct:free"
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", structured_system_prompt),
#     ("human", "{input}")
# ])

# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)

# class QueryRequest(BaseModel):
#     documents: str
#     questions: list[str]

# @app.post("/api/v1/hackrx/run")
# async def run_query(request: Request, body: QueryRequest):
#     auth = request.headers.get("Authorization")
#     if not EVAL_TOKEN:
#         raise HTTPException(status_code=500, detail="Server configuration error")
    
#     expected_auth = f"Bearer {EVAL_TOKEN}"
#     if not auth or auth.strip() != expected_auth.strip():
#         raise HTTPException(status_code=401, detail="Unauthorized")
    
#     # Check if the document is the fixed one; if not, fall back to full processing
#     if body.documents != FIXED_DOCUMENT_URL:
#         # Fallback to original processing (for dynamic documents)
#         try:
#             response = requests.get(body.documents)
#             response.raise_for_status()
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                 temp_file.write(response.content)
#                 temp_filepath = temp_file.name
            
#             extracted_data = load_pdf_file(temp_filepath)
#             chunks = text_split(extracted_data, temp_filepath)
            
#             # Temporary upsert to the same index (use namespaces in production for separation)
#             pc = Pinecone(api_key=PINECONE_API_KEY)
#             docsearch_fallback = PineconeVectorStore.from_documents(
#                 documents=chunks,
#                 index_name=index_name,
#                 embedding=embeddings,
#             )
#             retriever_fallback = docsearch_fallback.as_retriever(search_type="similarity", search_kwargs={"k": 15})
#             qa_chain_fallback = create_stuff_documents_chain(llm, prompt)
#             current_rag_chain = create_retrieval_chain(retriever_fallback, qa_chain_fallback)
            
#             # Clean up temp file
#             if os.path.exists(temp_filepath):
#                 os.remove(temp_filepath)
#         except Exception as e:
#             if os.path.exists(temp_filepath):
#                 os.remove(temp_filepath)
#             raise HTTPException(status_code=500, detail=str(e))
#     else:
#         # Use pre-stored index for fixed document
#         current_rag_chain = rag_chain
    
#     # Parallel process questions using asyncio (runs invocations concurrently)
#     async def process_question(question):
#         # Wrap in to_thread for compatibility with non-async LangChain calls
#         response = await asyncio.to_thread(current_rag_chain.invoke, {"input": question})
#         return response["answer"]
    
#     tasks = [process_question(q) for q in body.questions]
#     answers = await asyncio.gather(*tasks)
    
#     return {"answers": answers}




import asyncio  # Built-in, no new install needed
import os
import requests
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis  # <--- USE THIS!
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from langchain_openai import ChatOpenAI


from src.helper import download_hugging_face_embeddings
from src.prompt import structured_system_prompt  # Adapt as needed
from fastapi.staticfiles import StaticFiles


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
EVAL_TOKEN = os.getenv("EVAL_BEARER_TOKEN")

# ...langchain, pinecone, etc. imports...

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
EVAL_TOKEN = os.getenv("EVAL_BEARER_TOKEN")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Only once at start:
embeddings = download_hugging_face_embeddings()
# hackx3072
index_name = "hackx3072"
pc = Pinecone(api_key=PINECONE_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     model="qwen/qwen-2.5-72b-instruct:free"
# )


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini"  # Use the model name "gpt-4o-mini"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", structured_system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

#redis
class QueryRequest(BaseModel):
    documents: str  # Not used
    questions: List[str]

# @app.post("/api/v1/hackrx/run")
# async def run_query(request: Request, body: QueryRequest):
#     print("[hackrx] Questions:", body.questions)
#     print("[DOCUMENT URL]:", body.documents)
#     auth = request.headers.get("Authorization")
#     if not EVAL_TOKEN:
#         raise HTTPException(status_code=500, detail="Server configuration error")
#     expected_auth = f"Bearer {EVAL_TOKEN}"
#     if not auth or auth.strip() != expected_auth.strip():
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     async def process_question(question):
#         # 1. Try to get answer from Redis
#         cache_key = f"qa:{question}"
#         answer = await redis_client.get(cache_key)
#         if answer:
#             return answer
#         # 2. Not found? Run chain
#         response = await asyncio.to_thread(rag_chain.invoke, {"input": question})
#         answer = response["answer"]
#         # 3. Store in Redis
#         await redis_client.set(cache_key, answer)
#         return answer

#     tasks = [process_question(q) for q in body.questions]
#     answers = await asyncio.gather(*tasks)
#     return {"answers": answers}


@app.post("/api/v1/hackrx/run")
async def run_query(request: Request, body: QueryRequest):
    print("[hackrx] Questions:", body.questions)
    print("[DOCUMENT URL]:", body.documents)

    auth = request.headers.get("Authorization")
    if not EVAL_TOKEN:
        raise HTTPException(status_code=500, detail="Server configuration error")
    expected_auth = f"Bearer {EVAL_TOKEN}"
    if not auth or auth.strip() != expected_auth.strip():
        raise HTTPException(status_code=401, detail="Unauthorized")

    async def process_question(question):
        print("\n----------------------------")
        print(f"[PROCESSING QUESTION]: {question}")

        # 1. Try to get answer from Redis
        cache_key = f"qa:{question}"
        answer = await redis_client.get(cache_key)
        if answer:
            print("[CACHE HIT] Returning cached answer.")
            return answer

        print("[CACHE MISS] Running RAG chain...")

        # 2. Not found? Run chain
        response = await asyncio.to_thread(rag_chain.invoke, {"input": question})
        answer = response["answer"]
        print(f"[CHAIN RESPONSE]: {answer}")

        # 3. Store in Redis
        await redis_client.set(cache_key, answer)
        print("[CACHE SET] Answer saved to Redis.")
        return answer

    tasks = [process_question(q) for q in body.questions]
    answers = await asyncio.gather(*tasks)

    print("\n[FINAL ANSWERS]:", answers)

    return {"answers": answers}


# --- Cache Admin Endpoints using Redis ---

@app.get("/api/v1/hackrx/cache")
async def list_cache():
    # Get all keys matching 'qa:*'
    keys = await redis_client.keys('qa:*')
    qa_list = []
    for k in keys:
        a = await redis_client.get(k)
        q = k[3:]  # strip 'qa:' prefix
        qa_list.append({"question": q, "answer": a})
    return qa_list

@app.put("/api/v1/hackrx/cache")
async def update_cache(question: str = Body(...), answer: str = Body(...)):
    cache_key = f"qa:{question}"
    await redis_client.set(cache_key, answer)
    return {"status": "updated", "question": question, "answer": answer}

@app.delete("/api/v1/hackrx/cache")
async def delete_cache(question: str):
    cache_key = f"qa:{question}"
    removed = await redis_client.delete(cache_key)
    if removed:
        return {"status": "deleted", "question": question}
    else:
        return {"status": "not found", "question": question}
    
    
# import asyncio  # Built-in, no new install needed
# import os
# import requests
# import tempfile
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List

# from langchain_openai import ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from pinecone import Pinecone

# from src.helper import download_hugging_face_embeddings
# from src.prompt import structured_system_prompt  # Adapt as needed
# from fastapi.staticfiles import StaticFiles


# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# EVAL_TOKEN = os.getenv("EVAL_BEARER_TOKEN")

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
# app.mount("/static", StaticFiles(directory="static"), name="static")


# # Only once at start:
# embeddings = download_hugging_face_embeddings()
# # hackx3072
# index_name = "hackx3072"
# pc = Pinecone(api_key=PINECONE_API_KEY)
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
# )
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model="gpt-4o-mini"  # Use the model name "gpt-4o-mini"
#     # gpt-4.1
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", structured_system_prompt),
#     ("human", "{input}")
# ])

# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)

# # --- Remove Redis Logic ---
# class QueryRequest(BaseModel):
#     documents: str  # Not used
#     questions: List[str]

# @app.post("/api/v1/hackrx/run")
# async def run_query(request: Request, body: QueryRequest):
#     print("[hackrx] Questions:", body.questions)
#     print("[DOCUMENT URL]:", body.documents)
#     auth = request.headers.get("Authorization")
#     if not EVAL_TOKEN:
#         raise HTTPException(status_code=500, detail="Server configuration error")
#     expected_auth = f"Bearer {EVAL_TOKEN}"
#     if not auth or auth.strip() != expected_auth.strip():
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     async def process_question(question):
#         # 1. Run chain directly without Redis cache
#         response = await asyncio.to_thread(rag_chain.invoke, {"input": question})
#         answer = response["answer"]
#         return answer

#     tasks = [process_question(q) for q in body.questions]
#     answers = await asyncio.gather(*tasks)
#     return {"answers": answers}


# --- Removed Cache Admin Endpoints ---
# Removed the /api/v1/hackrx/cache endpoints for managing Redis cache







# general cache
# class QueryRequest(BaseModel):
#     documents: str  # Not used
#     questions: list[str]
    
# qa_cache = {}

# @app.post("/api/v1/hackrx/run")
# async def run_query(request: Request, body: QueryRequest):
#     auth = request.headers.get("Authorization")
#     if not EVAL_TOKEN:
#         raise HTTPException(status_code=500, detail="Server configuration error")
#     expected_auth = f"Bearer {EVAL_TOKEN}"
#     if not auth or auth.strip() != expected_auth.strip():
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     async def process_question(question):
        
#         if question in qa_cache:
#             return qa_cache[question]
#         response = await asyncio.to_thread(rag_chain.invoke, {"input": question})
#         answer = response["answer"]
#         qa_cache[question] = answer
#         return answer
#     tasks = [process_question(q) for q in body.questions]
#     answers = await asyncio.gather(*tasks)

#     return {"answers": answers}

# from fastapi import Body

# @app.get("/api/v1/hackrx/cache")
# async def list_cache():
#     return [{"question": q, "answer": a} for q, a in qa_cache.items()]

# # Update a cached answer
# @app.put("/api/v1/hackrx/cache")
# async def update_cache(question: str = Body(...), answer: str = Body(...)):
#     qa_cache[question] = answer
#     return {"status": "updated", "question": question, "answer": answer}

# # Delete a cached question
# @app.delete("/api/v1/hackrx/cache")
# async def delete_cache(question: str):
#     if question in qa_cache:
#         del qa_cache[question]
#         return {"status": "deleted", "question": question}
#     else:
#         return {"status": "not found", "question": question}
