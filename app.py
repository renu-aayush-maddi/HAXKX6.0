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

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import tempfile

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from src.prompt import structured_system_prompt  # Adapt as needed

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

embeddings = download_hugging_face_embeddings()
index_name = "hackx"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="qwen/qwen-2.5-72b-instruct:free"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", structured_system_prompt),
    ("human", "{input}")
])

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]


@app.post("/hackrx/run")
async def run_query(request: Request, body: QueryRequest):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Download document from URL using a safe temporary file path
    try:
        response = requests.get(body.documents)
        response.raise_for_status()  # Raise error if download fails
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_filepath = temp_file.name  # Gets a valid temp path like C:\Users\renua\AppData\Local\Temp\tmp123.pdf
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    
    # Process the document using your helper functions
    try:
        extracted_data = load_pdf_file(temp_filepath)  # Adapted to use file path, not directory
        chunks = text_split(extracted_data, temp_filepath)
        
        # Create or use Pinecone index
        pc = Pinecone(api_key=PINECONE_API_KEY)
        docsearch = PineconeVectorStore.from_documents(
            documents=chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Create RAG chain
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        
        # Process each question
        answers = []
        for question in body.questions:
            response = rag_chain.invoke({"input": question})
            answers.append(response["answer"])
        
        # Clean up temp file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return {"answers": answers}
    except Exception as e:
        # Clean up temp file if an error occurs
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(status_code=500, detail=str(e))
