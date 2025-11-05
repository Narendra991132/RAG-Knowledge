import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import (
    S3FileLoader,
    YoutubeLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import PostgresChatMessageHistory
import boto3
import time
import hashlib
import secrets
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import logging
import random
from botocore.exceptions import ClientError
from botocore.config import Config

# ===============================================================
# Compatibility Patch for Older Streamlit Versions
# ===============================================================
if not hasattr(st, "cache_resource"):
    st.cache_resource = st.experimental_singleton

# ===============================================================
# Logging Configuration
# ===============================================================
logging.basicConfig(
    filename="app_metrics.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logging.getLogger('botocore').setLevel(logging.ERROR)

# ===============================================================
# Streamlit Page Setup
# ===============================================================
st.set_page_config(page_title="Knowledge Assistant", page_icon="💡", layout="wide")

# ===============================================================
# UI Styling Helpers
# ===============================================================
def add_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #f9fafb; }
        .main-header {
            background-color: #003366;
            padding: 1.2rem;
            border-radius: 8px;
            text-align: center;
            color: white;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
        }
        .user-msg, .bot-msg {
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin: 0.4rem 0;
            line-height: 1.5;
            word-wrap: break-word;
        }
        .user-msg { background-color: #d1e7dd; color: #0f5132; border-left: 5px solid #0f5132; }
        .bot-msg { background-color: #e2e3e5; color: #1b1b1b; border-left: 5px solid #495057; }
        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
        .stButton>button:hover { background-color: #00509e; color: #fff; }
        </style>
    """, unsafe_allow_html=True)

# ===============================================================
# Utility Functions
# ===============================================================
def generate_session_id():
    t = int(time.time() * 1000)
    r = secrets.randbelow(1000000)
    return hashlib.md5(bytes(str(t) + str(r), 'utf-8'), usedforsecurity=False).hexdigest()

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDFs concurrently."""
    def extract(pdf):
        pdf_reader = PdfReader(pdf)
        return " ".join(page.extract_text() or "" for page in pdf_reader.pages)
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract, pdf_docs))
    return " ".join(texts)

def get_text_chunks(text):
    """Split text into moderately large chunks to reduce Bedrock API calls."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

def save_uploaded_file(uploaded_file):
    """Save Streamlit UploadedFile to a temporary file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# ===============================================================
# Database / Vectorstore Setup with Throttling Protection
# ===============================================================
@st.cache_resource
def get_vectorstore(text_chunks):
    retry_config = Config(retries={'max_attempts': 10, 'mode': 'standard'})

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        config=retry_config
    )

    embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")

    def embed_with_rate_limit(embeddings, texts, batch_size=5, delay=0.5):
        """Batch embed documents safely with exponential backoff and delay."""
        all_vecs = []
        total = len(texts)
        progress = st.progress(0, text="Embedding documents...")

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(6):
                try:
                    vecs = embeddings.embed_documents(batch)
                    all_vecs.extend(vecs)
                    break
                except ClientError as e:
                    if e.response['Error']['Code'] == "ThrottlingException":
                        wait = min(2 ** attempt + random.uniform(0, 1), 30)
                        logging.warning(f"Throttled at batch {i//batch_size}, retrying in {round(wait,2)}s...")
                        time.sleep(wait)
                    else:
                        raise
            time.sleep(delay)
            progress.progress(min((i + batch_size) / total, 1.0),
                              text=f"Embedding batch {i//batch_size+1}/{(total//batch_size)+1}")
        progress.empty()
        return all_vecs

    if text_chunks is None:
        return PGVector(connection_string=CONNECTION_STRING, embedding_function=embeddings)

    vectors = embed_with_rate_limit(embeddings, text_chunks)

    # ✅ Safe universal method that works across LangChain versions
    class PrecomputedPGVector(PGVector):
        @classmethod
        def from_texts_with_vectors(cls, texts, text_embeddings, connection_string, embedding_function):
            instance = cls(connection_string=connection_string, embedding_function=embedding_function)
            instance.add_texts(texts=texts, embeddings=text_embeddings)
            return instance

    return PrecomputedPGVector.from_texts_with_vectors(
        texts=text_chunks,
        text_embeddings=vectors,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings
    )

# ===============================================================
# Bedrock LLM Setup
# ===============================================================
@st.cache_resource
def get_bedrock_llm(selected_llm):
    if selected_llm.startswith("amazon.titan"):
        return Bedrock(model_id=selected_llm, model_kwargs={"maxTokenCount": 4096, "temperature": 0})
    elif selected_llm == "amazon.nova-pro-v1:0":
        return Bedrock(model_id=selected_llm)
    elif selected_llm.startswith("anthropic."):
        return Bedrock(model_id=selected_llm, model_kwargs={'max_tokens_to_sample': 4096})
    else:
        raise ValueError(f"Unsupported LLM selected: {selected_llm}")

# ===============================================================
# Conversation Chain Setup
# ===============================================================
def get_conversation_chain(vectorstore, selected_llm):
    llm = get_bedrock_llm(selected_llm)
    _connection_string = CONNECTION_STRING.replace('+psycopg2', '').replace(':5432', '')
    message_history = PostgresChatMessageHistory(connection_string=_connection_string, session_id=generate_session_id())
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# ===============================================================
# Document Loading Logic
# ===============================================================
def load_docs_from_source(source, files_or_key):
    """Unified loader for different data sources."""
    text = ""
    if source == "PDF Documents":
        text = get_pdf_text(files_or_key)
    elif source == "Word Document":
        for f in files_or_key:
            path = save_uploaded_file(f)
            loader = Docx2txtLoader(path)
            text += " ".join([d.page_content for d in loader.load()])
            os.remove(path)
    elif source == "PowerPoint":
        for f in files_or_key:
            path = save_uploaded_file(f)
            loader = UnstructuredPowerPointLoader(path)
            text += " ".join([d.page_content for d in loader.load()])
            os.remove(path)
    elif source == "CSV File":
        loader = CSVLoader(files_or_key)
        text = " ".join([d.page_content for d in loader.load()])
    elif source == "S3 Bucket":
        loader = S3FileLoader("knowledge-warehouse", f"documentEmbeddings/{files_or_key}")
        text = " ".join([d.page_content for d in loader.load()])
    return text

# ===============================================================
# Chat Handling
# ===============================================================
def handle_userinput(user_question):
    try:
        start_time = time.time()
        with st.spinner("💭 Thinking... Generating your answer..."):
            response = st.session_state.conversation({'question': user_question})
        latency = round(time.time() - start_time, 3)
        retriever = st.session_state.conversation.retriever
        retrieved_docs = retriever.get_relevant_documents(user_question)
        retrieved_preview = [d.page_content[:150].replace('\n', ' ') for d in retrieved_docs]
        logging.info(f"[EVAL] Query='{user_question}' | Latency={latency}s | Retrieved={len(retrieved_docs)} chunks")

        st.markdown(f"<div class='user-msg'><b>🧑 You:</b> {user_question}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-msg'><b>🤖 Assistant:</b> {response['answer']}</div>", unsafe_allow_html=True)
        st.session_state.chat_history = response['chat_history']
        st.markdown(f"**⏱️ Response Time:** {latency}s | **📄 Retrieved Chunks:** {len(retrieved_docs)}")

    except Exception as e:
        st.warning("⚠️ Something went wrong, please try again.")
        logging.error(f"[ERROR] {str(e)}")

# ===============================================================
# Main App Logic
# ===============================================================
def main():
    add_custom_css()
    st.markdown("<div class='main-header'>💡 Knowledge Assistant</div>", unsafe_allow_html=True)
    st.write("Interact with enterprise knowledge using **Amazon Bedrock LLMs** and **Aurora pgvector** for retrieval-augmented generation (RAG).")

    with st.sidebar:
        st.header("⚙️ Configuration")
        source_options = ["PDF Documents", "Word Document", "PowerPoint", "CSV File", "S3 Bucket"]
        selected_source = st.selectbox("📂 Select Knowledge Source:", source_options)
        llm_options = [
            'amazon.nova-pro-v1:0',
            'amazon.titan-text-express-v1',
            'amazon.titan-text-lite-v1',
            'amazon.titan-tg1-large',
            'anthropic.claude-3-5-sonnet-20240620-v1:0',
            'anthropic.claude-3-haiku-20240307-v1:0',
            'anthropic.claude-3-opus-20240229-v1:0'
        ]
        selected_llm = st.selectbox("🤖 Choose Model:", llm_options, index=2)

    tab1, tab2 = st.tabs(["💬 Chat Interface", "📈 Evaluation Metrics"])
    with tab1:
        all_text = ""
        if selected_source != "S3 Bucket":
            uploaded_files = st.file_uploader(f"Upload files for {selected_source}", accept_multiple_files=True)
        else:
            s3_client = boto3.client('s3')
            response = s3_client.list_objects_v2(Bucket='knowledge-warehouse', Prefix='documentEmbeddings/')
            keys = [obj['Key'].split('/')[-1] for obj in response.get('Contents', [])]
            uploaded_files = st.selectbox("Select S3 Document", keys)

        if st.button(f"Process {selected_source}") and uploaded_files:
            with st.spinner(f"⚙️ Processing {selected_source}... Please wait."):
                start = time.time()
                all_text = load_docs_from_source(selected_source, uploaded_files)
                st.success(f"✅ {selected_source} processed successfully in {round(time.time() - start, 2)}s!")

        if all_text:
            with st.spinner("⚙️ Creating text chunks and embeddings..."):
                chunks = get_text_chunks(all_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore, selected_llm)
            st.success("🎉 Documents processed and vectorstore initialized successfully!")

        question = st.text_input("💭 Ask your question:")
        if st.button("Submit Question") and question:
            handle_userinput(question)

        for msg in st.session_state.get("chat_history", []):
            role = "User" if msg.type == "human" else "Assistant"
            css_class = "user-msg" if role == "User" else "bot-msg"
            st.markdown(f"<div class='{css_class}'><b>{role}:</b> {msg.content}</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Evaluation Metrics")
        st.write("Check the `app_metrics.log` file for query latency, retrieval performance, and system logs.")
        st.code(open("app_metrics.log").read()[-1500:], language="text")

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(get_vectorstore(None), selected_llm)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# ===============================================================
# Entry Point
# ===============================================================
if __name__ == '__main__':
    load_dotenv()
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER"),
        user=os.environ.get("PGVECTOR_USER"),
        password=os.environ.get("PGVECTOR_PASSWORD"),
        host=os.environ.get("PGVECTOR_HOST"),
        port=os.environ.get("PGVECTOR_PORT"),
        database=os.environ.get("PGVECTOR_DATABASE"),
    )
    print(f"[DEBUG] Connection string initialized: {CONNECTION_STRING}")
    main()
