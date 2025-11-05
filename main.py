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
from dotenv import load_dotenv
import logging


# ---------------------------------------------------------------
logging.getLogger('botocore').setLevel(logging.ERROR)


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
        .subheader {
            background-color: #e9ecef;
            padding: 0.8rem;
            border-left: 5px solid #003366;
            border-radius: 5px;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 1rem;
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
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #003366; }
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
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=512,
        chunk_overlap=103,
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
# Database / Vectorstore Setup
# ===============================================================
# def get_vectorstore(text_chunks):
#     embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-west-2")
#     if text_chunks is None:
#         return PGVector(connection_string=CONNECTION_STRING, embedding_function=embeddings)
#     return PGVector.from_texts(texts=text_chunks, embedding=embeddings, connection_string=CONNECTION_STRING)
def get_vectorstore(text_chunks):
    # Create a boto3 client using credentials from .env
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN")  # optional
    )

    # Create embeddings using the explicit client
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v1"
    )

    if text_chunks is None:
        return PGVector(connection_string=CONNECTION_STRING, embedding_function=embeddings)

    return PGVector.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        connection_string=CONNECTION_STRING
    )


# ===============================================================
# Bedrock LLM Setup
# ===============================================================
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
# Chat UI
# ===============================================================
# def handle_userinput(user_question):
#     try:
#         response = st.session_state.conversation({'question': user_question})
#         st.markdown(f"<div class='user-msg'>User: {user_question}</div>", unsafe_allow_html=True)
#         st.markdown(f"<div class='bot-msg'>Assistant: {response['answer']}</div>", unsafe_allow_html=True)
#         st.session_state.chat_history = response['chat_history']
#     except ValueError:
#         st.warning("Sorry, please try rephrasing your question.")


def handle_userinput(user_question):
    try:
        # --- Start timer for latency measurement ---
        start_time = time.time()

        # --- Run the retrieval + generation chain ---
        response = st.session_state.conversation({'question': user_question})

        # --- End timer and compute latency ---
        latency = round(time.time() - start_time, 3)

        # --- Log retrieved chunks for evaluation ---
        retriever = st.session_state.conversation.retriever
        retrieved_docs = retriever.get_relevant_documents(user_question)
        retrieved_preview = [d.page_content[:150].replace('\n', ' ') for d in retrieved_docs]

        logging.info(f"[EVAL] Query='{user_question}' | Latency={latency}s | Retrieved={len(retrieved_docs)} chunks")
        for i, doc in enumerate(retrieved_preview, 1):
            logging.info(f"[EVAL] Chunk {i}: {doc}")

        # --- Display response in UI ---
        st.markdown(f"<div class='user-msg'>User: {user_question}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-msg'>Assistant: {response['answer']}</div>", unsafe_allow_html=True)
        st.session_state.chat_history = response['chat_history']

        # --- Show latency in Streamlit UI ---
        st.info(f"Response time: {latency} seconds")

    except ValueError as e:
        st.warning("Sorry, please try rephrasing your question.")
        logging.error(f"[ERROR] {str(e)}")


# ===============================================================
# Main App Logic
# ===============================================================
def main():
    add_custom_css()
    st.markdown("<div class='main-header'> Knowledge Assistant</div>", unsafe_allow_html=True)
    st.write("Integrates **Amazon Bedrock**, **Aurora**, and **pgvector** for conversational insights from enterprise knowledge.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='subheader'>Knowledge Source</div>", unsafe_allow_html=True)
        options = ["PDF Documents", "S3 Bucket", "CSV File", "PowerPoint", "Word Document"]
        selected_source = st.radio("Select your data source:", options)

        st.markdown("<div class='subheader'>Model Selection</div>", unsafe_allow_html=True)
        llm_options = [
            'amazon.nova-pro-v1:0',
            'amazon.titan-text-express-v1',
            'amazon.titan-text-lite-v1',
            'amazon.titan-tg1-large',
            'anthropic.claude-3-5-sonnet-20240620-v1:0',
            'anthropic.claude-3-haiku-20240307-v1:0',
            'anthropic.claude-3-opus-20240229-v1:0'
        ]
        selected_llm = st.selectbox("Select a model", llm_options, index=2)

        # --- File Upload & Processing ---
        all_text = ""

        if selected_source == "PDF Documents":
            files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
            if st.button("Process PDF Documents") and files:
                for f in files:
                    all_text += get_pdf_text([f])

        elif selected_source == "Word Document":
            files = st.file_uploader("Upload Word files", type=["docx"], accept_multiple_files=True, key="word_files")
            if files:
                if st.button("Process Word Documents"):
                    st.session_state.processing_status = "Processing Word documents..."
                    st.session_state.all_text = ""
                    with st.spinner(st.session_state.processing_status):
                        for f in files:
                            st.sidebar.write(f"[DEBUG] Processing file: {f.name}")
                            path = save_uploaded_file(f)
                            try:
                                loader = Docx2txtLoader(path)
                                docs = loader.load()
                                for d in docs:
                                    st.session_state.all_text += d.page_content + " "
                                st.sidebar.write(f"[DEBUG] Finished loading file: {f.name}")
                            finally:
                                os.remove(path)
                    st.session_state.processing_status = "Word documents processed successfully!"
                    st.success(st.session_state.processing_status)

        elif selected_source == "PowerPoint":
            files = st.file_uploader("Upload PowerPoint files", type=["pptx"], accept_multiple_files=True)
            if st.button("Process PowerPoint Files") and files:
                for f in files:
                    path = save_uploaded_file(f)
                    loader = UnstructuredPowerPointLoader(path)
                    docs = loader.load()
                    all_text += " ".join([d.page_content for d in docs])
                    os.remove(path)

        elif selected_source == "CSV File":
            file = st.file_uploader("Upload a CSV file", type=["csv"])
            if st.button("Process CSV File") and file:
                loader = CSVLoader(file)
                docs = loader.load()
                all_text = " ".join([d.page_content for d in docs])

        # elif selected_source == "YouTube":
        #     url = st.text_input("Enter YouTube URL")
        #     if st.button("Process YouTube Video") and url:
        #         loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        #         docs = loader.load()
        #         all_text = " ".join([d.page_content for d in docs])

        elif selected_source == "S3 Bucket":
            s3_client = boto3.client('s3')
            BUCKET_NAME = 'vectors-warehouse'
            PREFIX = 'documentEmbeddings/'
            
            # Use a list to accumulate keys
            keys = []
            
            try:
                # Use a paginator to handle more than 1000 files (good practice)
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)

                for page in pages:
                    for obj in page.get('Contents', []):
                        # The full key, e.g., 'documentEmbeddings/my_file.pdf'
                        full_key = obj['Key']
                        
                        # Exclude the folder itself (which typically ends with a /)
                        if not full_key.endswith('/'):
                            # Extract the filename only
                            filename = full_key[len(PREFIX):]
                            keys.append(filename)
                
                if not keys:
                    st.warning(f"No documents found in S3 bucket '{BUCKET_NAME}' under prefix '{PREFIX}'.")

            except Exception as e:
                st.error(f"Error listing S3 objects: {e}")
                keys = [] # Ensure keys is empty on error

            key_selected = st.selectbox("Select S3 Document", keys)
            
            # Update the S3FileLoader to use the full key path
            if st.button("Load from S3") and key_selected:
                full_key_path = f"{PREFIX}{key_selected}"
                st.write(f"[DEBUG] Loading S3 key: {full_key_path}")
                loader = S3FileLoader(BUCKET_NAME, full_key_path)
                docs = loader.load()
                all_text = " ".join([d.page_content for d in docs])

        if all_text:
            chunks = get_text_chunks(all_text)
            vectorstore = get_vectorstore(chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, selected_llm)

    # --- Chat Section ---
    with col2:
        st.markdown("<div class='subheader'>Ask a Question</div>", unsafe_allow_html=True)
        question = st.text_input("Enter your query:")
        if st.button("Submit Question") and question:
            handle_userinput(question)

        st.markdown("<div class='subheader'>Conversation History</div>", unsafe_allow_html=True)
        for msg in st.session_state.get("chat_history", []):
            role = "User" if msg.type == "human" else "Assistant"
            css_class = "user-msg" if role == "User" else "bot-msg"
            st.markdown(f"<div class='{css_class}'><b>{role}:</b> {msg.content}</div>", unsafe_allow_html=True)

    # Initialize defaults
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
