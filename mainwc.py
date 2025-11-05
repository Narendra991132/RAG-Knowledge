import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import S3FileLoader, YoutubeLoader, UnstructuredPowerPointLoader, Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import PostgresChatMessageHistory
import boto3
import tempfile
import time
import hashlib
import secrets
import os
from dotenv import load_dotenv
import logging

# 🔇 Suppress excessive botocore logs for cleaner output
logging.getLogger('botocore').setLevel(logging.ERROR)

# ===============================================================
# 🧱 UI Styling Helpers
# ===============================================================
def styled_header(text):
    """Creates a green header bar with centered white text."""
    return f"""
    <div style="background-color:#4CAF50;text-align:center;padding:10px">
        <h1 style="color:white;">{text}</h1>
    </div>
    """

def styled_subheader(text, font_size="24px", color="#8A2BE2", background_color="#f0e5ff"):
    """Styled subheader box for section titles."""
    return f"""
    <div style="box-shadow:0 2px 10px #ddd; padding: 5px; background-color: {background_color};
                border-radius: 5px; margin: 10px 0; text-align: center;">
        <h3 style="color: {color}; font-size: {font_size}; margin: 0;">{text}</h3>
    </div>
    """

# ===============================================================
# 🔑 Utility Functions
# ===============================================================
def generate_session_id():
    """Generates a unique session ID using timestamp + random salt."""
    t = int(time.time() * 1000)
    r = secrets.randbelow(1000000)
    session_id = hashlib.md5(bytes(str(t) + str(r), 'utf-8'), usedforsecurity=False).hexdigest()
    print(f"[DEBUG] Generated session_id: {session_id}")
    return session_id

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(f"[DEBUG] Extracted {len(text)} characters from PDFs.")
    return text

def get_text_chunks(text):
    """Splits large text into manageable chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=512,
        chunk_overlap=103,
        length_function=len
    )
    chunks = splitter.split_text(text)
    print(f"[DEBUG] Created {len(chunks)} text chunks.")
    return chunks

# ===============================================================
# 🧩 Database / Vectorstore Setup
# ===============================================================
def get_vectorstore(text_chunks):
    """
    Creates a PGVector vectorstore in PostgreSQL (Aurora) using pgvector extension.
    """
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-west-2")
    try:
        if text_chunks is None:
            print("[DEBUG] Initializing empty PGVector connection...")
            return PGVector(connection_string=CONNECTION_STRING, embedding_function=embeddings)
        print(f"[DEBUG] Creating embeddings for {len(text_chunks)} chunks...")
        return PGVector.from_texts(texts=text_chunks, embedding=embeddings, connection_string=CONNECTION_STRING)
    except Exception as e:
        print("[ERROR] Vectorstore initialization failed:", str(e))
        raise e

# ===============================================================
# 🧠 Bedrock LLM Setup
# ===============================================================
def get_bedrock_llm(selected_llm):
    """Creates an LLM interface to Amazon Bedrock for the chosen model."""
    print(f"[DEBUG] Initializing Bedrock LLM: {selected_llm}")
    if selected_llm.startswith("anthropic."):
        return Bedrock(model_id=selected_llm, model_kwargs={'max_tokens_to_sample': 4096})
    elif selected_llm.startswith("amazon.titan"):
        return Bedrock(model_id=selected_llm, model_kwargs={"maxTokenCount": 4096, "temperature": 0})
    elif selected_llm == "amazon.nova-pro-v1:0":
        return Bedrock(model_id=selected_llm)
    else:
        raise ValueError(f"Unsupported LLM selected: {selected_llm}")

# ===============================================================
# 🧵 Conversation Chain Setup
# ===============================================================
def get_conversation_chain(vectorstore, selected_llm):
    """
    Builds the conversational retrieval chain:
    - Connects LLM
    - Sets up Postgres-backed chat memory
    - Links the retriever to handle context-based answers
    """
    llm = get_bedrock_llm(selected_llm)
    _connection_string = CONNECTION_STRING.replace('+psycopg2', '').replace(':5432', '')

    print("[DEBUG] Connecting PostgresChatMessageHistory with session_id...")
    message_history = PostgresChatMessageHistory(connection_string=_connection_string, session_id=generate_session_id())

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print("[DEBUG] Conversation chain initialized successfully.")
    return conversation_chain

# ===============================================================
# 💬 Chat UI
# ===============================================================
def color_text(text, color="black"):
    return f'<span style="color:{color}">{text}</span>'

def handle_userinput(user_question):
    """Handles user chat inputs and displays bot responses."""
    print(f"[DEBUG] User asked: {user_question}")
    try:
        response = st.session_state.conversation({'question': user_question})
        st.markdown(color_text(f"👤 USER : {user_question}", color="blue"), unsafe_allow_html=True)
        st.markdown(color_text(f"🤖 BOT : {response['answer']}", color="green"), unsafe_allow_html=True)
    except ValueError as e:
        print("[ERROR] Chat processing failed:", e)
        st.warning("😞 Sorry, please ask again in a different way.")
        return

    st.session_state.chat_history = response['chat_history']

# ===============================================================
# 🚀 Main App Logic
# ===============================================================
def main():
    st.markdown(styled_header("Unified AI Q&A: Harnessing pgvector, Amazon Aurora & Amazon Bedrock 📚🦜"), unsafe_allow_html=True)

    # User selects input data source
    options = ["📄 PDFs", "☁️ S3 Bucket", "📺 Youtube", "📑 CSV", "🖼️ PPT", "📝 Word"]
    st.markdown(styled_subheader("📌 Select a source 📌"), unsafe_allow_html=True)
    selected_source = st.radio("", options)
    print(f"[DEBUG] Selected source: {selected_source}")

    # LLM selection
    llm_options = [
        'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'anthropic.claude-3-haiku-20240307-v1:0',
        'anthropic.claude-3-opus-20240229-v1:0',
        'amazon.nova-pro-v1:0',
        'amazon.titan-text-express-v1',
        'amazon.titan-text-lite-v1',
        'amazon.titan-tg1-large'
    ]
    selected_llm = st.radio("Choose an LLM", options=llm_options, index=5)
    print(f"[DEBUG] Selected LLM: {selected_llm}")

    # Data source processing logic
    if selected_source == "📄 PDFs":
        pdf_docs = st.file_uploader("📥 Upload your PDFs:", type="pdf", accept_multiple_files=True)
        if st.button("🔄 Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore, selected_llm)

    elif selected_source == "☁️ S3 Bucket":
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket='aurora-genai-2023', Prefix='documentEmbeddings/')
        keys = [obj['Key'].split('/')[1] for obj in response['Contents']][1:]
        user_input = st.selectbox("Select S3 document:", keys)
        if st.button("Process"):
            with st.spinner("Loading S3 file..."):
                loader = S3FileLoader("aurora-genai-2023", f"documentEmbeddings/{user_input}")
                docs = loader.load()
                for doc in docs:
                    chunks = get_text_chunks(doc.page_content)
                    vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore, selected_llm)

    # ... (Other file types omitted for brevity, same pattern applies)

    # Sidebar chat input
    st.sidebar.header("🗣️ Chat with Bot")
    question = st.sidebar.text_input("💬 Ask a question:")
    if question:
        handle_userinput(question)

    # Initialize default states
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(get_vectorstore(None), selected_llm)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

# ===============================================================
# 🧩 Entry Point
# ===============================================================
if __name__ == '__main__':
    load_dotenv()
    print("[DEBUG] Environment variables loaded.")

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
