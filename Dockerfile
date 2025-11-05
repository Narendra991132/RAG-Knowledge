# Start from Python
FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install only what you need
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir streamlit boto3 python-dotenv PyPDF2 langchain psycopg2-binary

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
