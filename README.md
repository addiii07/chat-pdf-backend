# PDF Question Answering System

This project implements a question answering system for PDF documents using hybrid search (vector and BM25) and language models. It provides an API for uploading PDFs, creating vector stores, and querying the documents.

## Features

- PDF text extraction and chunking
- Hybrid search combining vector search (Chroma) and BM25
- Language model integration for generating answers (using Ollama)
- FastAPI-based API for PDF upload and querying
- Persistence of vector stores for previously uploaded PDFs

## Project Structure

```
pdf-qa-system/
│
├── data/
│   ├── docs/             # Directory for storing uploaded PDFs
│   ├── vector_stores/    # Directory for Chroma vector stores
│   └── bm25_stores/      # Directory for BM25 stores
│
├── src/
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py
│   │   └── vector_store.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_model.py
│   ├── search/
│   │   ├── __init__.py
│   │   └── hybrid_search.py
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py
│
├── main.py               # Script for running queries directly
├── main_api.py           # FastAPI application for the API
├── requirements.txt
└── README.md
```

## Setup

1. Ensure you have Python 3.8+ installed on your system.

2. Navigate to the project directory.

3. Create a virtual environment:
   ```
   python -m venv .venv
   ```

4. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On macOS and Linux: `source .venv/bin/activate`

5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Set up the Ollama server:
   - Follow the installation instructions at: https://github.com/jmorganca/ollama
   - Ensure the Ollama server is running at http://localhost:11434/ before executing the scripts

## Usage

### Running the API

1. Start the API server:
   ```
   python main_api.py
   ```
   The API will be available at http://localhost:8000

2. Use the following endpoints:

   - Upload a PDF:
     ```
     POST /upload_pdf
     Content-Type: multipart/form-data
     file: [PDF file]
     ```

   - Query a single question:
     ```
     POST /query
     Content-Type: application/json
     {
       "query": "Your question here",
       "pdf_id": "PDF_ID_returned_from_upload"
     }
     ```

   - Query multiple questions:
     ```
     POST /multiple_queries
     Content-Type: application/json
     {
       "queries": ["Question 1", "Question 2", ...],
       "pdf_id": "PDF_ID_returned_from_upload"
     }
     ```

### Running the Original Script

To use the original script without the API:

1. Place your PDF file in the project root directory.

2. Modify the `questions` list in `main.py` to add or change the questions you want to ask about the PDF document.

3. Run the main script:
   ```
   python main.py
   ```

## Troubleshooting

- If you encounter OpenMP-related errors, add the following line at the beginning of your Python scripts:
  ```python
  import os
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  ```
- Ensure that the Ollama server is running before making queries.
- If you're having issues with specific PDFs, try re-uploading them to generate new vector stores.

## Note

This project uses the Ollama library, which requires separate installation and setup of the Ollama server. Make sure to complete the Ollama setup before running the scripts or API.

