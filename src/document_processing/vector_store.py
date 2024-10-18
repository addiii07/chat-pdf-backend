import os
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from src.document_processing.pdf_extractor import extract_text_from_pdf

def create_or_load_vector_store(
    pdf_path, vector_store_dir="data/vector_stores", bm25_store_dir="data/bm25_stores"
):
    # Ensure the store directories exist
    os.makedirs(vector_store_dir, exist_ok=True)
    os.makedirs(bm25_store_dir, exist_ok=True)

    file_name = os.path.basename(pdf_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    chroma_index_dir = os.path.join(vector_store_dir, f"chroma_{file_name_without_ext}")
    bm25_index_file = os.path.join(bm25_store_dir, f"bm25_{file_name_without_ext}.pkl")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")

    if os.path.exists(chroma_index_dir) and os.path.exists(bm25_index_file):
        print(f"Loading existing Chroma store: {chroma_index_dir} and BM25 store: {bm25_index_file}")
        chroma_db = Chroma(persist_directory=chroma_index_dir, embedding_function=embeddings)
        with open(bm25_index_file, 'rb') as f:
            bm25_docs = pickle.load(f)
        bm25 = BM25Retriever.from_documents(bm25_docs)
    else:
        print(f"Creating new Chroma store: {chroma_index_dir} and BM25 store: {bm25_index_file}")
        pdf_text_with_pages = extract_text_from_pdf(pdf_path)
        
        bm25_docs = []
        for text, page_num in pdf_text_with_pages:
            bm25_docs.append(Document(page_content=text, metadata={"page": page_num}))

        chroma_db = Chroma.from_documents(
            documents=bm25_docs,
            embedding=embeddings,
            persist_directory=chroma_index_dir
        )
        chroma_db.persist()

        bm25 = BM25Retriever.from_documents(bm25_docs)
        with open(bm25_index_file, 'wb') as f:
            pickle.dump(bm25_docs, f)

    bm25_texts = [doc.page_content for doc in bm25_docs]

    return chroma_db, bm25, bm25_texts