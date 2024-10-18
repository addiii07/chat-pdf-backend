import os
from src.document_processing.pdf_extractor import extract_text_from_pdf
from src.document_processing.vector_store import create_or_load_vector_store
from src.search.hybrid_search import process_questions

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    pdf_path = "data/docs/ToR_70c4a009-c892-4468-bc1c-a48df7191a5c.pdf"
    chroma_db, bm25_retriever, bm25_texts = create_or_load_vector_store(pdf_path)
    questions = [
        "What is the title of the document?",
        "What is the full form of NFSA?"
        "What details can beneficiaries check?",
        "What is the duration of the warranty period?",
        "How the ranking will be shown?"

        "What is the minimum number of years of IT/ITeS services experience required to earn the first level of marks in the technical evaluation?",
        "What is the validity period of e-Bids after opening?",
        "What is the maximum number of consortium members allowed?",
        "How much is the Earnest Money Deposit (EMD) required?",
        "What is the amount of the Tender Fee including GST?",
    ]

    pdf_text_with_pages = extract_text_from_pdf(pdf_path)
    responses = process_questions(
        questions, pdf_text_with_pages, chroma_db, bm25_retriever, bm25_texts
    )

    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"Question {i}: {question}")
        print(f"Response: {response['response']}")
        print(f"Page: {response['pages']}\n")
