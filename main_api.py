import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import glob
from sqlalchemy.orm import Session
from src.document_processing.pdf_extractor import extract_text_from_pdf
from src.document_processing.vector_store import create_or_load_vector_store
from src.search.hybrid_search import process_questions
from src.utils.file_utils import generate_file_hash
from fastapi.middleware.cors import CORSMiddleware
from database import get_db, QAFeedback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Dictionary to store PDF information
pdf_store = {}

class Query(BaseModel):
    query: str
    pdf_id: str

class MultipleQueries(BaseModel):
    queries: List[str]
    pdf_id: str

class FeedbackModel(BaseModel):
    question: str
    answer: str
    feedback: str
    pdf_id: str

def find_duplicate_pdf(new_file_path):
    new_file_hash = generate_file_hash(new_file_path)
    for pdf_id, pdf_info in pdf_store.items():
        if generate_file_hash(pdf_info["path"]) == new_file_hash:
            return pdf_id
    return None

def load_existing_stores():
    vector_store_dir = "data/vector_stores"
    pdf_dir = "data/docs"

    for pdf_file in glob.glob(f"{pdf_dir}/*.pdf"):
        file_name = os.path.basename(pdf_file)
        pdf_id = file_name.split("_")[-1].split(".")[0]

        chroma_dir = f"{vector_store_dir}/chroma_{file_name.replace('.pdf', '')}"

        if os.path.exists(chroma_dir):
            chroma_db, bm25_retriever, bm25_texts = create_or_load_vector_store(
                pdf_file
            )
            pdf_store[pdf_id] = {
                "path": pdf_file,
                "chroma_db": chroma_db,
                "bm25_retriever": bm25_retriever,
                "bm25_texts": bm25_texts,
            }
            print(f"Loaded existing stores for PDF: {file_name}")

@app.on_event("startup")
async def startup_event():
    load_existing_stores()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    temp_file_path = f"data/docs/temp_{file.filename}"

    # Save the uploaded file temporarily
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Check for duplicates
    duplicate_id = find_duplicate_pdf(temp_file_path)
    if duplicate_id:
        os.remove(temp_file_path)  # Remove the temporary file
        return {"pdf_id": duplicate_id, "message": "PDF already exists in the system"}

    # If no duplicate, proceed with the upload
    pdf_id = str(uuid.uuid4())
    original_filename = file.filename.replace(".pdf", "")
    safe_filename = "".join(c if c.isalnum() else "_" for c in original_filename)
    new_filename = f"{safe_filename}_{pdf_id}.pdf"

    pdf_path = f"data/docs/{new_filename}"
    os.rename(temp_file_path, pdf_path)

    chroma_db, bm25_retriever, bm25_texts = create_or_load_vector_store(pdf_path)

    pdf_store[pdf_id] = {
        "path": pdf_path,
        "chroma_db": chroma_db,
        "bm25_retriever": bm25_retriever,
        "bm25_texts": bm25_texts,
    }

    return {"pdf_id": pdf_id, "message": "PDF uploaded and processed successfully"}

@app.post("/query")
async def query(query: Query, db: Session = Depends(get_db)):
    if query.pdf_id not in pdf_store:
        raise HTTPException(status_code=404, detail="PDF not found")

    pdf_info = pdf_store[query.pdf_id]
    pdf_text_with_pages = extract_text_from_pdf(pdf_info["path"])

    responses = process_questions(
        [query.query],
        pdf_text_with_pages,
        pdf_info["chroma_db"],
        pdf_info["bm25_retriever"],
        pdf_info["bm25_texts"],
    )

    response = responses[0]

    # Save question and answer to the database
    qa_feedback = QAFeedback(
        question=query.query,
        answer=response["response"],
        pdf_id=query.pdf_id
    )
    db.add(qa_feedback)
    db.commit()

    return response

@app.post("/feedback")
async def save_feedback(feedback: FeedbackModel, db: Session = Depends(get_db)):
    qa_feedback = db.query(QAFeedback).filter(
        QAFeedback.question == feedback.question,
        QAFeedback.answer == feedback.answer,
        QAFeedback.pdf_id == feedback.pdf_id
    ).first()

    if qa_feedback:
        qa_feedback.feedback = feedback.feedback
        db.commit()
        return {"message": "Feedback saved successfully"}
    else:
        raise HTTPException(status_code=404, detail="Question-Answer pair not found")

@app.post("/multiple_queries")
async def multiple_queries(queries: MultipleQueries, db: Session = Depends(get_db)):
    if queries.pdf_id not in pdf_store:
        raise HTTPException(status_code=404, detail="PDF not found")

    pdf_info = pdf_store[queries.pdf_id]
    pdf_text_with_pages = extract_text_from_pdf(pdf_info["path"])

    responses = process_questions(
        queries.queries,
        pdf_text_with_pages,
        pdf_info["chroma_db"],
        pdf_info["bm25_retriever"],
        pdf_info["bm25_texts"],
    )

    # Save questions and answers to the database
    for query, response in zip(queries.queries, responses):
        qa_feedback = QAFeedback(
            question=query,
            answer=response["response"],
            pdf_id=queries.pdf_id
        )
        db.add(qa_feedback)
    db.commit()

    return responses

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)