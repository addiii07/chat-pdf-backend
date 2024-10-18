from langchain_community.document_loaders import PyMuPDFLoader

def extract_text_from_pdf(pdf_path, chunk_size=200, overlap=50):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    
    text_with_pages = []
    for page_num, page in enumerate(data, start=1):
        text = page.page_content
        if text.strip():
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                text_with_pages.append((chunk, page_num))
                # print(f"Extracted chunk from page {page_num}: {chunk[:50]}...")
                start += chunk_size - overlap
        else:
            print(f"Page {page_num} is empty.")
    
    return text_with_pages