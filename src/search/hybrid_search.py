import numpy as np
from langchain_community.llms import Ollama

def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

def hybrid_search(query, chroma_db, bm25_retriever, bm25_texts, k=10):
    # Vector search using Chroma
    chroma_results = chroma_db.similarity_search_with_score(query, k=k)
    chroma_indices = [bm25_texts.index(result[0].page_content) for result in chroma_results]
    chroma_scores = [-score for _, score in chroma_results]  # Negate scores to align with BM25 scores

    # BM25 search
    bm25_results = bm25_retriever.get_relevant_documents(query)
    bm25_indices = [bm25_texts.index(doc.page_content) for doc in bm25_results[:k]]
    bm25_scores = [doc.metadata.get('score', 0) for doc in bm25_results[:k]]

    # Combine results
    combined_indices = np.concatenate((chroma_indices, bm25_indices))
    combined_scores = np.concatenate((chroma_scores, bm25_scores))
    sorted_indices = np.argsort(combined_scores)[::-1][:k]  # Sort in descending order

    contexts_with_pages = [(bm25_texts[idx], idx) for idx in combined_indices[sorted_indices]]

    return contexts_with_pages

def process_questions(questions, pdf_text_with_pages, chroma_db, bm25_retriever, bm25_texts):
    responses = []
    llm = Ollama(model='llama3:8b')

    for question in questions:
        try:
            contexts_with_pages = hybrid_search(question, chroma_db, bm25_retriever, bm25_texts, k=10)

            retrieved_contexts = []
            for content, idx in contexts_with_pages:
                if content not in retrieved_contexts:
                    retrieved_contexts.append(content)
                    page_num = pdf_text_with_pages[idx][1]
                    print(f"Retrieved chunk from page {page_num}: {content}")

            combined_context = " ".join(retrieved_contexts)
            combined_input = f"Context: {combined_context}\n\nQuery: {question}"
            print(f"Combined input for the model:\n{combined_input}\n")

            try:
                response = llm.invoke(combined_input)
                response_text = response if isinstance(response, str) else response.get('text', 'No response text available.')
            except Exception as e:
                print(f"Error calling Ollama: {str(e)}")
                response_text = f"Error: Unable to generate response. Please try again later. (Error: {str(e)})"

            highest_similarity = 0
            final_page = None

            for content, idx in contexts_with_pages:
                similarity_score = jaccard_similarity(content, response_text)
                if similarity_score > highest_similarity:
                    highest_similarity = similarity_score
                    final_page = pdf_text_with_pages[idx][1]

            if final_page:
                print(f"Final Page: {final_page} (Jaccard Similarity: {highest_similarity:.2f})")
            else:
                print("No page found for this question.")

            response_with_page_info = {
                "question": question,
                "response": response_text,
                "page": final_page if final_page else []
            }
            responses.append(response_with_page_info)
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            responses.append({
                "question": question,
                "response": f"Error: Unable to process the question. Please try again later. (Error: {str(e)})",
                "page": []
            })

    return responses
