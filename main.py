import os
import fitz  # PyMuPDF for PDFs
import docx
from transformers import pipeline

# Load NLP models
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(file_path):
    """Extract text from PDF."""
    doc = fitz.open(file_path)
    return "\n".join([page.get_text("text") for page in doc])

def extract_text_from_docx(file_path):
    """Extract text from DOCX."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    """Extract text from TXT."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_document(file_path):
    """Determine file type and extract text."""
    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    elif ext == "txt":
        return extract_text_from_txt(file_path)
    else:
        print("❌ Unsupported format. Only PDF, DOCX, TXT are allowed.")
        return None

def summarize_document(text, max_chunk_size=800):
    """Summarize document content in chunks to handle large files."""
    print("\nGenerating summary...")

    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summary_texts = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
        summary_texts.append(summary[0]["summary_text"])

    final_summary = " ".join(summary_texts)
    
    print("\nSummary:", final_summary)
    return final_summary

def classify_question_length(question):
    """Classifies question type: Short, Medium, or Long."""
    short_keywords = ["who", "when", "where", "how many", "how much"]
    long_keywords = ["explain", "describe", "what are", "how does", "why", "impact", "effects"]

    if any(keyword in question.lower() for keyword in short_keywords):
        return "short"
    elif any(keyword in question.lower() for keyword in long_keywords):
        return "long"
    else:
        return "medium"

def chat_with_document(text, max_chunk_size=1500):
    """Interactive chat with document and provides detailed responses."""
    print("\n💬 Chatbot ready! Ask a question based on the document.")

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer_length = classify_question_length(query)
        best_chunk = find_best_chunk(text, query, max_chunk_size)

        if answer_length == "short":
            answer = qa_pipeline(question=query, context=best_chunk)
            print("\nAnswer (Short & Precise):", answer["answer"])

        elif answer_length == "long":
            print("\nGenerating a detailed response...")
            detailed_answer = summarize_text(best_chunk, max_length=500, min_length=150)
            print("\nAnswer (Detailed):", detailed_answer)

        else:
            print("\nGenerating a well-explained response...")
            medium_answer = summarize_text(best_chunk, max_length=300, min_length=100)
            print("\nAnswer (Medium-Length Explanation):", medium_answer)

def summarize_text(text, max_length=300, min_length=100):
    """Generates a more structured, well-explained answer."""
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

def find_best_chunk(text, query, max_chunk_size=1500):
    """Finds the most relevant text chunk for answering the question."""
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    best_chunk = max(chunks, key=lambda chunk: query.lower() in chunk.lower())
    return best_chunk

if __name__ == "__main__":
    file_path = input("Enter the document path (PDF, DOCX, TXT): ").strip()
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
    else:
        document_text = load_document(file_path)
        if document_text:
            print("\n✅ Document loaded successfully!")
            summarize_document(document_text)
            chat_with_document(document_text)
