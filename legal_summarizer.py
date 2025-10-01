# legal_summarizer.py
import spacy
from transformers import pipeline

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Load summarization pipeline (BART or T5)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract key clauses
def extract_clauses(text):
    doc = nlp(text)
    clauses = {}

    # Keywords to detect clauses
    keywords = ["governing law", "termination", "confidentiality", "liability", "dispute resolution"]

    for sent in doc.sents:
        for key in keywords:
            if key in sent.text.lower():
                clauses[key] = clauses.get(key, []) + [sent.text.strip()]
    return clauses

# Function to summarize entire document
def summarize_document(text, max_len=130, min_len=40):
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]["summary_text"]

# Main
if __name__ == "__main__":
    with open("sample_legal.txt", "r", encoding="utf-8") as f:
        legal_text = f.read()

    # Extract clauses
    clauses = extract_clauses(legal_text)
    print("\n Extracted Clauses:")
    for k, v in clauses.items():
        print(f"\n-- {k.upper()} --")
        for clause in v:
            print(clause)

    # Summarize full document
    print("\n Summary of Document:")
    print(summarize_document(legal_text))
