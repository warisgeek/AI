import io
import pdfplumber
from docx import Document
import pandas as pd

CHUNK_SIZE = 1000

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]


def extract_text_from_pdf(file_stream):
    text = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def extract_text_from_docx(file_stream):
    doc = Document(file_stream)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text_from_file(uploaded_file):
    """Returns list of text chunks"""
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'):
        raw = extract_text_from_pdf(uploaded_file)
    elif filename.endswith('.docx'):
        raw = extract_text_from_docx(uploaded_file)
    elif filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        raw = df.to_csv(index=False)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
        raw = df.to_csv(index=False)
    else:
        raw = uploaded_file.read().decode('utf-8', errors='ignore')
    return chunk_text(raw)
