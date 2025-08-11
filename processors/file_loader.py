import fitz
import docx
import os
import re
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
from processors.text_cleaner import clean_text
from processors.chunker import chunk_text

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_eml(file_path):
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_content()
            elif part.get_content_type() == "text/html":
                soup = BeautifulSoup(part.get_content(), "html.parser")
                return soup.get_text(separator=" ")
    else:
        if msg.get_content_type() == "text/plain":
            return msg.get_content()
        elif msg.get_content_type() == "text/html":
            soup = BeautifulSoup(msg.get_content(), "html.parser")
            return soup.get_text(separator=" ")
    return ""

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        raw_text = extract_text_from_docx(file_path)
    elif ext == ".eml":
        raw_text = extract_text_from_eml(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    cleaned = clean_text(raw_text)
    return chunk_text(cleaned, chunk_size=300, overlap=50)
