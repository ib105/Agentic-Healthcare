import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a 1-page (or multi-page) PDF medical report.
    Returns a single string of text.
    """
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def chunk_text(text: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits long text into overlapping chunks to handle llm token limits.
    Uses RecursiveCharacterTextSplitter for semantic chunking.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks
