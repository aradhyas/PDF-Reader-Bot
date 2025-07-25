import fitz

def load_pdf(file):
    """
    Load text content from a PDF file object.
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of a specified size.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap): #so keeping overlap helps in keeping the context intact. sentences can loose value if split in the middle because of chunking. so we use overlap.
        chunks.append(text[i:i + chunk_size])
    return chunks