import streamlit as st
from qa_pipeline import create_vector, get_answer

from utils import load_pdf, chunk_text

with open("data/mahabharata.pdf", "rb") as file:
    text = load_pdf(file)
    chunks = chunk_text(text)
    print(chunks[:3])

st.set_page_config(page_title="Mahabharata Q&A", page_icon=":book:")
st.title("Get Answers from Mahabharata")

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
question = st.text_input("What do you want to know about Mahabharata?")

if pdf_file and question:
    text = load_pdf(pdf_file)
    chunks = chunk_text(text)
    st.success(f"Loaded PDF with {len(chunks)} chunks.")

    with st.spinner("Creating vector embeddings..."):
        vector_store, embedding_model = create_vector(chunks)
        st.success("📦 Document is ready for Q&A!")

    if question:
        with st.spinner("Generating answer..."):
            answer = get_answer(question, vector_store, embedding_model)
            st.markdown("### 🧠 Answer")
            st.write(answer)