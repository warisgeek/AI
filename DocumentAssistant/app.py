import streamlit as st
from src.ingest import extract_text_from_file
from src.embeddings import Embedder
from src.vectorstore import FaissStore
from src.qa import QASession

st.title("Smart Document Assistant - Prototype")

uploader = st.file_uploader("Upload a document", type=["pdf","docx","csv","xlsx"])

if uploader:
    txt_chunks = extract_text_from_file(uploader)
    st.write(f"Extracted {len(txt_chunks)} chunks")

    embedder = Embedder()
    vectors = embedder.embed_documents(txt_chunks)

    store = FaissStore()
    store.add_documents(txt_chunks, vectors)

    st.success("Indexed document into FAISS")

    q = st.text_input("Ask a question about the document")
    if q:
        qa = QASession(store=store, embedder=embedder)
        answer = qa.answer(q)
        st.write("**Answer:**")
        st.write(answer)
