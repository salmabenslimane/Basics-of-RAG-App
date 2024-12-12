from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# 1. CREATING TEXT CHUNKS

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks


# 2. CREATING VECTOR STORE 

def create_vector_store (chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #for french usage
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# 3. RETRIEVAL

def build_retrieval_qa_chain(vector_store, model_name="tiiuae/falcon-7b-instruct"):
    retriever = vector_store.as_retriever()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=hf_pipeline),
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

# 5. Streamlit App
def run_app():
    st.title("Multi-PDF Chatbot")

    # Upload files
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        all_chunks = []
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            chunks = split_text_into_chunks(text)
            all_chunks.extend(chunks)

        # Update vector store
        vector_store = create_vector_store(all_chunks)

        # Build the Retrieval QA Chain
        qa_chain = build_retrieval_qa_chain(vector_store)

        # Question answering
        query = st.text_input("Ask a question about the PDFs:")
        if query:
            answer = qa_chain.run(query)
            st.write("Answer:", answer["result"])
            st.write("Sources:", answer["source_documents"])


# Run the app
if __name__ == "__main__":
    run_app()