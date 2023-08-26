from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

st.title("ChatPDF")
st.write("---")

uploaded_file = st.file_uploader("Choose a file")
st.write("---")

def pdf_to_document(upload_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
    return pages

# If a file is uploaded, execute the following code
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    
    text_splitter = RecursiveCharacterTextSplitter(    
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # embedding
    embeddings_model = OpenAIEmbeddings()
    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Take a question as input
    st.header("Ask about the content.")
    question = st.text_input("Enter your question:", "아내가 먹고 싶어하는 음식은 뭐야?")

    if st.button('Ask'):
        llm = ChatOpenAI(temperature=0) 
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        result = qa_chain({"query": question})
        st.write(result)
