from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import PyPDF2
import io
import os
import streamlit as st

st.title(""" Find any information you're looking for in your PDF""")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
else:
    domanda = st.text_input("Ask me anything")
    uploaded_file = st.file_uploader("Upload your file", type=["pdf"])
    button = st.button("Run")

    if button:
        if uploaded_file is not None and uploaded_file.name.endswith('.pdf'):
            pdf_file = io.BytesIO(uploaded_file.getvalue())
            reader = PyPDF2.PdfReader(pdf_file)

            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text

            text_splitter = CharacterTextSplitter(        
                separator = "\n",
                chunk_size = 1000,
                chunk_overlap  = 200,
                length_function = len,
            )
            texts = text_splitter.split_text(raw_text)

            embeddings = OpenAIEmbeddings(api_key=openai_api_key)

            docsearch = FAISS.from_texts(texts, embeddings)

            chain = load_qa_chain(OpenAI(api_key=openai_api_key), chain_type="stuff")

            query = domanda
            docs = docsearch.similarity_search(query)
            risposta = chain.run(input_documents=docs, question=query, verbose=True)

            st.write(risposta)
