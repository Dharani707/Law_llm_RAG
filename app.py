import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain




from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret('HF_TOKEN')
HUGGINGFACEHUB_API_KEY = user_secrets.get_secret('HUGGINGFACEHUB_API_KEY')
LANGCHAIN_API_KEY = user_secrets.get_secret('LANGCHAIN_API_KEY')




def HuggingFace_Embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = 'mixedbread-ai/mxbai-embed-large-v1', 
                                      model_kwargs = {'device' : 'cuda'}, 
                                      encode_kwargs = {'normalize_embeddings' : True})
    return embeddings


llm = CTransformers(model="TheBloke/Llama-2-13B-Ensemble-v5-GGUF", 
                    model_file="llama-2-13b-ensemble-v5.Q5_K_M.gguf",
                    model_type="llama",
                    gpu_layers = 20)






def session_state():
    
    if 'vectors' not in st.session_state:
        
        st.session_state.embeddings = HuggingFace_Embeddings()
        st.session_state.loader = PyPDFDirectoryLoader('/kaggle/input/documents/Law documents')
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.session_state.vectors.save_local('faiss_index')
        st.session_state.new_db = FAISS.load_local('faiss_index', st.session_state.embeddings, allow_dangerous_deserialization = True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
st.title('Law guidence with llama2')
prompt = ChatPromptTemplate.from_template(
                                          """
                                          Answer the questions based on the provided context only.
                                          understand the question and give the correct law advise and suggestions. 
                                          please don't share false information.
                                          <context>
                                          {context}
                                          <context>
                                          Questions : {input}
                                          """
                                         )
input_prompt = st.text_input("Enter the question realted to LAW")
if st.button("Document embedding"):
    session_state()
    st.write("Database is ready..")

import time

if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start=time.process_time()
    
    response = retrieval_chain.invoke({'input':input_prompt})

    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
