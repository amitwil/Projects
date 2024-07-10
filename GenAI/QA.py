from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
import streamlit as st
import os
import cryptography
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings



from dotenv import load_dotenv
load_dotenv()

## load the API KEY 
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HF_API_KEY")

loader = PyPDFDirectoryLoader("./Data")
documents = loader.load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents= text_splitter.split_documents(documents)
print(len(final_documents))
# print(final_documents[0])
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ## Embedding Using Huggingface
# huggingface_embeddings=HuggingFaceBgeEmbeddings(
#     model_name='sentence-transformers/all-MiniLM-L6-v2',      #sentence-transformers/all-MiniLM-l6-v2
#     model_kwargs={'device':'cpu'},
#     encode_kwargs={'normalize_embeddings':True}

# )
huggingface_embeddings = HuggingFaceEndpointEmbeddings(
    model= "sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
print('huggingface_embeddings',huggingface_embeddings)

## VectorStore Creation
vectorstore=FAISS.from_documents(final_documents,huggingface_embeddings)
st.title("TMForum Search demo")
print('title done')
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# hf=HuggingFaceHub(
#     # repo_id="mistralai/Mistral-7B-v0.3",
#     repo_id="mistralai/Mistral-7B-v0.1",
#     model_kwargs={"temperature":0.6,"max_length":500},
#     huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]  
# )
from langchain_huggingface import HuggingFaceEndpoint
print('inm model')
hf=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    # repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"max_length": 200}, 
    temperature=0.7,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
print('modelling done')
prompt=ChatPromptTemplate.from_template(
"""
{context}
<done>
Question: {input}

"""
)

document_chain = create_stuff_documents_chain(hf, prompt)
#retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})
retrieval_chain = create_retrieval_chain(retriever, document_chain)
prompt=st.text_input("Input YOUR prompt here")
print('PROMPT',prompt)
if prompt:
    # start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    # print("Response time :",time.process_time()-start)
    # print(response)
    # print('answerssss:',response['answer'])
    # Adjusting beam search parameters
    
    # unique_responses = set()
    # for output in response['answer']:
    #     if output not in unique_responses:
    #         unique_responses.add(output)
    #         # Process the unique response further

    #st.write(response)
    # print('unique response',unique_responses)
    st.write(response['answer'])
    # ans=response['answer']
    # parts = ans.split("<done>")
    # st.write(parts[1])
    

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("________________________________")
