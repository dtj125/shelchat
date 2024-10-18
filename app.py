# import libraries
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st


# Set up AWS Bedrock Client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = "us-east-1")

#Create Vector Embedding - TITAN
bedrock_embeddings = BedrockEmbeddings(
    model_id = 'amazon.titan-embed-text-v1',
    client=bedrock
)

#Loading and Splitting PDFs
def data_ingestion():
    
    #loader = PyPDFDirectoryLoader("https://shel-underground-expressions.s3.amazonaws.com/public") 
    
    #PDFs inside of data folder
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    #Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

#Create vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Query documents
def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={"max_tokens_to_sample": 512})
    return llm

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. Please summarize with at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


# Streamlit UI


def main():
    st.set_page_config(page_title="Shel Chat", layout="wide")

    st.title("Shel Chat AI: AI-powered PDF Document Assistant 💼")

    #st.markdown("### Ask questions directly from your PDF documents using advanced AI models!")

    user_question = st.text_input("Enter your question related to the PDF files", "")

    st.sidebar.title("Manage Vector Store")
    if st.sidebar.button("Update Vectors"):
        with st.spinner("Updating vectors..."):
            docs = data_ingestion()
            get_vector_store(docs)
            st.sidebar.success("Vectors updated successfully!")
            st.write(docs)

    st.sidebar.title("Select AI Model")
    model_choice = st.sidebar.radio("Choose a model to generate responses:", ("Claude", "LLaMA3"))

    if st.button("Get Response"):
        with st.spinner(f"Generating response with {model_choice}..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            if model_choice == "Claude":
                llm = get_claude_llm()
            else:
                llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.markdown(f"### {model_choice}'s Response")
            st.write(response)
            st.success("Response generated successfully!")
    st.write("THIS IS A TEST")
    

if __name__ == "__main__":
    main()