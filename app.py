# Import libraries
import os
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

# Set up AWS clients
s3 = boto3.client('s3')
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Create Vector Embedding - TITAN
bedrock_embeddings = BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v1',
    client=bedrock
)

# Loading and Splitting PDFs from S3
def data_ingestion(bucket_name, prefix=''):
    # List objects in the specified S3 bucket
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    documents = []

    # Download each PDF file from S3
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            if file_key.endswith('.pdf'):
                local_file_path = f"/tmp/{os.path.basename(file_key)}"  # Temporary local path
                s3.download_file(bucket_name, file_key, local_file_path)  # Download the file

                # Load the PDF
                loader = PyPDFDirectoryLoader("/tmp")
                loaded_docs = loader.load()
                documents.extend(loaded_docs)

                # Debugging: Print the name of the loaded document
                print(f"Loaded document from S3: {local_file_path}")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    # Debugging: Print total documents after splitting
    print(f"Total documents after splitting: {len(docs)}")
    return docs

# Create vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    print(f"Vector store created with {len(docs)} documents.")  # Debugging line
    vectorstore_faiss.save_local("faiss_index")

# Query documents
def get_llm(model_choice):
    if model_choice == "Claude":
        return Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={"max_tokens_to_sample": 512})
    else:
        return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

# Define prompt template
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

    # Debugging: Print query and answer details
    print(f"Query: {query} - Answer: {answer}")
    return answer['result']

# Streamlit UI
def main():
    st.set_page_config(page_title="Shel Chat", layout="wide")
    st.title("Shel Chat AI: AI-powered PDF Document Assistant 💼")

    # Input for S3 Bucket and Prefix
    bucket_name = st.text_input("Enter S3 Bucket Name:")
    prefix = st.text_input("Enter Prefix (optional):")

    user_question = st.text_input("Enter your question related to the PDF files", "")

    st.sidebar.title("Manage Vector Store")
    if st.sidebar.button("Update Source Documents"):
        with st.spinner("Updating sourcce documents..."):
            docs = data_ingestion(bucket_name, prefix)
            get_vector_store(docs)
            st.sidebar.success("Vectors updated successfully!")

    st.sidebar.title("Select AI Model")
    model_choice = st.sidebar.radio("Choose a model to generate responses:", ("LLaMA3"))

    if st.button("Get Response"):
        with st.spinner(f"Generating response with {model_choice}..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llm(model_choice)
            response = get_response_llm(llm, faiss_index, user_question)
            st.markdown(f"### {model_choice}'s Response")
            st.write(response)
            st.success("Response generated successfully!")

if __name__ == "__main__":
    main()
