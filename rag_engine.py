import os
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize the API configurations

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_document(file_path):
    # Load document
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    # Convert chunks to vector embeddings and store in Pinecone
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=index_name,
    )
    return len(splits)

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question(query):
    # Retrieve relevant chunks from Pinecone and generate an answer
    
    # Initialize the LLM and vector store
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Temperature 0 for factual accuracy
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Retrieve top 3 most similar chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define the strict compliance prompt
    system_prompt = (
        "You are an expert compliance document assistant. Use the following retrieved "
        "context to answer the user's question. If the answer is not contained in the "
        "context, explicitly state that you do not know. Do not hallucinate."
        "\n\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # Building the modern LCEL RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Execute and return the answer as a string
    return rag_chain.invoke(query)