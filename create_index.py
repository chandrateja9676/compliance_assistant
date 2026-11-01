import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load your API key from .env
load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "compliance-docs"

print("Creating vector index...")
pc.create_index(
    name=index_name,
    dimension=384, # Required for the free HuggingFace model
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1" # You can change this to your preferred region
    )
)
print("Index created successfully!")