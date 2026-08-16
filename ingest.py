import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader(
    "./data"
).load_data()

print(f"Loaded {len(documents)} documents")

pc = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pc.Index("simplyrag")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("Documents successfully ingested!")