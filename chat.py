import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = OpenAILike(
    model="meta-llama/llama-3.3-70b-instruct",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    is_chat_model=True,
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

pinecone_index = pc.Index("simplyrag")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=(
        "You are a helpful assistant that answers questions "
        "using the provided documents. "
        "If the answer cannot be found in the documents, "
        "say that you don't know rather than making something up."
    )
)

print("RAG chatbot ready!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    response = chat_engine.chat(question)

    print("\nBot:", response)
    print()